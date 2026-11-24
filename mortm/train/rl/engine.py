"""Model-agnostic Unified CARL Training Engine (Odd MIL Discriminator ⇄ Even GRPO)."""

from __future__ import annotations

from collections import deque
from dataclasses import replace as dc_replace
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .domain import (
    CARLExample,
    CriticCalibration,
    RolloutBatch,
    StepLogProbsKL,
    UnifiedEvenConfig,
    UnifiedOddConfig,
)
from .interfaces import (
    CARLActorAdapter,
    CARLCriticAdapter,
    CARLDataSamplePool,
    GuardrailSuite,
)
from .optim import MasterWeightAdamW
from .q_table import QTable, accuracy_by_source, auc_by_source, mix_batch


def compute_discriminator_reward(
    disc_logits: Tensor,
    span: float = 8.0,
    calibration: Optional[CriticCalibration] = None,
) -> Tensor:
    """Linearly map Critic logits (`+` = AI, `-` = Human) to `[0.0, 1.0]` without GAN saturation."""
    if calibration is not None:
        return calibration.score_to_reward(disc_logits)
    if span <= 0.0:
        raise ValueError(f"disc_span must be positive, got {span}.")
    return CriticCalibration(center=0.0, span=float(span)).score_to_reward(disc_logits)


def compute_grpo_advantages(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    """Group-standardize scalar rewards `A_i = (r_i - mean_g) / (std_g + eps)`."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    grouped = rewards.float().view(-1, group_size)
    mean_g = grouped.mean(dim=1, keepdim=True)
    if group_size == 1:
        return torch.zeros_like(rewards, dtype=torch.float32)
    std_g = grouped.std(dim=1, keepdim=True, unbiased=True)
    return ((grouped - mean_g) / (std_g + float(eps))).reshape(-1)


class ClippedGRPOLoss(nn.Module):
    """Clipped policy-gradient surrogate + exact conditional KL regularization."""

    def __init__(self, clip_eps: float = 0.2, kl_coef: float = 0.05) -> None:
        super().__init__()
        self.clip_eps = float(clip_eps)
        self.kl_coef = float(kl_coef)

    def forward(
        self,
        step_stats: StepLogProbsKL,
        old_action_logprobs: Tensor,
        advantages: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mask = step_stats.action_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)

        ratio = torch.exp(step_stats.action_logprobs - old_action_logprobs)
        adv = advantages.to(device=ratio.device, dtype=ratio.dtype).unsqueeze(1)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        pg_tok = -torch.minimum(unclipped, clipped)

        pg_seq = (pg_tok * mask).sum(dim=1) / denom
        kl_seq = (step_stats.exact_kl * mask).sum(dim=1) / denom
        loss = (pg_seq + self.kl_coef * kl_seq).mean()
        return loss, pg_seq.mean().detach(), kl_seq.mean().detach()

    def forward_with_diagnostics(
        self,
        step_stats: StepLogProbsKL,
        old_action_logprobs: Tensor,
        advantages: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mask = step_stats.action_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)

        ratio = torch.exp((step_stats.action_logprobs - old_action_logprobs).clamp(-20.0, 20.0))
        adv = advantages.to(device=ratio.device, dtype=ratio.dtype).unsqueeze(1)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        pg_tok = -torch.minimum(unclipped, clipped)

        is_clipped = ((ratio - 1.0).abs() > self.clip_eps).float() * mask
        clip_frac = (is_clipped.sum(dim=1) / denom).mean().detach()

        pg_seq = (pg_tok * mask).sum(dim=1) / denom
        kl_seq = (step_stats.exact_kl * mask).sum(dim=1) / denom
        loss = (pg_seq + self.kl_coef * kl_seq).mean()
        return loss, pg_seq.mean().detach(), kl_seq.mean().detach(), clip_frac


class OddRoundEngine:
    """Trains the Critic (`CARLCriticAdapter`) with `QTable` experience replay and AUC slope stopping."""

    def __init__(
        self,
        actor: CARLActorAdapter,
        critic: CARLCriticAdapter,
        qtable: QTable,
        cfg: UnifiedOddConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._actor = actor
        self._critic = critic
        self._qtable = qtable
        self._cfg = cfg
        self._rng = rng if rng is not None else np.random.default_rng()

    @torch.no_grad()
    def refresh_gem_pool(
        self,
        ana_pool: CARLDataSamplePool,
        quiet: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        count = min(self._cfg.pool_size, len(ana_pool))
        examples = ana_pool.sample(count, self._rng)
        self._actor.freeze_all()
        roll = self._actor.rollout_group(
            examples=examples,
            group_size=1,
            sampling=self._cfg.sampling,
            chunk_size=self._cfg.gen_chunk,
        )
        valid_items: List[Dict[str, Any]] = []
        dropped_empty = 0
        for c_seq in roll.critic_sequences:
            if c_seq is None or c_seq.size == 0:
                dropped_empty += 1
                continue
            valid_items.append({"seq": c_seq})

        stats = {"kept": len(valid_items), "total": len(examples), "empty": dropped_empty}
        if not quiet:
            print(
                f"  [Odd/GemPool] {stats['kept']}/{stats['total']} AI continuations retained "
                f"(dropped empty={dropped_empty})",
                flush=True,
            )
        return valid_items, stats

    def _sample_human_dicts(self, ana_pool: CARLDataSamplePool, count: int) -> List[Dict[str, Any]]:
        sampled = ana_pool.sample(count, self._rng)
        return [{"seq": ex.human_full_tokens} for ex in sampled]

    def run(
        self,
        ana_pool: CARLDataSamplePool,
        round_no: int = 1,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self._cfg
        self._actor.freeze_all()
        param_groups = self._critic.mark_trainable()
        total_trainable = sum(param_groups.values())
        if total_trainable <= 0:
            raise RuntimeError("OddRoundEngine: Critic reported 0 trainable parameters.")
        group_desc = " / ".join(f"{k}={v:,}" for k, v in param_groups.items())
        print(f"[R{round_no} Odd] Critic trainable params: {total_trainable:,} ({group_desc})", flush=True)

        optimizer = MasterWeightAdamW(
            [p for p in self._critic.critic_module.parameters() if p.requires_grad],
            lr=cfg.lr,
        )
        self._critic.set_train_mode(True)

        gem_pool, pool_stats = self.refresh_gem_pool(ana_pool)
        if not gem_pool:
            raise RuntimeError(
                f"OddRoundEngine: Actor produced 0 valid continuations ({pool_stats}). "
                "Actor weights may have collapsed."
            )
        self._qtable.add(gem_pool)

        hist = {k: deque(maxlen=cfg.eval_window) for k in ("gem", "human", "qtable", "overall")}
        raw_buf: deque[Tuple[np.ndarray, np.ndarray, List[str]]] = deque(maxlen=cfg.eval_window)
        auc_hist: deque[Tuple[int, float]] = deque(maxlen=cfg.auc_slope_window)

        reason: Optional[str] = None
        best_auc = -1.0
        stale = 0
        slope = float("nan")
        last_auc: Dict[str, float] = {}
        cur_cal = CriticCalibration(center=0.0, span=cfg.fallback_disc_span, calibrated=False)
        z_pass = cur_cal.reward_to_logit_threshold(cfg.pass_reward_target)
        refresh_interval = max(1, cfg.pool_size // max(1, cfg.batch_size))
        non_finite_streak = 0

        step = 0
        for step in range(1, cfg.max_steps + 1):
            if step % refresh_interval == 0:
                gem_pool, pool_stats = self.refresh_gem_pool(ana_pool)
                if not gem_pool:
                    raise RuntimeError(
                        f"OddRoundEngine: gem_pool became empty at step {step} ({pool_stats})."
                    )
                self._qtable.add(gem_pool)

            # Sample exact human share `max(1, cfg.batch_size // 2)` consumed by `mix_batch`.
            human_share = max(1, cfg.batch_size // 2)
            human_candidates = self._sample_human_dicts(ana_pool, human_share)
            items, is_ai, source = mix_batch(
                human_candidates, gem_pool, self._qtable, cfg.batch_size, self._rng
            )
            seqs = [it["seq"] for it in items]
            targets = torch.as_tensor(is_ai, dtype=torch.float32, device=self._critic.device)

            disc_logits = self._critic.score_disc_logits(
                seqs, chunk_size=cfg.critic_chunk, grad_enabled=True
            )
            bce_loss = F.binary_cross_entropy_with_logits(disc_logits.float(), targets)
            if cfg.logit_reg_coef > 0.0:
                reg_loss = float(cfg.logit_reg_coef) * (disc_logits.float() ** 2).mean()
                loss = bce_loss + reg_loss
            else:
                loss = bce_loss
            if not bool(torch.isfinite(loss).all().item()):
                optimizer.zero_grad()
                non_finite_streak += 1
                if non_finite_streak >= 5:
                    raise RuntimeError(
                        f"OddRoundEngine encountered {non_finite_streak} consecutive non-finite BCE losses."
                    )
                continue

            loss.backward()
            stepped = optimizer.step_and_zero_grad(grad_clip=cfg.grad_clip)
            if not stepped:
                non_finite_streak += 1
                if non_finite_streak >= 5:
                    raise RuntimeError(
                        f"OddRoundEngine encountered {non_finite_streak} consecutive non-finite gradient steps."
                    )
                continue
            non_finite_streak = 0

            logits_np = disc_logits.detach().float().cpu().numpy()
            raw_buf.append((logits_np, np.asarray(is_ai, dtype=np.float32), list(source)))

            # Compute buffered ROC-AUC, logit calibration & threshold-aligned accuracy over `eval_window`.
            effective_eval_win = min(cfg.eval_window, cfg.max_steps)
            buf_acc: Dict[str, float] = {}
            if len(raw_buf) >= 1 and (step % max(1, cfg.log_every) == 0 or len(raw_buf) >= effective_eval_win):
                sc = np.concatenate([r[0] for r in raw_buf])
                ya = np.concatenate([r[1] for r in raw_buf])
                sr = [x for r in raw_buf for x in r[2]]
                last_auc = auc_by_source(sc, ya, sr)
                lh_buf = last_auc.get("logit_mean_human", float("nan"))
                lg_buf = last_auc.get("logit_mean_gem", float("nan"))
                cur_cal = CriticCalibration.from_odd_stats(
                    lh_buf,
                    lg_buf,
                    pass_reward=cfg.pass_reward_target,
                    fallback_span=cfg.fallback_disc_span,
                )
                z_pass = cur_cal.reward_to_logit_threshold(cfg.pass_reward_target)
                buf_acc = accuracy_by_source(sc, ya, sr, logit_threshold=z_pass)
                zero_acc = accuracy_by_source(sc, ya, sr, logit_threshold=0.0)

                a_gem = last_auc.get("auc_gem", float("nan"))
                if math.isfinite(a_gem):
                    best_auc = max(best_auc, a_gem)
                    if len(raw_buf) >= effective_eval_win and step % max(1, cfg.log_every) == 0:
                        auc_hist.append((step, a_gem))
                        if len(auc_hist) == cfg.auc_slope_window:
                            xs = np.array([h[0] for h in auc_hist], dtype=np.float64)
                            ys = np.array([h[1] for h in auc_hist], dtype=np.float64)
                            slope = float(np.polyfit(xs, ys, 1)[0] * 1000.0)
                            stale = stale + 1 if slope < cfg.auc_min_slope else 0

                        hum_pass_ok = buf_acc.get("human", 0.0) >= cfg.min_class_acc
                        gem_fail_ok = buf_acc.get("gem", 0.0) >= cfg.min_class_acc
                        centered_ok = (
                            zero_acc.get("human", 0.0) >= 0.50 and zero_acc.get("gem", 0.0) >= 0.50
                        )
                        if (
                            a_gem >= cfg.auc_threshold
                            and cur_cal.calibrated
                            and hum_pass_ok
                            and gem_fail_ok
                            and centered_ok
                        ):
                            reason = (
                                f"discriminator_ready: AUC(human vs gem) {a_gem:.4f} >= {cfg.auc_threshold} "
                                f"& acc@z={z_pass:+.2f} (H={buf_acc.get('human', 0.0):.3f}, "
                                f"G={buf_acc.get('gem', 0.0):.3f} >= {cfg.min_class_acc})"
                            )
                        elif step >= cfg.auc_min_steps and stale >= cfg.auc_patience:
                            reason = (
                                f"auc_converged: slope {slope:+.4f}/1k < {cfg.auc_min_slope} "
                                f"for {stale} checks (best={best_auc:.4f}, cur={a_gem:.4f})"
                            )

            acc = accuracy_by_source(logits_np, is_ai, source, logit_threshold=z_pass)
            for k in hist:
                if k in acc:
                    hist[k].append(acc[k])

            step_log: Dict[str, Any] = {
                "odd/loss": float(loss.detach().item()),
                "odd/bce_loss": float(bce_loss.detach().item()),
                "odd/z_pass": z_pass,
                "odd/step": step,
                "odd/qtable_size": len(self._qtable),
                "odd/auc_best": best_auc if best_auc >= 0.0 else float("nan"),
                "odd/auc_stale": stale,
                **{f"odd/acc_{k}": v for k, v in acc.items()},
                **{f"odd/{k}": v for k, v in last_auc.items()},
            }
            if on_step is not None:
                on_step(step_log)

            if step % max(1, cfg.log_every) == 0:
                a_gem = last_auc.get("auc_gem", float("nan"))
                lh = last_auc.get("logit_mean_human", float("nan"))
                lg = last_auc.get("logit_mean_gem", float("nan"))
                auc_str = f"{a_gem:.4f}" if math.isfinite(a_gem) else "n/a"
                best_str = f"{best_auc:.4f}" if best_auc >= 0.0 else "n/a"
                slope_str = (
                    f"{slope:+.4f}/1k"
                    if math.isfinite(slope)
                    else f"warmup({len(auc_hist)}/{cfg.auc_slope_window})"
                )
                win_tag = "" if len(raw_buf) >= effective_eval_win else f" [buf {len(raw_buf)}/{effective_eval_win}]"
                h_acc_w = buf_acc.get("human", float("nan"))
                g_acc_w = buf_acc.get("gem", float("nan"))
                print(
                    f"[R{round_no} Odd] step {step:5d}/{cfg.max_steps} | "
                    f"bce={float(bce_loss.detach().item()):.4f} | "
                    f"AUC(gem)={auc_str}{win_tag} (best={best_str}, slope={slope_str}, stale={stale}) | "
                    f"logit(H={lh:+.2f}, G={lg:+.2f}, z_pass={z_pass:+.2f}) | "
                    f"acc(H={h_acc_w:.2f}, G={g_acc_w:.2f}) | Q={len(self._qtable)}",
                    flush=True,
                )
            if reason is not None:
                break

        if raw_buf:
            sc_f = np.concatenate([r[0] for r in raw_buf])
            ya_f = np.concatenate([r[1] for r in raw_buf])
            sr_f = [x for r in raw_buf for x in r[2]]
            final_stats = auc_by_source(sc_f, ya_f, sr_f)
            final_cal = CriticCalibration.from_odd_stats(
                final_stats.get("logit_mean_human", float("nan")),
                final_stats.get("logit_mean_gem", float("nan")),
                pass_reward=cfg.pass_reward_target,
                fallback_span=cfg.fallback_disc_span,
            )
        else:
            final_cal = cur_cal

        qstats = self._qtable.commit()
        print(
            f"[R{round_no} Odd] Done ({reason or 'max_steps'}) | "
            f"Calibrated={final_cal.calibrated} (center={final_cal.center:+.3f}, span={final_cal.span:.3f}) | "
            f"QTable: {qstats['before']} -> -{qstats['forgot']} +{qstats['merged']} = {qstats['size']}",
            flush=True,
        )
        return {
            "steps": step,
            "reason": reason,
            "qtable": qstats,
            "best_auc": best_auc,
            "calibration": final_cal.to_dict(),
        }


class EvenRoundEngine:
    """Updates the Actor (`CARLActorAdapter`) via Clipped GRPO + Exact KL + Guardrails."""

    def __init__(
        self,
        actor: CARLActorAdapter,
        base_actor: CARLActorAdapter,
        critic: CARLCriticAdapter,
        guardrail_suite: GuardrailSuite[Any],
        qtable: QTable,
        cfg: UnifiedEvenConfig,
        rng: Optional[np.random.Generator] = None,
        calibration: Optional[CriticCalibration] = None,
    ) -> None:
        self._actor = actor
        self._base_actor = base_actor
        self._critic = critic
        self._guardrails = guardrail_suite
        self._qtable = qtable
        self._cfg = cfg
        self._rng = rng if rng is not None else np.random.default_rng()
        self._calibration = calibration
        self._loss_fn = ClippedGRPOLoss(clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef)

    def _effective_calibration(self) -> CriticCalibration:
        if (
            self._cfg.calibrate_reward
            and self._calibration is not None
            and self._calibration.calibrated
        ):
            return self._calibration
        return CriticCalibration(center=0.0, span=self._cfg.disc_span, calibrated=False)

    def _compute_rewards(
        self,
        batch: RolloutBatch,
    ) -> Tuple[Tensor, Dict[str, float]]:
        cfg = self._cfg
        cal = self._effective_calibration()
        guard_eval = self._guardrails.evaluate(batch)
        is_empty = guard_eval.is_empty

        # Score non-empty critic sequences in one batched forward pass.
        n = len(batch)
        disc_rewards = torch.zeros((n,), dtype=torch.float32)
        mean_disc_logit = float("nan")
        valid_indices = [
            idx
            for idx, (emp, c_seq) in enumerate(zip(is_empty.tolist(), batch.critic_sequences))
            if (not emp) and (c_seq is not None) and (c_seq.size > 0)
        ]
        if valid_indices:
            valid_seqs = [batch.critic_sequences[idx] for idx in valid_indices if batch.critic_sequences[idx] is not None]
            disc_logits = self._critic.score_disc_logits(
                valid_seqs, chunk_size=cfg.critic_chunk, grad_enabled=False
            ).cpu()
            mean_disc_logit = float(disc_logits.float().mean().item())
            valid_r_disc = cfg.disc_weight * compute_discriminator_reward(
                disc_logits, span=cfg.disc_span, calibration=cal
            )
            for out_pos, batch_idx in enumerate(valid_indices):
                disc_rewards[batch_idx] = valid_r_disc[out_pos]
        else:
            is_empty = torch.ones((n,), dtype=torch.bool)

        combined = disc_rewards + guard_eval.penalties
        empty_tensor = torch.full((n,), float(cfg.empty_penalty), dtype=torch.float32)
        rewards = torch.where(is_empty, empty_tensor, combined)

        valid_mask = ~is_empty
        mean_disc = (
            float(disc_rewards[valid_mask].mean().item()) if bool(valid_mask.any()) else 0.0
        )
        metrics: Dict[str, float] = {
            "disc": mean_disc,
            "disc_logit": mean_disc_logit,
            "pen_total": float(guard_eval.penalties[valid_mask].mean().item())
            if bool(valid_mask.any())
            else 0.0,
            **dict(guard_eval.metrics),
        }
        return rewards, metrics

    @staticmethod
    def _materialize_old_logprobs(
        sub_old_lps: Optional[Sequence[np.ndarray]],
        step_stats: StepLogProbsKL,
    ) -> Tensor:
        new_lp = step_stats.action_logprobs
        if sub_old_lps is None:
            return new_lp.detach()
        bsz, t_act = new_lp.shape
        if all(arr.ndim == 1 and arr.shape[0] == t_act for arr in sub_old_lps):
            stacked = np.stack(sub_old_lps, axis=0)
            return torch.as_tensor(stacked, dtype=new_lp.dtype, device=new_lp.device)
        old_t = new_lp.detach().clone()
        for row_i, arr in enumerate(sub_old_lps):
            arr_t = torch.as_tensor(arr.reshape(-1), dtype=new_lp.dtype, device=new_lp.device)
            mask_row = step_stats.action_mask[row_i].bool()
            if int(mask_row.sum().item()) == arr_t.numel():
                old_t[row_i, mask_row] = arr_t
            else:
                copy_n = min(t_act, arr_t.numel())
                old_t[row_i, :copy_n] = arr_t[:copy_n]
        return old_t

    @torch.no_grad()
    def _precompute_fallback_old_logprobs(self, batch: RolloutBatch) -> Tuple[np.ndarray, ...]:
        total_n = len(batch)
        chunk_sz = max(1, int(self._cfg.train_chunk))
        parts: List[np.ndarray] = []
        for start in range(0, total_n, chunk_sz):
            stop = min(start + chunk_sz, total_n)
            st = self._actor.forward_logprobs_and_exact_kl(
                base_actor=self._base_actor,
                examples=batch.examples[start:stop],
                generated_sequences=batch.generated_sequences[start:stop],
                sampling=self._cfg.sampling,
            )
            lp_np = st.action_logprobs.detach().float().cpu().numpy()
            for row_i in range(lp_np.shape[0]):
                parts.append(lp_np[row_i])
        return tuple(parts)

    def _step_backward(
        self,
        batch: RolloutBatch,
        rewards: Tensor,
        override_old_lps: Optional[Tuple[np.ndarray, ...]] = None,
    ) -> Tuple[float, float, float, float]:
        advantages = compute_grpo_advantages(rewards, batch.group_size, eps=self._cfg.adv_eps)
        total_n = len(batch)
        chunk_sz = max(1, int(self._cfg.train_chunk))
        acc_loss = 0.0
        acc_pg = 0.0
        acc_kl = 0.0
        acc_clip = 0.0
        old_source = batch.old_action_logprobs if batch.old_action_logprobs is not None else override_old_lps

        for start in range(0, total_n, chunk_sz):
            stop = min(start + chunk_sz, total_n)
            weight = float(stop - start) / float(total_n)
            sub_ex = batch.examples[start:stop]
            sub_gen = batch.generated_sequences[start:stop]
            sub_adv = advantages[start:stop]
            sub_old = old_source[start:stop] if old_source is not None else None

            step_stats = self._actor.forward_logprobs_and_exact_kl(
                base_actor=self._base_actor,
                examples=sub_ex,
                generated_sequences=sub_gen,
                sampling=self._cfg.sampling,
            )
            old_logp = self._materialize_old_logprobs(sub_old, step_stats)
            loss, pg_mean, kl_mean, clip_frac = self._loss_fn.forward_with_diagnostics(
                step_stats, old_logp, sub_adv
            )
            (loss * weight).backward()

            acc_loss += float(loss.detach().item()) * weight
            acc_pg += float(pg_mean.item()) * weight
            acc_kl += float(kl_mean.item()) * weight
            acc_clip += float(clip_frac.item()) * weight

        return acc_loss, acc_pg, acc_kl, acc_clip

    def should_stop(self, mean_reward: float, mean_kl: float, mean_disc: float) -> Optional[str]:
        if mean_kl > self._cfg.kl_stop and mean_disc > self._cfg.disc_stop:
            return (
                f"reward_hacking: KL {mean_kl:.3f} > {self._cfg.kl_stop} "
                f"and R_disc {mean_disc:.3f} > {self._cfg.disc_stop}"
            )
        if mean_reward > self._cfg.reward_stop:
            return f"achieved: reward {mean_reward:.3f} > {self._cfg.reward_stop}"
        return None

    def run(
        self,
        gem_pool: CARLDataSamplePool,
        round_no: int = 2,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self._cfg
        self._base_actor.freeze_all()
        self._critic.set_train_mode(False)

        n_train, n_total = self._actor.unfreeze_for_grpo()
        if n_train <= 0:
            raise RuntimeError("EvenRoundEngine: Actor reported 0 trainable parameters for GRPO.")
        self._actor.prepare_for_grpo_forward()
        pct = 100.0 * float(n_train) / float(max(1, n_total))
        cal = self._effective_calibration()
        z_goal = cal.reward_to_logit_threshold(cfg.reward_stop / max(1e-6, cfg.disc_weight))
        print(
            f"[R{round_no} Even] Actor trainable: {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M ({pct:.1f}%) | "
            f"lr={cfg.lr} | ppo_epochs={cfg.ppo_epochs} | G={cfg.group_size} x P={cfg.prompts_per_step} = "
            f"{cfg.group_size * cfg.prompts_per_step} rollouts/step\n"
            f"[R{round_no} Even] Reward ruler ({'calibrated' if cal.calibrated else 'fixed'}): "
            f"center={cal.center:+.3f}, span={cal.span:.3f}, z_pass<={z_goal:+.3f} for R>={cfg.reward_stop:.3f}",
            flush=True,
        )

        trainable_params = [p for p in self._actor.actor_module.parameters() if p.requires_grad]
        optimizer = MasterWeightAdamW(trainable_params, lr=cfg.lr)
        if optimizer.num_low_precision_tensors > 0:
            print(
                f"[R{round_no} Even] FP32 master weights active for "
                f"{optimizer.num_low_precision_tensors} low-precision (bf16/fp16) tensors.",
                flush=True,
            )

        win = max(1, int(cfg.stop_window))
        mv = {k: deque(maxlen=win) for k in ("reward", "kl", "disc")}
        healthy_snap = self._actor.snapshot_state()
        snap_step = 0

        reason: Optional[str] = None
        last_batch: Optional[RolloutBatch] = None
        step = 0

        while reason is None and step < cfg.max_steps:
            step += 1
            prompts = gem_pool.sample(max(1, cfg.prompts_per_step), self._rng)
            roll = self._actor.rollout_group(
                examples=prompts,
                group_size=cfg.group_size,
                sampling=cfg.sampling,
                chunk_size=cfg.gen_chunk,
            )
            last_batch = roll
            # Re-ensure `.eval()` after rollout in case `generate()` restored `.train()`.
            self._actor.prepare_for_grpo_forward()

            rewards, r_metrics = self._compute_rewards(roll)
            mean_r = float(rewards.mean().item())
            fallback_old_lps = (
                self._precompute_fallback_old_logprobs(roll)
                if (roll.old_action_logprobs is None and cfg.ppo_epochs > 1)
                else None
            )

            loss_val = float("nan")
            pg_val = float("nan")
            kl_val = float("nan")
            clip_val = 0.0
            aborted_epoch = False
            for _ in range(cfg.ppo_epochs):
                loss_val, pg_val, kl_val, clip_val = self._step_backward(
                    roll, rewards, override_old_lps=fallback_old_lps
                )
                if not (
                    math.isfinite(loss_val)
                    and math.isfinite(pg_val)
                    and math.isfinite(kl_val)
                    and math.isfinite(mean_r)
                ):
                    optimizer.zero_grad()
                    reason = (
                        f"non_finite_abort: step={step}, loss={loss_val}, kl={kl_val}, reward={mean_r}"
                    )
                    aborted_epoch = True
                    break

                stepped = optimizer.step_and_zero_grad(grad_clip=cfg.grad_clip)
                if not stepped:
                    reason = f"non_finite_abort: non-finite gradients detected at step={step}"
                    aborted_epoch = True
                    break

            if aborted_epoch:
                break

            mv["reward"].append(mean_r)
            mv["kl"].append(kl_val)
            mv["disc"].append(r_metrics["disc"])
            avg = {k: float(np.mean(v)) for k, v in mv.items()}

            step_log: Dict[str, Any] = {
                "even/step": step,
                "even/reward": mean_r,
                "even/reward_sd": float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
                "even/reward_min": float(rewards.min().item()),
                "even/reward_max": float(rewards.max().item()),
                "even/reward_disc": r_metrics["disc"],
                "even/pen_total": r_metrics["pen_total"],
                "even/kl": kl_val,
                "even/pg_loss": pg_val,
                "even/clip_frac": clip_val,
                "even/loss": loss_val,
                "even/reward_ma": avg["reward"],
                "even/kl_ma": avg["kl"],
                "even/disc_ma": avg["disc"],
                **{f"even/{k}": v for k, v in r_metrics.items() if k not in ("disc", "pen_total")},
            }
            if on_step is not None:
                on_step(step_log)

            if step % max(1, cfg.log_every) == 0:
                z_mean = r_metrics.get("disc_logit", float("nan"))
                print(
                    f"[R{round_no} Even] step {step:4d}/{cfg.max_steps} | "
                    f"R={mean_r:+.3f} (ma={avg['reward']:+.3f}/{cfg.reward_stop:+.2f}) | "
                    f"R_disc={r_metrics['disc']:+.3f} (z={z_mean:+.2f}) | pen={r_metrics['pen_total']:+.3f} | "
                    f"empty={100.0 * r_metrics.get('empty_rate', 0.0):.1f}% | "
                    f"KL={kl_val:.4f} (ma={avg['kl']:.4f}/{cfg.kl_stop:.2f}) | clip={100.0 * clip_val:.1f}%",
                    flush=True,
                )

            if len(mv["kl"]) == win:
                reason = self.should_stop(avg["reward"], avg["kl"], avg["disc"])

            if reason is None and step % max(1, cfg.snapshot_every) == 0 and avg["kl"] <= cfg.kl_stop:
                healthy_snap = self._actor.snapshot_state()
                snap_step = step

        rolled_back = False
        injected_hacked = 0
        if reason is not None and reason.startswith(("reward_hacking", "non_finite_abort")):
            # Inject hacked sequences into `qtable.table` only on genuine reward hacking (not NaN abort).
            if reason.startswith("reward_hacking") and last_batch is not None:
                hacked_items = [
                    {"seq": c_seq}
                    for c_seq in last_batch.critic_sequences
                    if c_seq is not None and c_seq.size > 0
                ]
                injected_hacked = self._qtable.inject_committed(hacked_items)
            self._actor.restore_state(healthy_snap)
            optimizer.sync_from_model()
            rolled_back = True
            print(
                f"[R{round_no} Even] Rollback ({reason}) -> restored Actor snapshot from step {snap_step} "
                f"(injected {injected_hacked} hacked sequences directly into QTable).",
                flush=True,
            )

        print(f"[R{round_no} Even] Done ({reason or 'max_steps'}).", flush=True)
        return {
            "steps": step,
            "reason": reason,
            "rolled_back": rolled_back,
            "snap_step": snap_step,
            "injected_hacked": injected_hacked,
        }


class UnifiedCARLDriver:
    """Top-level orchestrator alternating Odd (Critic) and Even (GRPO Actor) rounds."""

    def __init__(
        self,
        actor: CARLActorAdapter,
        base_actor: CARLActorAdapter,
        critic: CARLCriticAdapter,
        ana_pool: CARLDataSamplePool,
        gem_pool: CARLDataSamplePool,
        guardrail_suite: GuardrailSuite[Any],
        out_dir: str | Path,
        odd_cfg: Optional[UnifiedOddConfig] = None,
        even_cfg: Optional[UnifiedEvenConfig] = None,
        q_max: int = 3000,
        seed: int = 42,
        ckpt_prefix: str = "CARL",
        keep_round_ckpts: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "MORTM-CARL",
        run_name: Optional[str] = None,
        eval_hook: Optional[Callable[[int, CARLActorAdapter], Dict[str, float]]] = None,
    ) -> None:
        self._actor = actor
        self._base_actor = base_actor
        self._critic = critic
        self._ana_pool = ana_pool
        self._gem_pool = gem_pool
        self._guardrails = guardrail_suite
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._odd_cfg = odd_cfg or UnifiedOddConfig()
        self._even_cfg = even_cfg or UnifiedEvenConfig()
        # Synchronize Odd-round pass-threshold target with Even-round `reward_stop` unless explicitly overridden.
        linked_target = min(
            0.92,
            max(0.50, float(self._even_cfg.reward_stop) / max(1e-6, float(self._even_cfg.disc_weight))),
        )
        if abs(self._odd_cfg.pass_reward_target - 0.625) < 1e-9 and abs(linked_target - 0.625) > 1e-9:
            self._odd_cfg = dc_replace(
                self._odd_cfg,
                pass_reward_target=linked_target,
                fallback_disc_span=self._even_cfg.disc_span,
            )
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._qtable = QTable(q_max=q_max, rng=self._rng)
        self._calibration: Optional[CriticCalibration] = None
        self._ckpt_prefix = str(ckpt_prefix)
        self._keep_round_ckpts = bool(keep_round_ckpts)
        self._eval_hook = eval_hook
        self.history: List[Dict[str, Any]] = []
        self._step_global = 0
        self._wandb = self._init_wandb(use_wandb, wandb_project, run_name, q_max, seed)

    @property
    def qtable(self) -> QTable:
        return self._qtable

    @property
    def calibration(self) -> Optional[CriticCalibration]:
        return self._calibration

    def _init_wandb(
        self,
        enabled: bool,
        project: str,
        run_name: Optional[str],
        q_max: int,
        seed: int,
    ) -> Any:
        if not enabled:
            return None
        try:
            import wandb

            wandb.init(
                project=project,
                name=run_name,
                config={
                    "q_max": q_max,
                    "seed": seed,
                    "n_ana": len(self._ana_pool),
                    "n_gem": len(self._gem_pool),
                    "odd": self._odd_cfg.to_dict(),
                    "even": self._even_cfg.to_dict(),
                },
                dir=str(self._out_dir),
            )
            return wandb
        except Exception as exc:
            print(f"[CARL] W&B disabled ({exc})", flush=True)
            return None

    def _log(self, payload: Dict[str, Any]) -> None:
        self._step_global += 1
        if self._wandb is not None:
            self._wandb.log(payload, step=self._step_global)

    def resume(
        self,
        actor_ckpt: Optional[str | Path] = None,
        critic_ckpt: Optional[str | Path] = None,
        qtable_ckpt: Optional[str | Path] = None,
    ) -> None:
        if actor_ckpt is not None and str(actor_ckpt).strip():
            if not os.path.isfile(actor_ckpt):
                raise FileNotFoundError(f"Actor resume checkpoint not found: {actor_ckpt}")
            sd = torch.load(actor_ckpt, map_location="cpu", weights_only=False)
            self._actor.load_checkpoint_state(sd)
            print(f"[CARL] Resumed Actor from {actor_ckpt}", flush=True)

        if qtable_ckpt is not None and str(qtable_ckpt).strip():
            if not os.path.isfile(qtable_ckpt):
                raise FileNotFoundError(f"QTable resume checkpoint not found: {qtable_ckpt}")
            q_obj = torch.load(qtable_ckpt, map_location="cpu", weights_only=False)
            self._qtable.table = list(q_obj.get("table", []))
            self._qtable.round = int(q_obj.get("round", 0))
            if isinstance(q_obj.get("calibration"), dict):
                self._calibration = CriticCalibration.from_dict(q_obj["calibration"])
            print(f"[CARL] Resumed QTable ({len(self._qtable)} items) from {qtable_ckpt}", flush=True)

        if critic_ckpt is not None and str(critic_ckpt).strip():
            if not os.path.isfile(critic_ckpt):
                raise FileNotFoundError(f"Critic resume checkpoint not found: {critic_ckpt}")
            c_sd = torch.load(critic_ckpt, map_location="cpu", weights_only=False)
            self._critic.load_compact_state(c_sd)
            print(f"[CARL] Resumed Critic from {critic_ckpt}", flush=True)

    def prefill_qtable(self, target_size: Optional[int] = None) -> None:
        goal = int(target_size or self._qtable.q_max)
        if len(self._qtable) >= goal:
            return
        print(f"[CARL] Prefilling QTable with current Actor rollouts (goal={goal})...", flush=True)
        odd = OddRoundEngine(
            actor=self._actor,
            critic=self._critic,
            qtable=self._qtable,
            cfg=self._odd_cfg,
            rng=self._rng,
        )
        while len(self._qtable.new) < goal:
            items, _ = odd.refresh_gem_pool(self._ana_pool, quiet=True)
            added = self._qtable.add(items)
            if added == 0:
                break
        st = self._qtable.commit()
        q_path = self._out_dir / "qtable.prefill.pt"
        torch.save({"table": self._qtable.table, "round": self._qtable.round}, q_path)
        print(f"[CARL] QTable prefilled ({st['size']} items) -> {q_path}", flush=True)

    def _save_qtable(self, suffix: str) -> Path:
        q_payload: Dict[str, Any] = {"table": self._qtable.table, "round": self._qtable.round}
        if self._calibration is not None:
            q_payload["calibration"] = self._calibration.to_dict()
        q_path = self._out_dir / f"qtable{suffix}.pt"
        torch.save(q_payload, q_path)
        latest_q = self._out_dir / "qtable_latest.pt"
        if latest_q != q_path:
            torch.save(q_payload, latest_q)
        return q_path

    def _save_round(self, kind: str, round_no: int) -> Path:
        suffix = f".r{round_no}" if self._keep_round_ckpts else ""
        q_path = self._save_qtable(suffix)
        if kind == "odd":
            c_state = self._critic.export_compact_state()
            c_path = self._out_dir / f"{self._ckpt_prefix}-ana.carl{suffix}.pth"
            torch.save(c_state, c_path)
            latest_c = self._out_dir / f"{self._ckpt_prefix}-ana.carl_latest.pth"
            if latest_c != c_path:
                torch.save(c_state, latest_c)
            print(f"[CARL] Saved Critic -> {c_path} & QTable -> {q_path}", flush=True)
            return c_path

        g_path = self._out_dir / f"{self._ckpt_prefix}-gem.carl{suffix}.pth"
        state = self._actor.export_checkpoint_state()
        torch.save(state, g_path)
        latest_g = self._out_dir / f"{self._ckpt_prefix}-gem.carl_latest.pth"
        if latest_g != g_path:
            torch.save(state, latest_g)
        print(f"[CARL] Saved Actor -> {g_path} & QTable -> {q_path}", flush=True)
        return g_path

    def run_odd(self, round_no: int) -> Dict[str, Any]:
        odd = OddRoundEngine(
            actor=self._actor,
            critic=self._critic,
            qtable=self._qtable,
            cfg=self._odd_cfg,
            rng=self._rng,
        )
        out = odd.run(
            self._ana_pool,
            round_no=round_no,
            on_step=lambda d: self._log({**d, "round": round_no}),
        )
        if isinstance(out.get("calibration"), dict):
            self._calibration = CriticCalibration.from_dict(out["calibration"])
        self._save_round("odd", round_no)
        return out

    def run_even(self, round_no: int) -> Dict[str, Any]:
        # Prefill QTable BEFORE unfreezing Actor so `refresh_gem_pool`'s freeze doesn't clobber GRPO grads.
        if len(self._qtable) == 0:
            self.prefill_qtable()
        even = EvenRoundEngine(
            actor=self._actor,
            base_actor=self._base_actor,
            critic=self._critic,
            guardrail_suite=self._guardrails,
            qtable=self._qtable,
            cfg=self._even_cfg,
            rng=self._rng,
            calibration=self._calibration,
        )
        out = even.run(
            self._gem_pool,
            round_no=round_no,
            on_step=lambda d: self._log({**d, "round": round_no}),
        )
        self._save_round("even", round_no)
        if self._eval_hook is not None:
            eval_metrics = self._eval_hook(round_no, self._actor)
            out["eval_metrics"] = eval_metrics
            self._log({f"eval/{k}": v for k, v in eval_metrics.items()} | {"round": round_no})
        return out

    def run(
        self,
        max_total_steps: int = 100000,
        target_hits: int = 3,
        start_round: int = 1,
        max_rounds: int = 100,
    ) -> List[Dict[str, Any]]:
        total_steps = 0
        hits = 0
        rnd = int(start_round)

        while rnd < int(start_round) + int(max_rounds):
            is_odd = (rnd % 2 == 1)
            out = self.run_odd(rnd) if is_odd else self.run_even(rnd)
            kind = "odd" if is_odd else "even"
            total_steps += int(out.get("steps", 0))
            reason = str(out.get("reason") or "")
            if kind == "even" and reason.startswith("achieved"):
                hits += 1
                print(f"[CARL] Target reward achieved ({hits}/{target_hits})!", flush=True)

            rec = {
                "round": rnd,
                "kind": kind,
                "total_steps": total_steps,
                "hits": hits,
                **out,
            }
            self.history.append(rec)
            self._log(
                {
                    "round_end/round": rnd,
                    "round_end/steps": out.get("steps", 0),
                    "round_end/total_steps": total_steps,
                    "round_end/hits": hits,
                    "round_end/qtable_size": len(self._qtable),
                }
            )
            if hits >= target_hits or total_steps >= max_total_steps:
                break
            rnd += 1

        if self._wandb is not None:
            self._wandb.finish()
        return self.history
