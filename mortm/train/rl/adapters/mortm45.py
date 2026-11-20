"""MORTM4.5 (CONST-MIDI + Any-Order SFT) Adapters for Unified CARL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from mortm.utils.key_from_tokens import tokens_to_notes
from ..carl_driver import (
    _UNMERGED_MARK,
    _is_legacy_merged,
    load_ana,
    load_gem,
    unfreeze_attention_and_gate,
)
from ..carl_even import (
    _has_note,
    density_violation,
    exact_kl,
    key_pitch_classes,
    key_violation,
    notes_to_roll,
    parse_meta,
    roll_similarity,
    similarity_reward,
    structural_reward,
)
from ..carl_odd import AnalysisMORTM, build_analysis_sample, generate_chunked, inst_names_of
from ..domain import CARLExample, RolloutBatch, StepLogProbsKL, UnifiedSamplingConfig
from ..gen_sft_pool import fold_to_const, load_gen_sft_pool
from ..interfaces import (
    CARLActorAdapter,
    CARLCriticAdapter,
    CARLDataRepository,
    CARLDataSamplePool,
    GuardrailContextBuilder,
    GuardrailRule,
    GuardrailSuite,
    InMemorySamplePool,
)


def _pad_1d_sequences(seqs: Sequence[Sequence[int] | np.ndarray], device: torch.device) -> Tuple[Tensor, Tensor]:
    max_len = max(len(s) for s in seqs)
    out = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(seqs):
        arr = torch.as_tensor(np.asarray(seq, dtype=np.int64), dtype=torch.long, device=device)
        out[idx, : arr.numel()] = arr
    return out, (out != 0)


def _autocast_full_logprobs(model: nn.Module, x: Tensor, padding_mask: Tensor) -> Tensor:
    dev_type = x.device.type
    use_bf16 = dev_type in ("cuda", "mps")
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model.forward(x, padding_mask=padding_mask, is_causal=True, is_save_cache=False)
    return torch.log_softmax(logits[:, :-1, :].float(), dim=-1)


class Mortm45GenSftRepository(CARLDataRepository):
    """Loads MORTM4.5 generation SFT `.npz` records and splits disjointly 50:50 (`ana` vs `gem`)."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer: Any,
        max_blank_ratio: float = 0.5,
    ) -> None:
        self._manifest_path = str(manifest_path)
        self._tokenizer = tokenizer
        self._max_blank_ratio = float(max_blank_ratio)

    def load_disjoint_pools(
        self,
        limit: Optional[int] = 40000,
        seed: int = 42,
    ) -> Tuple[CARLDataSamplePool, CARLDataSamplePool]:
        rng = np.random.default_rng(seed)
        eff_limit = int(limit) if limit is not None else 40000
        raw_pool = load_gen_sft_pool(
            self._manifest_path,
            self._tokenizer,
            limit=eff_limit,
            max_blank_ratio=self._max_blank_ratio,
            rng=rng,
        )
        examples: List[CARLExample] = []
        for idx, rec in enumerate(raw_pool):
            examples.append(
                CARLExample(
                    example_id=f"m45_{idx:06d}_{rec.get('task', 'gen')}",
                    song_id=idx,
                    prompt_tokens=np.asarray(rec["prompt"], dtype=np.int64),
                    human_full_tokens=np.asarray(rec["seq"], dtype=np.int64),
                    aux_metadata=rec,
                )
            )
        cut = len(examples) // 2
        return InMemorySamplePool(examples[:cut]), InMemorySamplePool(examples[cut:])


class Mortm45ActorAdapter(CARLActorAdapter):
    """Actor adapter for `MORTM` (MORTM4.5 `CONST-MIDI` Meta/Past/Future/Infill generator)."""

    def __init__(self, model: nn.Module, tokenizer: Any, device: torch.device) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._te_id = int(tokenizer.get("<TE>"))

    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        ckpt_path: str,
        tokenizer: Any,
        device: torch.device,
        trainable: bool = True,
    ) -> "Mortm45ActorAdapter":
        model = load_gem(config_path, ckpt_path, device, train=trainable)
        return cls(model=model, tokenizer=tokenizer, device=device)

    @property
    def actor_module(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def unfreeze_for_grpo(self) -> Tuple[int, int]:
        return unfreeze_attention_and_gate(self._model)

    @torch.no_grad()
    def rollout_group(
        self,
        examples: Sequence[CARLExample],
        group_size: int,
        sampling: UnifiedSamplingConfig,
        chunk_size: int = 16,
        seed: Optional[int] = None,
    ) -> RolloutBatch:
        expanded_examples: List[CARLExample] = []
        prompts: List[List[int]] = []
        for ex in examples:
            p_list = [int(t) for t in ex.prompt_tokens.tolist()]
            for _ in range(group_size):
                expanded_examples.append(ex)
                prompts.append(p_list)

        gens = generate_chunked(
            self._model,
            prompts,
            self._tokenizer,
            self._device,
            sampling.max_measures,
            chunk=max(1, int(chunk_size)),
            p=sampling.top_p,
            temperature=sampling.temperature,
            max_new=sampling.max_new_tokens,
            seed=seed,
        )

        gen_arrays: List[np.ndarray] = []
        critic_arrays: List[Optional[np.ndarray]] = []
        for ex, raw_gen in zip(expanded_examples, gens):
            gen_arr = np.asarray(raw_gen, dtype=np.int64)
            gen_arrays.append(gen_arr)
            body = [int(t) for t in gen_arr.tolist() if int(t) != self._te_id]
            if not body or not _has_note(body, self._tokenizer):
                critic_arrays.append(None)
                continue
            rec = ex.aux_metadata
            if isinstance(rec, dict):
                folded, merged = fold_to_const(rec, body, self._tokenizer)
                critic_arrays.append(np.asarray(folded, dtype=np.int64) if merged else None)
            else:
                const_m = int(self._tokenizer.get("<CONST_M>"))
                tag_end = int(self._tokenizer.get("<TAG_END>"))
                critic_arrays.append(np.asarray([const_m] + body + [tag_end], dtype=np.int64))

        return RolloutBatch(
            examples=tuple(expanded_examples),
            generated_sequences=tuple(gen_arrays),
            critic_sequences=tuple(critic_arrays),
            group_size=group_size,
        )

    def forward_logprobs_and_exact_kl(
        self,
        base_actor: CARLActorAdapter,
        examples: Sequence[CARLExample],
        generated_sequences: Sequence[np.ndarray],
    ) -> StepLogProbsKL:
        seqs: List[List[int]] = []
        resp_masks: List[List[bool]] = []
        for ex, gen in zip(examples, generated_sequences):
            p_list = [int(t) for t in ex.prompt_tokens.tolist()]
            g_list = [int(t) for t in gen.tolist()]
            seqs.append(p_list + g_list)
            resp_masks.append([False] * len(p_list) + [True] * len(g_list))

        x, pad = _pad_1d_sequences(seqs, self._device)
        mask_full = torch.zeros_like(x, dtype=torch.bool)
        for idx, r_m in enumerate(resp_masks):
            mask_full[idx, : len(r_m)] = torch.tensor(r_m, dtype=torch.bool, device=self._device)

        with torch.no_grad():
            logp_base_full = _autocast_full_logprobs(base_actor.actor_module, x, pad)
        logp_new_full = _autocast_full_logprobs(self._model, x, pad)

        tgt = x[:, 1:]
        act_mask = mask_full[:, 1:]
        action_logprobs = logp_new_full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * act_mask.float()
        kl_vals = exact_kl(logp_new_full, logp_base_full) * act_mask.float()
        return StepLogProbsKL(
            action_logprobs=action_logprobs,
            exact_kl=kl_vals,
            action_mask=act_mask,
        )

    def export_checkpoint_state(self) -> Dict[str, Any]:
        was_training = self._model.training
        self._model.train()
        try:
            sd: Dict[str, Any] = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
        finally:
            if not was_training:
                self._model.eval()
        sd[_UNMERGED_MARK] = torch.tensor(1)
        return sd

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        cleaned = dict(state)
        if _is_legacy_merged("", cleaned):
            raise RuntimeError("Legacy merged LoRA checkpoint detected; run fix_merged_ckpt first.")
        cleaned.pop(_UNMERGED_MARK, None)
        self._model.load_state_dict(cleaned)


class Mortm45CriticAdapter(CARLCriticAdapter):
    """Critic adapter wrapping `AnalysisMORTM` (`LoRA` + `MILPool` / `PMA`)."""

    def __init__(self, ana_model: AnalysisMORTM, tokenizer: Any, device: torch.device) -> None:
        self._ana = ana_model
        self._tokenizer = tokenizer
        self._device = device
        self._meta_id = int(tokenizer.get("<META>"))

    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        ckpt_path: str,
        tokenizer: Any,
        device: torch.device,
        pool: str = "mil",
    ) -> "Mortm45CriticAdapter":
        ana = load_ana(config_path, ckpt_path, device, pool=pool)
        return cls(ana_model=ana, tokenizer=tokenizer, device=device)

    @property
    def critic_module(self) -> nn.Module:
        return self._ana

    @property
    def device(self) -> torch.device:
        return self._device

    def mark_trainable(self) -> Dict[str, int]:
        self._ana.mark_trainable()
        groups = {"lora": 0, "pool": 0, "out_head": 0}
        for name, param in self._ana.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                groups["lora"] += param.numel()
            elif name.startswith(("pma", "mil")):
                groups["pool"] += param.numel()
            elif name.startswith("out_head"):
                groups["out_head"] += param.numel()
        if groups["lora"] == 0 or groups["pool"] == 0:
            raise RuntimeError(f"Mortm45CriticAdapter missing trainable groups: {groups}")
        return {k: v for k, v in groups.items() if v > 0}

    def score_disc_logits(
        self,
        critic_sequences: Sequence[np.ndarray],
        chunk_size: Optional[int] = None,
        grad_enabled: bool = False,
    ) -> Tensor:
        samples = [build_analysis_sample(np.asarray(seq, dtype=np.int64), self._tokenizer) for seq in critic_sequences]
        step = len(samples) if chunk_size is None or chunk_size <= 0 else int(chunk_size)
        parts: List[Tensor] = []
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            for start in range(0, len(samples), step):
                part = samples[start : start + step]
                x, pad = _pad_1d_sequences(part, self._device)
                _, disc = self._ana.forward_with_disc(
                    x, pad, is_causal=True, meta_id=self._meta_id, return_logits=False
                )
                parts.append(disc.float())
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def export_compact_state(self) -> Dict[str, Any]:
        was_training = self._ana.training
        self._ana.train()
        try:
            sd: Dict[str, Any] = {
                k: v.detach().clone()
                for k, v in self._ana.state_dict().items()
                if ("lora_" in k) or k.startswith(("pma", "out_head", "mil"))
            }
        finally:
            if not was_training:
                self._ana.eval()
        sd[_UNMERGED_MARK] = torch.tensor(1)
        return sd

    def load_compact_state(self, state: Dict[str, Any]) -> None:
        cleaned = dict(state)
        cleaned.pop(_UNMERGED_MARK, None)
        _, unexpected = self._ana.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected Critic keys: {unexpected[:5]}")


@dataclass(frozen=True)
class Mortm45DecodedContext:
    """Shared decoded representation for MORTM4.5 guardrails (eliminates redundant note parsing)."""

    batch: RolloutBatch
    is_empty: Tensor
    cond_parsed: Tuple[Dict[str, object], ...]
    cond_raw: Tuple[np.ndarray, ...]
    gen_notes: Tuple[list, ...]
    ref_notes: Tuple[Optional[list], ...]


class Mortm45ContextBuilder(GuardrailContextBuilder[Mortm45DecodedContext]):
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def build_context(self, batch: RolloutBatch) -> Mortm45DecodedContext:
        is_empty_flags: List[bool] = []
        cond_parsed: List[Dict[str, object]] = []
        cond_raw_list: List[np.ndarray] = []
        gen_notes_list: List[list] = []
        ref_notes_list: List[Optional[list]] = []

        for ex, gen in zip(batch.examples, batch.generated_sequences):
            rec = ex.aux_metadata if isinstance(ex.aux_metadata, dict) else {}
            c_raw = np.asarray(rec.get("cond_meta", []), dtype=np.int64)
            c_parsed = parse_meta(c_raw, self._tok) if c_raw.size > 0 else {}
            cond_raw_list.append(c_raw)
            cond_parsed.append(c_parsed)

            if not _has_note(gen, self._tok):
                is_empty_flags.append(True)
                gen_notes_list.append([])
                ref_notes_list.append(None)
                continue

            is_empty_flags.append(False)
            gen_notes_list.append(tokens_to_notes(self._tok, gen))
            gt_seq = rec.get("gt")
            ref_notes_list.append(tokens_to_notes(self._tok, gt_seq) if gt_seq is not None else None)

        return Mortm45DecodedContext(
            batch=batch,
            is_empty=torch.tensor(is_empty_flags, dtype=torch.bool),
            cond_parsed=tuple(cond_parsed),
            cond_raw=tuple(cond_raw_list),
            gen_notes=tuple(gen_notes_list),
            ref_notes=tuple(ref_notes_list),
        )


class Mortm45StructureRule(GuardrailRule[Mortm45DecodedContext]):
    def __init__(self, tokenizer: Any, weight: float = 0.5) -> None:
        self._tok = tokenizer
        self._weight = float(weight)

    @property
    def name(self) -> str:
        return "struct"

    def evaluate(self, context: Mortm45DecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        pens: List[float] = []
        m_scores: List[float] = []
        i_scores: List[float] = []
        for idx, gen in enumerate(context.batch.generated_sequences):
            if bool(context.is_empty[idx]) or context.cond_raw[idx].size == 0:
                pens.append(0.0)
                continue
            s_score, sd = structural_reward(gen, context.cond_raw[idx], self._tok)
            pens.append(-self._weight * (1.0 - float(s_score)))
            m_scores.append(float(sd["measure"]))
            i_scores.append(float(sd["inst"]))
        return (
            torch.tensor(pens, dtype=torch.float32),
            {
                "struct_measure": float(np.mean(m_scores)) if m_scores else float("nan"),
                "struct_inst": float(np.mean(i_scores)) if i_scores else float("nan"),
            },
        )


class Mortm45DensityRule(GuardrailRule[Mortm45DecodedContext]):
    def __init__(self, tokenizer: Any, weight: float = 0.25, tolerance: int = 1) -> None:
        self._tok = tokenizer
        self._weight = float(weight)
        self._tolerance = int(tolerance)

    @property
    def name(self) -> str:
        return "density"

    def evaluate(self, context: Mortm45DecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        pens: List[float] = []
        ok_cnt = 0
        tot_cnt = 0
        for idx, gen in enumerate(context.batch.generated_sequences):
            if bool(context.is_empty[idx]) or context.cond_raw[idx].size == 0:
                pens.append(0.0)
                continue
            v_dens, all_ok = density_violation(
                gen,
                context.cond_parsed[idx],
                inst_names_of(context.cond_raw[idx], self._tok),
                self._tok,
                tolerance=self._tolerance,
            )
            pens.append(-self._weight * float(v_dens))
            if all_ok is not None:
                tot_cnt += 1
                ok_cnt += int(all_ok)
        return (
            torch.tensor(pens, dtype=torch.float32),
            {"density_ok": (ok_cnt / tot_cnt) if tot_cnt else float("nan")},
        )


class Mortm45KeyOutlierRule(GuardrailRule[Mortm45DecodedContext]):
    def __init__(
        self,
        tokenizer: Any,
        weight: float = 0.25,
        tau_floor: float = 0.20,
        margin: float = 0.10,
    ) -> None:
        self._tok = tokenizer
        self._weight = float(weight)
        self._tau_floor = float(tau_floor)
        self._margin = float(margin)

    @property
    def name(self) -> str:
        return "key"

    def evaluate(self, context: Mortm45DecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        pens: List[float] = []
        oo_vals: List[float] = []
        for idx in range(len(context.batch)):
            if bool(context.is_empty[idx]):
                pens.append(0.0)
                continue
            pcs = key_pitch_classes(context.cond_parsed[idx].get("key"), self._tok)
            v_key, oo = key_violation(
                context.gen_notes[idx],
                context.ref_notes[idx],
                pcs,
                self._tau_floor,
                self._margin,
            )
            pens.append(-self._weight * float(v_key))
            if oo is not None:
                oo_vals.append(float(oo))
        return (
            torch.tensor(pens, dtype=torch.float32),
            {"oo_key_mean": float(np.mean(oo_vals)) if oo_vals else float("nan")},
        )


class Mortm45SimilarityRule(GuardrailRule[Mortm45DecodedContext]):
    def __init__(self, weight: float = 0.5, center: float = 0.75, sharpness: float = 12.0) -> None:
        self._weight = float(weight)
        self._center = float(center)
        self._sharpness = float(sharpness)

    @property
    def name(self) -> str:
        return "sim"

    def evaluate(self, context: Mortm45DecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        pens: List[float] = []
        sims: List[float] = []
        for idx in range(len(context.batch)):
            if bool(context.is_empty[idx]) or context.ref_notes[idx] is None:
                pens.append(0.0)
                continue
            sim = roll_similarity(
                notes_to_roll(context.gen_notes[idx]),
                notes_to_roll(context.ref_notes[idx]),
            )
            sims.append(float(sim))
            pens.append(similarity_reward(sim, self._weight, self._center, self._sharpness))
        return (
            torch.tensor(pens, dtype=torch.float32),
            {"sim_mean": float(np.mean(sims)) if sims else float("nan")},
        )


def build_mortm45_guardrail_suite(tokenizer: Any) -> GuardrailSuite[Mortm45DecodedContext]:
    builder = Mortm45ContextBuilder(tokenizer)
    return GuardrailSuite(
        context_builder=builder,
        empty_detector=lambda ctx: ctx.is_empty,
        rules=(
            Mortm45StructureRule(tokenizer),
            Mortm45DensityRule(tokenizer),
            Mortm45KeyOutlierRule(tokenizer),
            Mortm45SimilarityRule(),
        ),
    )
