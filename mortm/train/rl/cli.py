"""Unified CLI & OCP Backend Registry for CARL training across `MORTM5 SyMIc` and `MORTM4.5`."""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from mortm.models.modules.config import MORTM5Args
from mortm.utils.mortm5_infer import select_device

from . import carl_paths as legacy_paths
from .adapters.mortm45 import (
    Mortm45ActorAdapter,
    Mortm45CriticAdapter,
    Mortm45GenSftRepository,
    build_mortm45_guardrail_suite,
)
from .adapters.symic import (
    SYMIC_GUARDRAIL_MODES,
    SYMIC_UNFREEZE_MODES,
    SyMIcCriticAdapter,
    SyMIcGroupedActorAdapter,
    SyMIcMemmapRepository,
    SyMIcMidiRepository,
    build_symic_guardrail_suite,
    load_midivision_codec,
)
from .domain import UnifiedEvenConfig, UnifiedOddConfig, UnifiedSamplingConfig
from .engine import UnifiedCARLDriver
from .interfaces import CARLActorAdapter, CARLCriticAdapter, CARLDataSamplePool, GuardrailSuite


DEFAULT_SYMIC_CONFIG = "configs/models/mortm5/symic_main/symic_g4l4_160m.json"
DEFAULT_SYMIC_CKPT = (
    "out/symic_main/symic_g4l4_160m_depth_4ep/"
    "MORTM5.SyMIc_G4L4_160M_Depth_Scratch_16bar_4ep_muon_wsd_latest.pth"
)
DEFAULT_SYMIC_TRAIN_MANIFEST = "data/gmd/train.json"


@dataclass(frozen=True)
class BuiltBackendComponents:
    """Resolved backend components injected into `UnifiedCARLDriver`."""

    actor: CARLActorAdapter
    base_actor: CARLActorAdapter
    critic: CARLCriticAdapter
    ana_pool: CARLDataSamplePool
    gem_pool: CARLDataSamplePool
    guardrails: GuardrailSuite[Any]
    ckpt_prefix: str


class CARLBackendBuilder(ABC):
    """Strategy interface (OCP) constructing all CARL components for a specific model family."""

    @abstractmethod
    def build(self, args: argparse.Namespace, device: torch.device) -> BuiltBackendComponents:
        ...


class SyMIcBackendBuilder(CARLBackendBuilder):
    """Builder for `MORTM5 SyMIc-G4L4 + Depth`."""

    def build(self, args: argparse.Namespace, device: torch.device) -> BuiltBackendComponents:
        cfg_path = args.config or DEFAULT_SYMIC_CONFIG
        ckpt_path = args.ckpt or DEFAULT_SYMIC_CKPT
        m_args = MORTM5Args(cfg_path)
        tok_key = m_args.tokenizer_key or "m4_g4l4"
        vq = load_midivision_codec(
            tokenizer_key=tok_key,
            vq_config_path=args.vq_config,
            vq_ckpt_path=args.vq_ckpt,
            device=device,
        )
        actor = SyMIcGroupedActorAdapter.from_pretrained(
            m_args,
            ckpt_path,
            device=device,
            prompt_bars=args.prompt_bars,
            total_bars=args.total_bars,
            unfreeze_mode=args.unfreeze_mode,
        )
        base_actor = SyMIcGroupedActorAdapter.from_pretrained(
            m_args,
            ckpt_path,
            device=device,
            prompt_bars=args.prompt_bars,
            total_bars=args.total_bars,
            unfreeze_mode=args.unfreeze_mode,
        )
        base_actor.freeze_all()
        critic = SyMIcCriticAdapter.from_pretrained(
            m_args,
            ckpt_path,
            device=device,
            prompt_bars=args.prompt_bars,
            lora_rank=args.critic_lora_rank,
            lora_alpha=args.critic_lora_alpha,
            vq_model=vq,
            canonicalize_pianoroll=bool(args.critic_pianoroll),
        )
        print(
            f"[CARL] SyMIc Critic discrete piano-roll canonicalization: "
            f"{'ON' if critic.uses_pianoroll_canonicalizer else 'OFF'}",
            flush=True,
        )

        if args.token_dir:
            repo = SyMIcMemmapRepository(
                token_dir=args.token_dir,
                model_args=m_args,
                split="train",
                prompt_bars=args.prompt_bars,
                total_bars=args.total_bars,
                min_active_tracks=args.min_active_tracks,
            )
            eff_limit = args.pool_limit
        else:
            manifest_path = args.midi_manifest or (
                DEFAULT_SYMIC_TRAIN_MANIFEST if os.path.isfile(DEFAULT_SYMIC_TRAIN_MANIFEST) else None
            )
            if not manifest_path:
                raise ValueError("SyMIc backend requires either --token-dir or --midi-manifest.")
            repo = SyMIcMidiRepository(
                midi_manifest_path=manifest_path,
                model=actor.actor_module,  # type: ignore[arg-type]
                vq_model=vq,
                device=device,
                prompt_bars=args.prompt_bars,
                total_bars=args.total_bars,
                min_active_tracks=args.min_active_tracks,
            )
            eff_limit = args.midi_pool_limit if args.midi_pool_limit is not None else args.pool_limit

        ana_pool, gem_pool = repo.load_disjoint_pools(limit=eff_limit, seed=args.seed)
        guardrails = build_symic_guardrail_suite(
            vq_model=vq,
            actor=actor,
            mode=args.guardrail_mode,
        )
        return BuiltBackendComponents(
            actor=actor,
            base_actor=base_actor,
            critic=critic,
            ana_pool=ana_pool,
            gem_pool=gem_pool,
            guardrails=guardrails,
            ckpt_prefix="SyMIc",
        )


class Mortm45BackendBuilder(CARLBackendBuilder):
    """Builder for `MORTM4.5` (`CONST-MIDI` SFT)."""

    def build(self, args: argparse.Namespace, device: torch.device) -> BuiltBackendComponents:
        from ..tokenizer import TO_MUSIC, Tokenizer, get_token_converter_pro

        tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
        tok.mode(TO_MUSIC)
        cfg_path = args.config or legacy_paths.GEM_CONFIG
        ckpt_path = args.ckpt or legacy_paths.GEM_CKPT
        manifest_path = args.midi_manifest or legacy_paths.GEN_SFT_MANIFEST

        actor = Mortm45ActorAdapter.from_pretrained(cfg_path, ckpt_path, tok, device, trainable=True)
        base_actor = Mortm45ActorAdapter.from_pretrained(cfg_path, ckpt_path, tok, device, trainable=False)
        critic = Mortm45CriticAdapter.from_pretrained(cfg_path, ckpt_path, tok, device, pool="mil")
        repo_m45 = Mortm45GenSftRepository(manifest_path=manifest_path, tokenizer=tok)
        ana_pool, gem_pool = repo_m45.load_disjoint_pools(limit=args.pool_limit, seed=args.seed)
        guardrails = build_mortm45_guardrail_suite(tok)
        return BuiltBackendComponents(
            actor=actor,
            base_actor=base_actor,
            critic=critic,
            ana_pool=ana_pool,
            gem_pool=gem_pool,
            guardrails=guardrails,
            ckpt_prefix="MORTM45",
        )


class CARLBackendRegistry:
    """Open/Closed registry mapping backend names (`'symic'`, `'mortm45'`, ...) to builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, CARLBackendBuilder] = {}

    def register(self, key: str, builder: CARLBackendBuilder) -> None:
        norm = str(key).strip().lower()
        if not norm:
            raise ValueError("Backend key must be non-empty.")
        self._builders[norm] = builder

    def get(self, key: str) -> CARLBackendBuilder:
        norm = str(key).strip().lower()
        builder = self._builders.get(norm)
        if builder is None:
            raise KeyError(
                f"Unsupported CARL backend '{key}'. Available: {sorted(self._builders.keys())}"
            )
        return builder

    def available_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


DEFAULT_BACKEND_REGISTRY = CARLBackendRegistry()
DEFAULT_BACKEND_REGISTRY.register("symic", SyMIcBackendBuilder())
DEFAULT_BACKEND_REGISTRY.register("mortm45", Mortm45BackendBuilder())


def resolve_default_paths_for_backend(
    backend: str,
    config: Optional[str] = None,
    ckpt: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve effective `(config_path, ckpt_path)` without cross-backend default pollution."""
    norm = str(backend).strip().lower()
    if norm == "symic":
        return (config or DEFAULT_SYMIC_CONFIG, ckpt or DEFAULT_SYMIC_CKPT)
    if norm == "mortm45":
        return (config or legacy_paths.GEM_CONFIG, ckpt or legacy_paths.GEM_CKPT)
    if config and ckpt:
        return (config, ckpt)
    raise ValueError(f"Explicit --config and --ckpt required for custom backend '{backend}'.")


def build_arg_parser(registry: Optional[CARLBackendRegistry] = None) -> argparse.ArgumentParser:
    reg = registry or DEFAULT_BACKEND_REGISTRY
    ap = argparse.ArgumentParser(
        description="Unified CARL (Critique-Augmented RL) Trainer for MORTM5 SyMIc & MORTM4.5"
    )
    ap.add_argument(
        "--backend",
        type=str,
        choices=reg.available_keys(),
        default="symic",
        help="Target model paradigm ('symic' for MORTM5 SyMIc-G4L4+Depth, 'mortm45' for MORTM4.5).",
    )
    ap.add_argument("--device", type=str, default=None, help="Compute device override (cuda, mps, cpu).")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for CARL checkpoints & QTable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pool-limit",
        type=int,
        default=20000,
        help="Max total unique songs/windows to split 50:50 across Odd (ana) & Even (gem) pools (20000 = 10k + 10k songs).",
    )
    ap.add_argument(
        "--midi-pool-limit",
        type=int,
        default=None,
        help="Optional override for --midi-manifest pool size (defaults to --pool-limit).",
    )
    ap.add_argument("--q-max", type=int, default=3000, help="QTable replay capacity.")

    # Model & Data paths (None by default so each backend resolves its own canonical defaults!)
    ap.add_argument("--config", type=str, default=None, help="Model config JSON path (backend-specific default if omitted).")
    ap.add_argument("--ckpt", type=str, default=None, help="Initial pretrained backbone checkpoint (backend-specific default if omitted).")
    ap.add_argument("--vq-config", type=str, default=None, help="Optional MidiVision config override (SyMIc).")
    ap.add_argument("--vq-ckpt", type=str, default=None, help="Optional MidiVision checkpoint override (SyMIc).")
    ap.add_argument(
        "--token-dir",
        type=str,
        default=None,
        help="SyMIc memmap token directory containing train_tokens.bin / train_offsets.npy / train_slots.npy.",
    )
    ap.add_argument(
        "--midi-manifest",
        type=str,
        default=None,
        help="Fallback MIDI JSON manifest for SyMIc on-the-fly encoding (or MORTM4.5 SFT manifest).",
    )

    # SyMIc horizon & unfreeze options
    ap.add_argument("--prompt-bars", type=int, default=4)
    ap.add_argument("--total-bars", type=int, default=16)
    ap.add_argument("--min-active-tracks", type=int, default=2)
    ap.add_argument(
        "--unfreeze-mode",
        type=str,
        choices=SYMIC_UNFREEZE_MODES,
        default="attn_and_depth",
        help="SyMIc Actor GRPO trainable scope ('full_decoder' unfreezes Attention+FFN+Wout, 'attn_and_depth' freezes FFN, 'depth_only' trains Wout only).",
    )
    ap.add_argument(
        "--guardrail-mode",
        type=str,
        choices=SYMIC_GUARDRAIL_MODES,
        default="empty_only",
        help="SyMIc reward mode ('empty_only' uses R_disc + empty_penalty only; 'full' adds silence/density/scale/loop_copy rules).",
    )
    ap.add_argument("--critic-lora-rank", type=int, default=8)
    ap.add_argument("--critic-lora-alpha", type=float, default=16.0)
    ap.add_argument(
        "--critic-pianoroll",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: decode SyMIc Critic inputs to discrete piano-roll and re-encode symmetrically before scoring (default: False).",
    )

    # Sampling (shared or Coarse-to-Fine)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--fine-temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--fine-top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--fine-top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)

    # Odd Round (Critic)
    ap.add_argument("--odd-batch-size", type=int, default=16)
    ap.add_argument("--odd-pool-size", type=int, default=256)
    ap.add_argument("--odd-lr", type=float, default=1e-4)
    ap.add_argument("--odd-max-steps", type=int, default=20000)
    ap.add_argument("--odd-eval-window", type=int, default=100)
    ap.add_argument("--auc-threshold", type=float, default=0.93)
    ap.add_argument("--auc-min-steps", type=int, default=1000)
    ap.add_argument(
        "--odd-logit-reg",
        type=float,
        default=1e-3,
        help="L2 logit regularization coefficient preventing Critic logit inflation (default: 1e-3).",
    )
    ap.add_argument(
        "--odd-min-acc",
        type=float,
        default=0.70,
        help="Minimum Human & latest-AI accuracy at the calibrated pass threshold required to finish Odd round.",
    )

    # Even Round (GRPO Actor)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--even-lr", type=float, default=2e-6)
    ap.add_argument("--even-max-steps", type=int, default=3000)
    ap.add_argument("--ppo-epochs", type=int, default=1, help="GRPO optimization passes per rollout batch.")
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--kl-stop", type=float, default=0.30)
    ap.add_argument("--disc-stop", type=float, default=0.80)
    ap.add_argument("--reward-stop", type=float, default=0.625)
    ap.add_argument(
        "--calibrate-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate Even-round R_disc scale to the preceding Odd-round Human/AI logit distribution (default: True).",
    )
    ap.add_argument("--gen-chunk", type=int, default=16)
    ap.add_argument("--critic-chunk", type=int, default=16)
    ap.add_argument("--train-chunk", type=int, default=8)

    # Outer loop & Resume & Logging
    ap.add_argument("--max-total-steps", type=int, default=100000)
    ap.add_argument("--target-hits", type=int, default=3)
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=50)
    ap.add_argument("--actor-resume", type=str, default=None)
    ap.add_argument("--critic-resume", type=str, default=None)
    ap.add_argument("--qtable-resume", type=str, default=None)
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    ap.add_argument("--wandb-project", type=str, default="MORTM-CARL")
    ap.add_argument("--run-name", type=str, default=None)
    return ap


def build_driver_from_args(
    args: argparse.Namespace,
    registry: Optional[CARLBackendRegistry] = None,
) -> UnifiedCARLDriver:
    reg = registry or DEFAULT_BACKEND_REGISTRY
    dev = select_device(args.device)
    sampling = UnifiedSamplingConfig(
        temperature=args.temperature,
        fine_temperature=args.fine_temperature,
        top_p=args.top_p,
        fine_top_p=args.fine_top_p,
        top_k=args.top_k,
        fine_top_k=args.fine_top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )
    pass_target = min(0.92, max(0.50, float(args.reward_stop)))
    odd_cfg = UnifiedOddConfig(
        batch_size=args.odd_batch_size,
        pool_size=args.odd_pool_size,
        lr=args.odd_lr,
        max_steps=args.odd_max_steps,
        eval_window=args.odd_eval_window,
        gen_chunk=args.gen_chunk,
        critic_chunk=args.critic_chunk,
        auc_threshold=args.auc_threshold,
        auc_min_steps=args.auc_min_steps,
        logit_reg_coef=args.odd_logit_reg,
        min_class_acc=args.odd_min_acc,
        pass_reward_target=pass_target,
        sampling=sampling,
    )
    even_cfg = UnifiedEvenConfig(
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        lr=args.even_lr,
        max_steps=args.even_max_steps,
        ppo_epochs=args.ppo_epochs,
        kl_coef=args.kl_coef,
        kl_stop=args.kl_stop,
        disc_stop=args.disc_stop,
        reward_stop=args.reward_stop,
        calibrate_reward=bool(args.calibrate_reward),
        gen_chunk=args.gen_chunk,
        critic_chunk=args.critic_chunk,
        train_chunk=args.train_chunk,
        sampling=sampling,
    )

    builder = reg.get(args.backend)
    built = builder.build(args, dev)

    ana_u = built.ana_pool.unique_song_count()
    gem_u = built.gem_pool.unique_song_count()
    ana_desc = f"{len(built.ana_pool):,} ({ana_u:,} songs)" if ana_u is not None else f"{len(built.ana_pool):,}"
    gem_desc = f"{len(built.gem_pool):,} ({gem_u:,} songs)" if gem_u is not None else f"{len(built.gem_pool):,}"
    print(
        f"[CARL] Initialized backend='{args.backend}' on device={dev} | "
        f"disjoint pools: ana={ana_desc}, gem={gem_desc}",
        flush=True,
    )
    driver = UnifiedCARLDriver(
        actor=built.actor,
        base_actor=built.base_actor,
        critic=built.critic,
        ana_pool=built.ana_pool,
        gem_pool=built.gem_pool,
        guardrail_suite=built.guardrails,
        out_dir=Path(args.out_dir),
        odd_cfg=odd_cfg,
        even_cfg=even_cfg,
        q_max=args.q_max,
        seed=args.seed,
        ckpt_prefix=built.ckpt_prefix,
        use_wandb=bool(args.wandb),
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )
    driver.resume(
        actor_ckpt=args.actor_resume,
        critic_ckpt=args.critic_resume,
        qtable_ckpt=args.qtable_resume,
    )
    return driver


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    driver = build_driver_from_args(args)
    driver.run(
        max_total_steps=args.max_total_steps,
        target_hits=args.target_hits,
        start_round=args.start_round,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
