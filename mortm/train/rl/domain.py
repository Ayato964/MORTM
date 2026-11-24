"""Immutable domain value objects and configuration specs for Unified CARL."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class CriticCalibration:
    """Immutable Odd⇄Even Critic logit-to-reward ruler calibration.

    Maps Critic logits (`+` = AI, `-` = Human) linearly onto `[0.0, 1.0]`:
        `R_disc(z) = clamp(0.5 - (z - center) / span, 0.0, 1.0)`
    so Odd-round stopping thresholds and Even-round `reward_stop` share the exact same scale.
    """

    center: float = 0.0
    span: float = 8.0
    human_mean: Optional[float] = None
    gem_mean: Optional[float] = None
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.center):
            raise ValueError(f"CriticCalibration.center must be finite, got {self.center}.")
        if not math.isfinite(self.span) or self.span <= 0.0:
            raise ValueError(f"CriticCalibration.span must be finite and > 0, got {self.span}.")

    def score_to_reward(self, disc_logits: Tensor) -> Tensor:
        return torch.clamp(
            0.5 - (disc_logits.float() - float(self.center)) / float(self.span),
            min=0.0,
            max=1.0,
        )

    def reward_to_logit_threshold(self, target_reward: float) -> float:
        r_clamped = min(1.0, max(0.0, float(target_reward)))
        return float(self.center) - (r_clamped - 0.5) * float(self.span)

    @classmethod
    def from_odd_stats(
        cls,
        human_mean: float,
        gem_mean: float,
        pass_reward: float = 0.625,
        min_span: float = 2.0,
        fallback_span: float = 8.0,
    ) -> "CriticCalibration":
        h_f = float(human_mean)
        g_f = float(gem_mean)
        if not (math.isfinite(h_f) and math.isfinite(g_f) and g_f > h_f + 1e-4):
            return cls(
                center=0.0,
                span=float(fallback_span),
                human_mean=h_f if math.isfinite(h_f) else None,
                gem_mean=g_f if math.isfinite(g_f) else None,
                calibrated=False,
            )
        center = 0.5 * (h_f + g_f)
        delta = max(float(min_span), g_f - h_f)
        anchor_r = min(0.92, max(0.85, float(pass_reward) + 0.05))
        span = delta / (2.0 * (anchor_r - 0.5))
        return cls(
            center=center,
            span=span,
            human_mean=h_f,
            gem_mean=g_f,
            calibrated=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center,
            "span": self.span,
            "human_mean": self.human_mean,
            "gem_mean": self.gem_mean,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticCalibration":
        return cls(
            center=float(data.get("center", 0.0)),
            span=float(data.get("span", 8.0)),
            human_mean=float(data["human_mean"]) if data.get("human_mean") is not None else None,
            gem_mean=float(data["gem_mean"]) if data.get("gem_mean") is not None else None,
            calibrated=bool(data.get("calibrated", False)),
        )


@dataclass(frozen=True)
class CARLExample:
    """Model-agnostic paired prompt + human reference sample.

    Attributes:
        example_id: Stable identifier for tracing and deduplication.
        song_id: Song / record index used to enforce strict disjoint Odd/Even splits.
        prompt_tokens: Input conditioning tokens passed to Actor rollout
            (e.g., ``[S_prompt, G]`` for SyMIc or ``[S_prompt]`` for MORTM4.5).
        human_full_tokens: Canonical human sequence fed to the Critic in Odd rounds
            (e.g., ``[S_full, G]`` for SyMIc or folded ``<CONST_M>..<TAG_END>`` for MORTM4.5).
        slot_codes: Optional 4-slot instrument assignment ``(slot0, slot1, slot2, slot3)``.
        aux_metadata: Optional paradigm-specific payload (e.g., MORTM4.5 SFT record dict).
    """

    example_id: str
    song_id: int
    prompt_tokens: np.ndarray
    human_full_tokens: np.ndarray
    slot_codes: Tuple[int, ...] = ()
    aux_metadata: Optional[Any] = None

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("CARLExample.example_id must be non-empty.")
        if self.prompt_tokens.size == 0:
            raise ValueError(f"CARLExample({self.example_id}) has empty prompt_tokens.")
        if self.human_full_tokens.size == 0:
            raise ValueError(f"CARLExample({self.example_id}) has empty human_full_tokens.")


@dataclass(frozen=True)
class RolloutBatch:
    """Grouped Actor rollouts for Odd pool generation or Even GRPO updates.

    ``examples``, ``generated_sequences``, and ``critic_sequences`` all have length
    ``N = num_prompts * group_size`` with items from the same prompt arranged contiguously.
    """

    examples: Tuple[CARLExample, ...]
    generated_sequences: Tuple[np.ndarray, ...]
    critic_sequences: Tuple[Optional[np.ndarray], ...]
    group_size: int
    old_action_logprobs: Optional[Tuple[np.ndarray, ...]] = None

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")
        n = len(self.examples)
        if len(self.generated_sequences) != n or len(self.critic_sequences) != n:
            raise ValueError(
                f"RolloutBatch length mismatch: examples={n}, "
                f"generated={len(self.generated_sequences)}, critic={len(self.critic_sequences)}."
            )
        if self.old_action_logprobs is not None and len(self.old_action_logprobs) != n:
            raise ValueError(
                f"RolloutBatch old_action_logprobs length ({len(self.old_action_logprobs)}) "
                f"must match examples ({n})."
            )
        if n % self.group_size != 0:
            raise ValueError(
                f"RolloutBatch size ({n}) must be divisible by group_size ({self.group_size})."
            )

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class StepLogProbsKL:
    """Per-step action log-probabilities and exact KL divergence against the base policy.

    Attributes:
        action_logprobs: ``[B, T_act]`` log-probabilities of sampled actions under Actor.
        exact_kl: ``[B, T_act]`` exact conditional KL ``KL(pi_actor || pi_base)``.
        action_mask: ``[B, T_act]`` float/bool mask marking valid generated action steps.
    """

    action_logprobs: Tensor
    exact_kl: Tensor
    action_mask: Tensor

    def __post_init__(self) -> None:
        if (
            self.action_logprobs.shape != self.exact_kl.shape
            or self.action_logprobs.shape != self.action_mask.shape
        ):
            raise ValueError(
                "StepLogProbsKL shape mismatch: "
                f"logp={tuple(self.action_logprobs.shape)}, "
                f"kl={tuple(self.exact_kl.shape)}, "
                f"mask={tuple(self.action_mask.shape)}."
            )


@dataclass(frozen=True)
class GuardrailEvaluation:
    """Aggregated rule-based guardrail penalties (strictly <= 0, 0 on human GT)."""

    penalties: Tensor
    is_empty: Tensor
    rule_penalties: Mapping[str, Tensor] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.penalties.dim() != 1 or self.is_empty.dim() != 1:
            raise ValueError("GuardrailEvaluation penalties and is_empty must be 1D tensors.")
        if self.penalties.numel() != self.is_empty.numel():
            raise ValueError(
                f"GuardrailEvaluation length mismatch: penalties={self.penalties.numel()}, "
                f"is_empty={self.is_empty.numel()}."
            )


@dataclass(frozen=True)
class UnifiedSamplingConfig:
    """Unified rollout sampling hyperparameters supporting both Symbolic and C2F RVQ models."""

    temperature: float = 1.0
    fine_temperature: Optional[float] = None
    top_p: float = 0.95
    fine_top_p: Optional[float] = None
    top_k: int = 0
    fine_top_k: Optional[int] = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    repetition_window: int = 24
    max_new_tokens: int = 1200
    max_measures: int = 8

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}.")
        if self.fine_temperature is not None and self.fine_temperature < 0.0:
            raise ValueError(f"fine_temperature must be >= 0, got {self.fine_temperature}.")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.fine_top_p is not None and not (0.0 < self.fine_top_p <= 1.0):
            raise ValueError(f"fine_top_p must be in (0, 1], got {self.fine_top_p}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}.")
        if self.fine_top_k is not None and self.fine_top_k < 0:
            raise ValueError(f"fine_top_k must be >= 0, got {self.fine_top_k}.")
        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}.")
        if self.max_new_tokens <= 0 or self.max_measures <= 0:
            raise ValueError("max_new_tokens and max_measures must be positive.")


@dataclass(frozen=True)
class UnifiedOddConfig:
    """Hyperparameters for Odd Rounds (Critic / MIL Discriminator training + Q-Table)."""

    batch_size: int = 16
    pool_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 1.0
    max_steps: int = 20000
    eval_window: int = 100
    log_every: int = 20
    gen_chunk: int = 16
    critic_chunk: int = 16
    auc_threshold: float = 0.93
    auc_min_slope: float = 0.01
    auc_slope_window: int = 50
    auc_patience: int = 10
    auc_min_steps: int = 1000
    logit_reg_coef: float = 1e-3
    min_class_acc: float = 0.70
    pass_reward_target: float = 0.625
    fallback_disc_span: float = 8.0
    sampling: UnifiedSamplingConfig = field(default_factory=UnifiedSamplingConfig)

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError(
                f"UnifiedOddConfig.batch_size must be >= 2 so mix_batch includes both Human and AI, got {self.batch_size}."
            )
        if self.pool_size < 1 or self.max_steps < 1 or self.eval_window < 1 or self.log_every < 1:
            raise ValueError("pool_size, max_steps, eval_window, and log_every must be >= 1.")
        if self.gen_chunk < 1 or self.critic_chunk < 1:
            raise ValueError("gen_chunk and critic_chunk must be >= 1.")
        if self.lr <= 0.0 or self.grad_clip < 0.0:
            raise ValueError("lr must be > 0 and grad_clip must be >= 0.")
        if not (0.0 < self.auc_threshold <= 1.0):
            raise ValueError(f"auc_threshold must be in (0, 1], got {self.auc_threshold}.")
        if self.auc_slope_window < 2:
            raise ValueError(
                f"auc_slope_window must be >= 2 for linear slope estimation, got {self.auc_slope_window}."
            )
        if self.auc_patience < 1 or self.auc_min_steps < 0:
            raise ValueError("auc_patience must be >= 1 and auc_min_steps must be >= 0.")
        if self.logit_reg_coef < 0.0:
            raise ValueError(f"logit_reg_coef must be >= 0, got {self.logit_reg_coef}.")
        if not (0.0 <= self.min_class_acc <= 1.0):
            raise ValueError(f"min_class_acc must be in [0, 1], got {self.min_class_acc}.")
        if not (0.0 < self.pass_reward_target < 1.0):
            raise ValueError(f"pass_reward_target must be in (0, 1), got {self.pass_reward_target}.")
        if self.fallback_disc_span <= 0.0:
            raise ValueError(f"fallback_disc_span must be > 0, got {self.fallback_disc_span}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "pool_size": self.pool_size,
            "lr": self.lr,
            "grad_clip": self.grad_clip,
            "max_steps": self.max_steps,
            "eval_window": self.eval_window,
            "log_every": self.log_every,
            "gen_chunk": self.gen_chunk,
            "critic_chunk": self.critic_chunk,
            "auc_threshold": self.auc_threshold,
            "auc_min_slope": self.auc_min_slope,
            "auc_slope_window": self.auc_slope_window,
            "auc_patience": self.auc_patience,
            "auc_min_steps": self.auc_min_steps,
            "logit_reg_coef": self.logit_reg_coef,
            "min_class_acc": self.min_class_acc,
            "pass_reward_target": self.pass_reward_target,
            "fallback_disc_span": self.fallback_disc_span,
        }


@dataclass(frozen=True)
class UnifiedEvenConfig:
    """Hyperparameters for Even Rounds (Actor Clipped GRPO + Exact KL + Guardrails)."""

    group_size: int = 16
    prompts_per_step: int = 4
    lr: float = 2e-6
    max_steps: int = 3000
    grad_clip: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    adv_eps: float = 1e-4
    gen_chunk: int = 16
    critic_chunk: int = 16
    train_chunk: int = 8
    kl_stop: float = 0.30
    disc_stop: float = 0.80
    reward_stop: float = 0.625
    stop_window: int = 20
    snapshot_every: int = 20
    log_every: int = 10
    disc_weight: float = 1.0
    disc_span: float = 8.0
    calibrate_reward: bool = True
    ppo_epochs: int = 1
    empty_penalty: float = -1.0
    sampling: UnifiedSamplingConfig = field(default_factory=UnifiedSamplingConfig)

    def __post_init__(self) -> None:
        if self.group_size < 1 or self.prompts_per_step < 1 or self.max_steps < 1:
            raise ValueError("group_size, prompts_per_step, and max_steps must be >= 1.")
        if self.gen_chunk < 1 or self.critic_chunk < 1 or self.train_chunk < 1:
            raise ValueError("gen_chunk, critic_chunk, and train_chunk must be >= 1.")
        if self.stop_window < 1 or self.snapshot_every < 1 or self.log_every < 1:
            raise ValueError("stop_window, snapshot_every, and log_every must be >= 1.")
        if self.lr <= 0.0 or self.grad_clip < 0.0:
            raise ValueError("lr must be > 0 and grad_clip must be >= 0.")
        if self.clip_eps <= 0.0 or self.kl_coef < 0.0 or self.adv_eps <= 0.0:
            raise ValueError("clip_eps > 0, kl_coef >= 0, and adv_eps > 0 are required.")
        if self.disc_span <= 0.0:
            raise ValueError(f"disc_span must be > 0, got {self.disc_span}.")
        if self.ppo_epochs < 1:
            raise ValueError(f"ppo_epochs must be >= 1, got {self.ppo_epochs}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_size": self.group_size,
            "prompts_per_step": self.prompts_per_step,
            "lr": self.lr,
            "max_steps": self.max_steps,
            "grad_clip": self.grad_clip,
            "clip_eps": self.clip_eps,
            "kl_coef": self.kl_coef,
            "adv_eps": self.adv_eps,
            "gen_chunk": self.gen_chunk,
            "critic_chunk": self.critic_chunk,
            "train_chunk": self.train_chunk,
            "kl_stop": self.kl_stop,
            "disc_stop": self.disc_stop,
            "reward_stop": self.reward_stop,
            "stop_window": self.stop_window,
            "snapshot_every": self.snapshot_every,
            "log_every": self.log_every,
            "disc_weight": self.disc_weight,
            "disc_span": self.disc_span,
            "calibrate_reward": self.calibrate_reward,
            "ppo_epochs": self.ppo_epochs,
            "empty_penalty": self.empty_penalty,
        }
