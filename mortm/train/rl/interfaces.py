"""Abstract role interfaces (SOLID: ISP, OCP, DIP) for Unified CARL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor, nn

from .domain import (
    CARLExample,
    GuardrailEvaluation,
    RolloutBatch,
    StepLogProbsKL,
    UnifiedSamplingConfig,
)


TContext = TypeVar("TContext")


class CARLDataSamplePool(ABC):
    """Read-only random-access pool of `CARLExample` items with O(1) memory overhead."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> CARLExample:
        ...

    def sample(self, count: int, rng: np.random.Generator) -> List[CARLExample]:
        total = len(self)
        if total == 0 or count <= 0:
            return []
        idx = rng.choice(total, size=count, replace=(total < count))
        return [self[int(i)] for i in np.atleast_1d(idx)]


class InMemorySamplePool(CARLDataSamplePool):
    """Immutable in-memory pool wrapping a pre-validated sequence of `CARLExample`."""

    def __init__(self, items: Sequence[CARLExample]) -> None:
        self._items: Tuple[CARLExample, ...] = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> CARLExample:
        return self._items[index]


class LazyIndexedSamplePool(CARLDataSamplePool):
    """On-demand sample pool holding only integer indices into a memmap/streaming source."""

    def __init__(
        self,
        indices: Sequence[int] | np.ndarray,
        loader_fn: Callable[[int], CARLExample],
    ) -> None:
        self._indices = np.asarray(indices, dtype=np.int64).copy()
        self._loader_fn = loader_fn

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, index: int) -> CARLExample:
        raw_idx = int(self._indices[int(index)])
        return self._loader_fn(raw_idx)


class CARLDataRepository(ABC):
    """Repository providing song/record-disjoint Odd (`ana`) and Even (`gem`) pools."""

    @abstractmethod
    def load_disjoint_pools(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[CARLDataSamplePool, CARLDataSamplePool]:
        """Return ``(ana_pool, gem_pool)`` with strictly disjoint ``song_id`` sets."""


class CARLActorAdapter(ABC):
    """Strategy encapsulating model-specific rollout, parameter unfreezing, and exact KL."""

    @property
    @abstractmethod
    def actor_module(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    def freeze_all(self) -> None:
        self.actor_module.eval()
        for param in self.actor_module.parameters():
            param.requires_grad_(False)

    @abstractmethod
    def unfreeze_for_grpo(self) -> Tuple[int, int]:
        """Unfreeze target Actor parameters for Even-round GRPO and return ``(trainable, total)``."""

    def prepare_for_grpo_forward(self) -> None:
        """Disable dropout noise (and merge `loralib` weights if present) while keeping `requires_grad`."""
        self.actor_module.eval()

    @abstractmethod
    def rollout_group(
        self,
        examples: Sequence[CARLExample],
        group_size: int,
        sampling: UnifiedSamplingConfig,
        chunk_size: int = 16,
        seed: Optional[int] = None,
    ) -> RolloutBatch:
        """Generate `group_size` continuations per example."""

    @abstractmethod
    def forward_logprobs_and_exact_kl(
        self,
        base_actor: "CARLActorAdapter",
        examples: Sequence[CARLExample],
        generated_sequences: Sequence[np.ndarray],
    ) -> StepLogProbsKL:
        """Compute per-step action log-probs and exact conditional KL on a micro-batch."""

    def snapshot_state(self) -> Dict[str, Tensor]:
        """Offload healthy Actor snapshot to host CPU RAM to save ~840 MB of GPU/MPS VRAM."""
        return {k: v.detach().cpu().clone() for k, v in self.actor_module.state_dict().items()}

    def restore_state(self, state: Dict[str, Tensor]) -> None:
        self.actor_module.load_state_dict(state)

    @abstractmethod
    def export_checkpoint_state(self) -> Dict[str, Any]:
        """Serialize Actor weights safely (handling LoRA unmerge markers if applicable)."""

    @abstractmethod
    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Load serialized Actor weights."""


class CARLCriticAdapter(ABC):
    """Strategy encapsulating Odd-round Discriminator training and Even-round scoring."""

    @property
    @abstractmethod
    def critic_module(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def mark_trainable(self) -> Dict[str, int]:
        """Enable gradients strictly on Critic adapter/pooling heads and return group counts."""

    def set_train_mode(self, training: bool) -> None:
        if training:
            self.critic_module.train()
        else:
            self.critic_module.eval()

    @abstractmethod
    def score_disc_logits(
        self,
        critic_sequences: Sequence[np.ndarray],
        chunk_size: Optional[int] = None,
        grad_enabled: bool = False,
    ) -> Tensor:
        """Return 1D float32 logits ``[N]`` (positive = AI, negative = Human)."""

    @abstractmethod
    def export_compact_state(self) -> Dict[str, Any]:
        """Return compact dictionary containing only trainable Critic adapter & pooling weights."""

    @abstractmethod
    def load_compact_state(self, state: Dict[str, Any]) -> None:
        """Restore compact Critic adapter & pooling weights onto the initialized backbone."""


class GuardrailContextBuilder(ABC, Generic[TContext]):
    """Decodes a `RolloutBatch` once per step so multiple `GuardrailRule`s share the result."""

    @abstractmethod
    def build_context(self, batch: RolloutBatch) -> TContext:
        ...


class GuardrailRule(ABC, Generic[TContext]):
    """Single zero-on-human penalty rule (OCP: add rules without modifying the engine)."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, context: TContext) -> Tuple[Tensor, Dict[str, float]]:
        """Return ``(penalties_1d_le_zero, diagnostic_scalars)``."""


class GuardrailSuite(Generic[TContext]):
    """Composite evaluator combining one `GuardrailContextBuilder` and N `GuardrailRule`s."""

    def __init__(
        self,
        context_builder: GuardrailContextBuilder[TContext],
        empty_detector: Callable[[TContext], Tensor],
        rules: Sequence[GuardrailRule[TContext]],
    ) -> None:
        self._context_builder = context_builder
        self._empty_detector = empty_detector
        self._rules: Tuple[GuardrailRule[TContext], ...] = tuple(rules)

    @property
    def rules(self) -> Tuple[GuardrailRule[TContext], ...]:
        return self._rules

    def evaluate(self, batch: RolloutBatch) -> GuardrailEvaluation:
        n = len(batch)
        if n == 0:
            empty_f = torch.zeros((0,), dtype=torch.float32)
            empty_b = torch.zeros((0,), dtype=torch.bool)
            return GuardrailEvaluation(penalties=empty_f, is_empty=empty_b)

        ctx = self._context_builder.build_context(batch)
        is_empty = self._empty_detector(ctx).to(dtype=torch.bool, device="cpu")
        if is_empty.numel() != n:
            raise RuntimeError(
                f"empty_detector returned {is_empty.numel()} flags for batch of size {n}."
            )

        total_pen = torch.zeros((n,), dtype=torch.float32)
        rule_penalties: Dict[str, Tensor] = {}
        metrics: Dict[str, float] = {"empty_rate": float(is_empty.float().mean().item())}

        valid_mask = ~is_empty
        for rule in self._rules:
            pen, rule_metrics = rule.evaluate(ctx)
            pen_cpu = pen.detach().to(dtype=torch.float32, device="cpu")
            if pen_cpu.shape != (n,):
                raise RuntimeError(
                    f"GuardrailRule '{rule.name}' returned shape {tuple(pen_cpu.shape)}, expected ({n},)."
                )
            # Enforce CARL v2 invariant: guardrails can never grant positive bonuses.
            pen_clamped = torch.clamp(pen_cpu, max=0.0)
            pen_masked = torch.where(valid_mask, pen_clamped, torch.zeros_like(pen_clamped))
            rule_penalties[rule.name] = pen_masked
            total_pen = total_pen + pen_masked
            metrics[f"pen_{rule.name}"] = float(pen_masked.mean().item())
            for mk, mv in rule_metrics.items():
                metrics[mk] = float(mv)

        return GuardrailEvaluation(
            penalties=total_pen,
            is_empty=is_empty,
            rule_penalties=rule_penalties,
            metrics=metrics,
        )
