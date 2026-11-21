"""CARL: Critique-Augmented Reinforcement Learning for MORTM (MORTM4.5 & MORTM5 SyMIc).

Provides the unified, model-agnostic Odd/Even self-critique + GRPO training engine
(`UnifiedCARLDriver`, `OddRoundEngine`, `EvenRoundEngine`) and SOLID role interfaces
(`CARLActorAdapter`, `CARLCriticAdapter`, `CARLDataRepository`, `GuardrailRule`).
"""

from .adapters import (
    AnalysisSyMIc,
    Mortm45ActorAdapter,
    Mortm45CriticAdapter,
    Mortm45GenSftRepository,
    SyMIcCriticAdapter,
    SyMIcGroupedActorAdapter,
    SyMIcMemmapRepository,
    SyMIcMidiRepository,
    build_mortm45_guardrail_suite,
    build_symic_guardrail_suite,
)
from .domain import (
    CARLExample,
    GuardrailEvaluation,
    RolloutBatch,
    StepLogProbsKL,
    UnifiedEvenConfig,
    UnifiedOddConfig,
    UnifiedSamplingConfig,
)
from .engine import (
    ClippedGRPOLoss,
    EvenRoundEngine,
    OddRoundEngine,
    UnifiedCARLDriver,
    compute_discriminator_reward,
    compute_grpo_advantages,
)
from .interfaces import (
    CARLActorAdapter,
    CARLCriticAdapter,
    CARLDataRepository,
    CARLDataSamplePool,
    GuardrailContextBuilder,
    GuardrailRule,
    GuardrailSuite,
)
from .lora_adapter import LoRALinearAdapter, inject_attention_lora
from .optim import MasterWeightAdamW
from .q_table import QTable

__all__ = [
    "AnalysisSyMIc",
    "CARLActorAdapter",
    "CARLCriticAdapter",
    "CARLDataRepository",
    "CARLDataSamplePool",
    "CARLExample",
    "ClippedGRPOLoss",
    "EvenRoundEngine",
    "GuardrailContextBuilder",
    "GuardrailEvaluation",
    "GuardrailRule",
    "GuardrailSuite",
    "LoRALinearAdapter",
    "MasterWeightAdamW",
    "Mortm45ActorAdapter",
    "Mortm45CriticAdapter",
    "Mortm45GenSftRepository",
    "OddRoundEngine",
    "QTable",
    "RolloutBatch",
    "StepLogProbsKL",
    "SyMIcCriticAdapter",
    "SyMIcGroupedActorAdapter",
    "SyMIcMemmapRepository",
    "SyMIcMidiRepository",
    "UnifiedCARLDriver",
    "UnifiedEvenConfig",
    "UnifiedOddConfig",
    "UnifiedSamplingConfig",
    "build_mortm45_guardrail_suite",
    "build_symic_guardrail_suite",
    "compute_discriminator_reward",
    "compute_grpo_advantages",
    "inject_attention_lora",
]
