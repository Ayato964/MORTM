"""
Stage 3 Main Benchmark Suite for SyMIc vs Octuple-MMT vs Multitrack-REMI.

Implements:
  - §5.1 Evaluation 1: Systems & Context Efficiency Scaling (`Notes/Bar` vs AR Steps, Latency, KV Cache)
  - §5.2 Evaluation 2: 4-to-16 Bar Prompted Continuation (`FMD [CLaMP 2]` + Paired `MusPy` PCE / SC / GC)
"""

from .domain import (
    DEFAULT_DENSITY_TIERS,
    DEFAULT_SAMPLING_SPEC,
    DEFAULT_STAGE3_CANDIDATES,
    DensityTierSpec,
    ModelCandidateSpec,
    MusPySongMetrics,
    RolloutTelemetry,
    SamplingSpec,
    TestSegmentItem,
)

__all__ = [
    "DEFAULT_DENSITY_TIERS",
    "DEFAULT_SAMPLING_SPEC",
    "DEFAULT_STAGE3_CANDIDATES",
    "DensityTierSpec",
    "ModelCandidateSpec",
    "MusPySongMetrics",
    "RolloutTelemetry",
    "SamplingSpec",
    "TestSegmentItem",
]
