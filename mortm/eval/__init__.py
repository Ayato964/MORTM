"""MORTM evaluation utilities and unified piano-roll benchmarks."""

from __future__ import annotations

__all__ = [
    "WindowSlice",
    "PianoRollMetrics",
    "ConfusionAccumulator",
    "StratifiedPianoRollStats",
    "CompactValidationSong",
    "SlidingWindowPlanner",
    "ValidationSongRepository",
    "RolloutStrategy",
    "FlatRolloutStrategy",
    "GroupedRolloutStrategy",
    "RolloutStrategyFactory",
    "MidiVisionCodecService",
    "AblationTargetSpec",
    "AblationPianoRollBenchmark",
    "MarkdownReportFormatter",
]

_LAZY_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from . import mortm5_pianoroll

        return getattr(mortm5_pianoroll, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
