from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


RgbColor = Tuple[int, int, int]

# General MIDI program family labels for readable track legends
_GM_FAMILY_NAMES: Tuple[str, ...] = (
    "Piano",
    "Chromatic Perc",
    "Organ",
    "Guitar",
    "Bass",
    "Strings",
    "Ensemble",
    "Brass",
    "Reed",
    "Pipe",
    "Synth Lead",
    "Synth Pad",
    "Synth FX",
    "Ethnic",
    "Percussive",
    "Sound FX",
)

# Disciplined 4-slot palette (high contrast on dark slate #0B0F19)
DEFAULT_SLOT_PALETTE: Tuple[RgbColor, ...] = (
    (245, 158, 11),   # Slot 0 (Drums): Amber
    (56, 189, 248),   # Slot 1: Sky Cyan
    (52, 211, 153),   # Slot 2: Emerald
    (167, 139, 250),  # Slot 3: Soft Violet
)


@dataclass(frozen=True)
class Resolution:
    """Immutable even-dimension video canvas resolution."""
    width: int = 1920
    height: int = 1080

    def __post_init__(self) -> None:
        if self.width < 640 or self.height < 360:
            raise ValueError(f"Resolution too small: {self.width}x{self.height}")
        if self.width % 2 != 0 or self.height % 2 != 0:
            raise ValueError(f"Resolution must have even dimensions for H.264: {self.width}x{self.height}")

    @property
    def byte_size_rgba(self) -> int:
        return self.width * self.height * 4


@dataclass(frozen=True)
class PitchWindow:
    """Immutable MIDI pitch viewing window [min_pitch, max_pitch] inclusive."""
    min_pitch: int
    max_pitch: int

    def __post_init__(self) -> None:
        if not (0 <= self.min_pitch < self.max_pitch <= 127):
            raise ValueError(f"Invalid PitchWindow [{self.min_pitch}, {self.max_pitch}]")

    @property
    def span(self) -> int:
        return self.max_pitch - self.min_pitch + 1

    @classmethod
    def from_ground_truth_roll(cls, flat_roll: np.ndarray, padding: int = 6, min_span: int = 60) -> "PitchWindow":
        """Derive optimal pitch framing from [8, steps, 128] roll while keeping context for noise."""
        sustain = flat_roll[1::2]
        active_pitches = np.nonzero(sustain.sum(axis=(0, 1)) > 0)[0]
        if len(active_pitches) == 0:
            return cls(24, 84)
        lo = max(12, int(active_pitches.min()) - padding)
        hi = min(108, int(active_pitches.max()) + padding)
        if hi - lo + 1 < min_span:
            deficit = min_span - (hi - lo + 1)
            lo = max(12, lo - deficit // 2)
            hi = min(108, lo + min_span - 1)
            lo = max(12, hi - min_span + 1)
        # Snap to C octave boundaries when close for clean musical grid
        lo = (lo // 12) * 12
        hi = min(108, ((hi + 11) // 12) * 12)
        return cls(lo, hi)


@dataclass(frozen=True)
class BarWindow:
    """Immutable bar slice specification (1 bar = 8 chunks = 192 steps)."""
    num_bars: int = 8
    steps_per_bar: int = 192
    chunks_per_bar: int = 8

    def __post_init__(self) -> None:
        if self.num_bars <= 0:
            raise ValueError(f"num_bars must be positive, got {self.num_bars}")

    @property
    def num_chunks(self) -> int:
        return self.num_bars * self.chunks_per_bar

    @property
    def total_steps(self) -> int:
        return self.num_bars * self.steps_per_bar


@dataclass(frozen=True)
class TrackSlotSpec:
    """Immutable metadata and visual identity for one of the 4 MidiVision track slots."""
    slot_index: int
    is_drum: bool
    program: int
    raw_name: str
    is_active: bool
    color: RgbColor

    def __post_init__(self) -> None:
        if not (0 <= self.slot_index < 4):
            raise ValueError(f"slot_index must be in 0..3, got {self.slot_index}")

    @property
    def display_label(self) -> str:
        if not self.is_active:
            return f"Slot {self.slot_index + 1}: Empty"
        if self.is_drum:
            return "Drums"
        cleaned = self.raw_name.strip()
        gm_family = _GM_FAMILY_NAMES[min(15, max(0, self.program // 8))]
        if not cleaned or cleaned.startswith("inst_") or cleaned.startswith("track_"):
            return f"{gm_family} (Prog {self.program})"
        # Keep concise readable name
        short = cleaned[:16].strip()
        if short.lower() == gm_family.lower():
            return gm_family
        return f"{short} ({gm_family})"

    @property
    def normalized_rgb(self) -> np.ndarray:
        return np.asarray(self.color, dtype=np.float32) / 255.0


@dataclass(frozen=True)
class QuantizedNote:
    """Immutable decoded MIDI note event on the quantized step grid."""
    slot: int
    start_step: int
    end_step: int
    pitch: int
    velocity: int

    @property
    def duration_steps(self) -> int:
        return max(1, self.end_step - self.start_step)

    @property
    def velocity_norm(self) -> float:
        return float(np.clip(self.velocity / 127.0, 0.0, 1.0))


@dataclass(frozen=True)
class SlotMetrics:
    """Per-slot reconstruction accuracy metrics."""
    slot_index: int
    note_f1: float
    onset_f1: float
    velocity_mae: float


@dataclass(frozen=True)
class StageMetrics:
    """Global and per-slot reconstruction metrics for a single decoded frame."""
    note_f1: float
    note_precision: float
    note_recall: float
    onset_f1: float
    onset_precision: float
    onset_recall: float
    velocity_mae: float
    relative_quant_error: float
    decoded_note_count: int
    slot_metrics: Tuple[SlotMetrics, ...]

    @property
    def velocity_accuracy(self) -> float:
        return float(np.clip(1.0 - self.velocity_mae, 0.0, 1.0))


@dataclass(frozen=True)
class StageDescriptor:
    """Human-readable metadata describing the current animation stage."""
    stage_id: str
    badge_text: str
    title_ja: str
    subtitle_ja: str
    active_layers_mask: Tuple[bool, bool, bool, bool]
    highlight_layer_index: int | None  # 0..3 when a specific RVQ layer is being emphasized
    progress_in_stage: float           # [0.0, 1.0]
    timeline_progress: float           # [0.0, 1.0]


@dataclass(frozen=True)
class DecodedFrameState:
    """Complete immutable domain state needed to render a single animation frame."""
    frame_index: int
    total_frames: int
    descriptor: StageDescriptor
    probs_flat: np.ndarray              # [8, total_steps, 128] float32 in [0, 1]
    prev_probs_flat: np.ndarray | None  # [8, total_steps, 128] float32 of anchor stage for delta view
    notes: Tuple[QuantizedNote, ...]
    metrics: StageMetrics


@dataclass(frozen=True)
class AnimationConfig:
    """Immutable top-level configuration for video generation."""
    midi_path: Path
    output_mp4_path: Path
    output_gif_path: Path | None = None
    export_stage_midis_dir: Path | None = None
    model_config_path: Path = Path("configs/models/mortm5/m4_deeprvq_g4l4_d128_k2048.json")
    model_ckpt_path: Path = Path(
        "out/models/mortm5/m4_deeprvq_g4l4_d128_k2048/MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth"
    )
    strategy_name: str = "noise_to_fine"
    bars: int = 8
    fps: int = 30
    resolution: Resolution = Resolution(1920, 1080)
    seed: int = 42
    mask_inactive_slots: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.fps > 120:
            raise ValueError(f"fps must be in 1..120, got {self.fps}")
        if self.bars <= 0:
            raise ValueError(f"bars must be positive, got {self.bars}")
