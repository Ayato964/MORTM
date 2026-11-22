from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from mortm.models.modules.config import SLOT_CODE_EMPTY
from mortm.utils.convert_pianoroll import CHUNKS_PER_BAR, CHUNK_STEPS, STEPS_PER_BAR


@dataclass(frozen=True)
class SamplingSpec:
    """
    Immutable Zero-Favoritism sampling configuration (§5.2.B).
    Applies identical temperature, top-p, top-k, min-p, and repetition penalty across
    all paradigms, residual layers, and attribute heads.
    """

    temperature: float = 1.05
    fine_temperature: float = 1.05
    top_p: float = 0.95
    fine_top_p: float = 0.95
    top_k: int = 0
    fine_top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0 or self.fine_temperature <= 0.0:
            raise ValueError(
                f"Temperatures must be > 0, got ({self.temperature}, {self.fine_temperature})"
            )
        if not (0.0 < self.top_p <= 1.0) or not (0.0 < self.fine_top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got ({self.top_p}, {self.fine_top_p})")
        if self.top_k < 0 or self.fine_top_k < 0:
            raise ValueError(f"top_k must be >= 0, got ({self.top_k}, {self.fine_top_k})")
        if not (0.0 <= self.min_p < 1.0):
            raise ValueError(f"min_p must be in [0, 1), got {self.min_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "fine_temperature": self.fine_temperature,
            "top_p": self.top_p,
            "fine_top_p": self.fine_top_p,
            "top_k": self.top_k,
            "fine_top_k": self.fine_top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass(frozen=True)
class DensityTierSpec:
    """Immutable note-density bucket specification for §5.1 context & latency scaling."""

    tier_id: str
    label: str
    min_notes_per_bar: float
    max_notes_per_bar: Optional[float]

    def __post_init__(self) -> None:
        if self.min_notes_per_bar < 0.0:
            raise ValueError(f"min_notes_per_bar must be >= 0, got {self.min_notes_per_bar}")
        if self.max_notes_per_bar is not None and self.max_notes_per_bar <= self.min_notes_per_bar:
            raise ValueError(
                f"max_notes_per_bar ({self.max_notes_per_bar}) must exceed min ({self.min_notes_per_bar})"
            )

    def contains(self, notes_per_bar: float) -> bool:
        if notes_per_bar < self.min_notes_per_bar:
            return False
        if self.max_notes_per_bar is not None and notes_per_bar >= self.max_notes_per_bar:
            return False
        return True


@dataclass(frozen=True)
class ModelCandidateSpec:
    """Immutable specification of a Stage-3 paradigm model candidate."""

    key: str
    display_name: str
    paradigm: str
    config_path: str
    checkpoint_path: str
    vq_checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.key or not self.config_path or not self.checkpoint_path:
            raise ValueError(f"ModelCandidateSpec requires non-empty key/config/checkpoint: {self}")


class TestSegmentItem:
    """
    Domain entity representing one canonical 16-bar unseen test segment (`[chunk_start, chunk_end)`).
    Encapsulates read-only quantized note arrays and deterministic prompt/continuation slicing.
    """

    def __init__(
        self,
        item_id: str,
        song_index: int,
        midi_path: str,
        chunk_start: int,
        chunk_end: int,
        slot_codes: Sequence[int],
        canonical_notes: np.ndarray,
    ) -> None:
        if chunk_end <= chunk_start:
            raise ValueError(f"Invalid chunk range [{chunk_start}, {chunk_end})")
        if len(slot_codes) != 4:
            raise ValueError(f"Expected 4 slot_codes, got {slot_codes}")
        arr = np.asarray(canonical_notes, dtype=np.int32).copy()
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"Expected canonical_notes [M, 6], got {arr.shape}")
        arr.setflags(write=False)

        self._item_id = str(item_id)
        self._song_index = int(song_index)
        self._midi_path = str(midi_path)
        self._chunk_start = int(chunk_start)
        self._chunk_end = int(chunk_end)
        self._slot_codes = tuple(int(c) for c in slot_codes)
        self._canonical_notes = arr

    @property
    def item_id(self) -> str:
        return self._item_id

    @property
    def song_index(self) -> int:
        return self._song_index

    @property
    def midi_path(self) -> str:
        return self._midi_path

    @property
    def chunk_start(self) -> int:
        return self._chunk_start

    @property
    def chunk_end(self) -> int:
        return self._chunk_end

    @property
    def slot_codes(self) -> Tuple[int, ...]:
        return self._slot_codes

    @property
    def num_chunks(self) -> int:
        return self._chunk_end - self._chunk_start

    @property
    def num_bars(self) -> int:
        return self.num_chunks // CHUNKS_PER_BAR

    @property
    def active_slot_count(self) -> int:
        return sum(1 for c in self._slot_codes if c != SLOT_CODE_EMPTY)

    @property
    def sounding_track_count(self) -> int:
        if len(self._canonical_notes) == 0:
            return 0
        return int(len(np.unique(self._canonical_notes[:, 2])))

    @property
    def total_notes(self) -> int:
        return int(len(self._canonical_notes))

    @property
    def notes_per_bar(self) -> float:
        return float(self.total_notes) / max(1.0, float(self.num_bars))

    def canonical_notes_view(self) -> np.ndarray:
        return self._canonical_notes

    def to_segment_5col_notes(self) -> np.ndarray:
        """Return read-only `[M, 5]` `(slot, start_step, end_step, pitch, velocity)` in `[0, num_chunks*24)`."""
        if len(self._canonical_notes) == 0:
            out = np.zeros((0, 5), dtype=np.int32)
            out.setflags(write=False)
            return out
        max_steps = self.num_chunks * CHUNK_STEPS
        starts = self._canonical_notes[:, 0] * STEPS_PER_BAR + self._canonical_notes[:, 1]
        ends = np.minimum(max_steps, starts + self._canonical_notes[:, 4])
        slots = self._canonical_notes[:, 2]
        pitches = self._canonical_notes[:, 3]
        vels = self._canonical_notes[:, 5]
        out = np.stack([slots, starts, ends, pitches, vels], axis=1).astype(np.int32, copy=False)
        out.setflags(write=False)
        return out

    def prompt_5col_notes(self, prompt_chunks: int = 32) -> np.ndarray:
        """Return `[Mp, 5]` `(slot, start_step, end_step, pitch, velocity)` clipped to `[0, prompt_chunks*24)`."""
        full = self.to_segment_5col_notes()
        limit_step = int(prompt_chunks) * CHUNK_STEPS
        if len(full) == 0 or limit_step <= 0:
            return np.zeros((0, 5), dtype=np.int32)
        mask = (full[:, 1] < limit_step) & (full[:, 2] > 0)
        if not np.any(mask):
            return np.zeros((0, 5), dtype=np.int32)
        sliced = full[mask].copy()
        sliced[:, 2] = np.minimum(limit_step, sliced[:, 2])
        valid = sliced[:, 2] > sliced[:, 1]
        return sliced[valid]

    def metadata_dict(self) -> Dict[str, object]:
        return {
            "item_id": self._item_id,
            "song_index": self._song_index,
            "midi_path": self._midi_path,
            "chunk_start": self._chunk_start,
            "chunk_end": self._chunk_end,
            "slot_codes": list(self._slot_codes),
            "active_slot_count": self.active_slot_count,
            "sounding_track_count": self.sounding_track_count,
            "total_notes": self.total_notes,
            "notes_per_bar": round(self.notes_per_bar, 3),
        }


@dataclass(frozen=True)
class RolloutTelemetry:
    """Immutable autoregressive execution profile for one segment rollout."""

    prompt_steps: int
    generated_steps: int
    total_steps: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    generated_bars: float
    peak_active_kv_mib: float
    kv_bar_trajectory_mib: Tuple[Tuple[int, float], ...] = ()

    @property
    def ms_per_bar(self) -> float:
        return self.total_ms / max(1e-6, self.generated_bars)

    @property
    def decode_ms_per_bar(self) -> float:
        return self.decode_ms / max(1e-6, self.generated_bars)

    @property
    def bars_per_sec(self) -> float:
        return self.generated_bars / max(1e-9, self.total_ms / 1000.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt_steps": self.prompt_steps,
            "generated_steps": self.generated_steps,
            "total_steps": self.total_steps,
            "prefill_ms": round(self.prefill_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "generated_bars": self.generated_bars,
            "ms_per_bar": round(self.ms_per_bar, 3),
            "decode_ms_per_bar": round(self.decode_ms_per_bar, 3),
            "bars_per_sec": round(self.bars_per_sec, 3),
            "peak_active_kv_mib": round(self.peak_active_kv_mib, 4),
            "kv_bar_trajectory_mib": [
                {"bar": int(b), "kv_mib": round(float(m), 4)}
                for b, m in self.kv_bar_trajectory_mib
            ],
        }


@dataclass(frozen=True)
class MusPySongMetrics:
    """Immutable per-song MusPy continuation metrics (§5.2.C)."""

    pitch_class_entropy: float
    scale_consistency: float
    groove_consistency: float
    note_count: int
    melodic_note_count: int

    @property
    def has_valid_pce(self) -> bool:
        return math.isfinite(self.pitch_class_entropy)

    @property
    def has_valid_sc(self) -> bool:
        return math.isfinite(self.scale_consistency)

    @property
    def has_valid_gc(self) -> bool:
        return math.isfinite(self.groove_consistency)

    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "pitch_class_entropy": (
                round(self.pitch_class_entropy, 6) if self.has_valid_pce else None
            ),
            "scale_consistency": (
                round(self.scale_consistency, 6) if self.has_valid_sc else None
            ),
            "groove_consistency": (
                round(self.groove_consistency, 6) if self.has_valid_gc else None
            ),
            "note_count": self.note_count,
            "melodic_note_count": self.melodic_note_count,
        }


DEFAULT_SAMPLING_SPEC = SamplingSpec(
    temperature=1.05,
    fine_temperature=1.05,
    top_p=0.95,
    fine_top_p=0.95,
    top_k=0,
    fine_top_k=0,
    min_p=0.0,
    repetition_penalty=1.0,
)

DEFAULT_DENSITY_TIERS: Tuple[DensityTierSpec, ...] = (
    DensityTierSpec("sparse", "Sparse (<15 Notes/Bar)", 0.0, 15.0),
    DensityTierSpec("light", "Light (15-30 Notes/Bar)", 15.0, 30.0),
    DensityTierSpec("medium", "Medium (30-50 Notes/Bar)", 30.0, 50.0),
    DensityTierSpec("dense", "Dense (50-80 Notes/Bar)", 50.0, 80.0),
    DensityTierSpec("tutti", "Tutti (80+ Notes/Bar)", 80.0, None),
)

DEFAULT_STAGE3_CANDIDATES: Tuple[ModelCandidateSpec, ...] = (
    ModelCandidateSpec(
        key="symic_g4l4_160m_depth_4ep",
        display_name="SyMIc-G4L4 + Depth (160M, 4ep)",
        paradigm="grouped",
        config_path="configs/models/mortm5/symic_main/symic_g4l4_160m.json",
        checkpoint_path=(
            "out/symic_main/symic_g4l4_160m_depth_4ep/"
            "MORTM5.SyMIc_G4L4_160M_Depth_Scratch_16bar_4ep_muon_wsd_latest.pth"
        ),
        vq_checkpoint_path=(
            "out/models/mortm5/m4_deeprvq_g4l4_d128_k2048/"
            "MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth"
        ),
    ),
    ModelCandidateSpec(
        key="symic_g4l4_160m_carl_v2",
        display_name="SyMIc-G4L4 + Depth + CARL (160M, R4)",
        paradigm="grouped",
        config_path="configs/models/mortm5/symic_main/symic_g4l4_160m.json",
        checkpoint_path="out/carl/symic_g4l4_160m_carl_v2/SyMIc-gem.carl_latest.pth",
        vq_checkpoint_path=(
            "out/models/mortm5/m4_deeprvq_g4l4_d128_k2048/"
            "MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth"
        ),
    ),
    ModelCandidateSpec(
        key="octuple_160m",
        display_name="Octuple-MMT (160M, 1ep)",
        paradigm="octuple",
        config_path="configs/models/mortm5/symic_main/octuple_160m.json",
        checkpoint_path=(
            "out/symic_main/octuple_160m/MORTM5.Octuple_MMT_160M_16bar_muon_wsd_latest.pth"
        ),
    ),
    ModelCandidateSpec(
        key="remi_160m",
        display_name="Multitrack-REMI (160M, 1ep)",
        paradigm="remi",
        config_path="configs/models/mortm5/symic_main/remi_160m.json",
        checkpoint_path=(
            "out/symic_main/remi_160m/MORTM5.Multitrack_REMI_160M_16bar_muon_wsd_latest.pth"
        ),
    ),
)
