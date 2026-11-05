"""
mortm/utils/symbolic_tokenizers.py

Domain-driven symbolic music segment tokenizers (Octuple-MMT, extensible via Strategy/Registry
to Multitrack-REMI and future symbolic paradigms) built strictly on the Single Source of Truth
`extract_quantized_notes_and_slots` (`mortm/utils/convert_pianoroll.py`).

Guarantees:
1. 1-to-1 song and 16-bar non-overlapping segment boundary parity (`max_chunks=128`, `min_chunks=8`,
   `stride_chunks=128`) with `MORTM5MemmapDataset` (`SyMIc-G4L4`).
2. Exact cross-segment sustain clipping (`start < s0 < end` -> clipped at `local_step = 0` with
   residual duration `min(s1, end) - s0`) and `(max_dur, max_vel)` collision merging matching
   `rasterize_notes_to_chunks_u8`.
3. Shared special & 4-slot instrument conditioning vocabulary IDs `0..132` (`INST_RVQ_TOKEN_OFFSET = 133`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mortm.models.modules.config import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_EMPTY_TOKEN_ID,
    INST_PROGRAM_OFFSET,
    INST_RVQ_TOKEN_OFFSET,
    NUM_SLOT_TOKENS,
    PAD_TOKEN_ID,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
    slot_code_to_token_id,
    token_id_to_slot_code,
)
from mortm.utils.convert_pianoroll import (
    CHUNK_STEPS,
    MAX_TRACKS,
    STEPS_PER_BAR,
    _notes_to_midi,
    extract_quantized_notes_and_slots,
    rasterize_notes_to_chunks_float,
    rasterize_notes_to_chunks_u8,
    slot_codes_to_roll_meta,
)


# ==============================================================================
# 1. Immutable Value Objects (Segment Windowing & Quantization Grids)
# ==============================================================================


@dataclass(frozen=True)
class SegmentWindowSpec:
    """
    Immutable 16-bar non-overlapping window specification guaranteed to produce
    exact 1-to-1 segment boundaries with `MORTM5MemmapDataset._build_windows()`.
    """

    max_chunks: int = 128
    min_chunks: int = 8
    stride_chunks: int = 128

    def __post_init__(self) -> None:
        if self.min_chunks <= 0 or self.max_chunks < self.min_chunks or self.stride_chunks <= 0:
            raise ValueError(
                f"Invalid SegmentWindowSpec: min_chunks={self.min_chunks}, "
                f"max_chunks={self.max_chunks}, stride_chunks={self.stride_chunks}"
            )

    @property
    def max_bars(self) -> int:
        return (self.max_chunks * CHUNK_STEPS + STEPS_PER_BAR - 1) // STEPS_PER_BAR

    @property
    def max_steps(self) -> int:
        return self.max_chunks * CHUNK_STEPS

    def compute_song_segments(self, num_chunks: int) -> List[Tuple[int, int, int, int]]:
        """
        Return list of `(chunk_start, chunk_end, is_first, is_last)` tuples matching
        `MORTM5MemmapDataset._build_windows()` identically.
        """
        n_chunks = int(num_chunks)
        if n_chunks < self.min_chunks:
            return []
        if n_chunks <= self.max_chunks:
            return [(0, n_chunks, 1, 1)]

        segments: List[Tuple[int, int, int, int]] = []
        pos = 0
        while pos < n_chunks:
            rem = n_chunks - pos
            if rem < self.min_chunks:
                break
            win_len = min(self.max_chunks, rem)
            is_first = 1 if pos == 0 else 0
            is_last = 1 if (pos + win_len >= n_chunks) else 0
            segments.append((pos, pos + win_len, is_first, is_last))
            if pos + win_len >= n_chunks:
                break
            pos += self.stride_chunks
        return segments


@dataclass(frozen=True)
class SymbolicGridSpec:
    """
    Shared quantized attribute grid specification for `Octuple-MMT` and `Multitrack-REMI`:
    - `dur`: 312 bins
        * Bins `0..191` (`192` bins): exact 1-step durations `1..192` (covers 99.40% of GMD losslessly).
        * Bins `192..311` (`120` bins): half-beat (`24`-step) durations `216..3072` (up to 16 bars).
    - `vel`: 32 uniform bins (`0..31` <-> MIDI velocity `1..127`).
    """

    exact_dur_max: int = 192
    coarse_dur_step: int = 24
    max_dur_steps: int = 3072
    velocity_bins: int = 32

    @property
    def coarse_dur_bins(self) -> int:
        return (self.max_dur_steps - (self.exact_dur_max + self.coarse_dur_step)) // self.coarse_dur_step + 1

    @property
    def total_dur_bins(self) -> int:
        return self.exact_dur_max + self.coarse_dur_bins

    def build_dur_lut(self) -> np.ndarray:
        exact = np.arange(1, self.exact_dur_max + 1, dtype=np.int32)
        coarse = np.arange(
            self.exact_dur_max + self.coarse_dur_step,
            self.max_dur_steps + 1,
            self.coarse_dur_step,
            dtype=np.int32,
        )
        return np.concatenate([exact, coarse], axis=0)

    def quantize_dur_np(self, dur_steps: np.ndarray) -> np.ndarray:
        d = np.clip(np.asarray(dur_steps, dtype=np.int32), 1, self.max_dur_steps)
        exact_bins = d - 1
        rounded_mult = np.rint(d.astype(np.float64) / float(self.coarse_dur_step)).astype(np.int32)
        clamped_mult = np.clip(
            rounded_mult,
            self.exact_dur_max // self.coarse_dur_step,
            self.max_dur_steps // self.coarse_dur_step,
        )
        # mult == 8 (192 steps) maps to exact bin 191; mult >= 9 (216..3072) maps to 192 + (mult - 9)
        coarse_bins = np.where(
            clamped_mult <= (self.exact_dur_max // self.coarse_dur_step),
            self.exact_dur_max - 1,
            self.exact_dur_max + (clamped_mult - (self.exact_dur_max // self.coarse_dur_step + 1)),
        )
        return np.where(d <= self.exact_dur_max, exact_bins, coarse_bins).astype(np.int32, copy=False)

    def dequantize_dur_np(self, dur_bins: np.ndarray, lut: Optional[np.ndarray] = None) -> np.ndarray:
        table = lut if lut is not None else self.build_dur_lut()
        b = np.clip(np.asarray(dur_bins, dtype=np.int32), 0, len(table) - 1)
        return table[b]

    def quantize_vel_np(self, velocity: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(velocity, dtype=np.int32), 0, 127)
        return (v * self.velocity_bins) // 128

    def dequantize_vel_np(self, vel_bins: np.ndarray) -> np.ndarray:
        b = np.clip(np.asarray(vel_bins, dtype=np.int32), 0, self.velocity_bins - 1)
        step = 128 // self.velocity_bins
        return np.clip(b * step + (step // 2), 1, 127).astype(np.int32, copy=False)


DEFAULT_SEGMENT_SPEC = SegmentWindowSpec(max_chunks=128, min_chunks=8, stride_chunks=128)
DEFAULT_GRID_SPEC = SymbolicGridSpec()
_DUR_DEQUANT_LUT: np.ndarray = DEFAULT_GRID_SPEC.build_dur_lut()


def extract_segment_canonical_notes(
    notes_arr: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    max_bars: int = 16,
) -> np.ndarray:
    """
    Slice quantized song notes `int32 [N, 5]` `(slot, start, end, pitch, velocity)` into
    a segment `[chunk_start, chunk_end)` with exact cross-segment sustain clipping and
    canonical `(bar, pos, slot, pitch)` ordering.

    Returns:
        `int32 [M, 6]` with columns `(bar, pos, slot, pitch, dur_steps, velocity)`.
    """
    c0 = int(chunk_start)
    c1 = int(chunk_end)
    if c1 <= c0 or len(notes_arr) == 0:
        return np.zeros((0, 6), dtype=np.int32)

    s0 = c0 * CHUNK_STEPS
    s1 = min(c1 * CHUNK_STEPS, s0 + int(max_bars) * STEPS_PER_BAR)

    slots = notes_arr[:, 0]
    starts = notes_arr[:, 1]
    ends = notes_arr[:, 2]
    pitches = notes_arr[:, 3]
    vels = notes_arr[:, 4]

    active_mask = (
        (starts < s1)
        & (ends > s0)
        & (ends > starts)
        & (slots >= 0)
        & (slots < MAX_TRACKS)
        & (pitches >= 0)
        & (pitches < 128)
    )
    if not np.any(active_mask):
        return np.zeros((0, 6), dtype=np.int32)

    sel_slots = slots[active_mask].astype(np.int32, copy=False)
    local_starts = np.maximum(0, starts[active_mask] - s0).astype(np.int32, copy=False)
    local_ends = (np.minimum(s1, ends[active_mask]) - s0).astype(np.int32, copy=False)
    durs = local_ends - local_starts
    valid_dur = durs > 0
    if not np.all(valid_dur):
        sel_slots = sel_slots[valid_dur]
        local_starts = local_starts[valid_dur]
        durs = durs[valid_dur]
        sel_pitches = pitches[active_mask][valid_dur].astype(np.int32, copy=False)
        sel_vels = np.clip(vels[active_mask][valid_dur], 1, 127).astype(np.int32, copy=False)
    else:
        sel_pitches = pitches[active_mask].astype(np.int32, copy=False)
        sel_vels = np.clip(vels[active_mask], 1, 127).astype(np.int32, copy=False)

    if len(sel_slots) == 0:
        return np.zeros((0, 6), dtype=np.int32)

    bars = local_starts // STEPS_PER_BAR
    poss = local_starts % STEPS_PER_BAR

    # Vectorized O(N log N) canonical sort on (bar, pos, slot, pitch) + O(N) collision merge
    coord_key = (
        ((bars.astype(np.int64) * STEPS_PER_BAR + poss.astype(np.int64)) * MAX_TRACKS + sel_slots.astype(np.int64))
        * 128
        + sel_pitches.astype(np.int64)
    )
    order = np.argsort(coord_key, kind="mergesort")
    sorted_coord = coord_key[order]
    bars_s = bars[order]
    poss_s = poss[order]
    slots_s = sel_slots[order]
    pitches_s = sel_pitches[order]
    durs_s = durs[order]
    vels_s = sel_vels[order]

    if len(sorted_coord) > 1 and np.any(sorted_coord[1:] == sorted_coord[:-1]):
        first_idx = np.flatnonzero(np.concatenate(([True], sorted_coord[1:] != sorted_coord[:-1])))
        bars_s = bars_s[first_idx]
        poss_s = poss_s[first_idx]
        slots_s = slots_s[first_idx]
        pitches_s = pitches_s[first_idx]
        durs_s = np.maximum.reduceat(durs_s, first_idx)
        vels_s = np.maximum.reduceat(vels_s, first_idx)

    return np.stack([bars_s, poss_s, slots_s, pitches_s, durs_s, vels_s], axis=1).astype(np.int32, copy=False)


# ==============================================================================
# 2. Octuple-MMT Vocabulary Spec & Strategy Hierarchy
# ==============================================================================

OCTUPLE_TUPLE_SIZE: int = 6
OCTUPLE_ATTR_NAMES: Tuple[str, ...] = ("bar", "pos", "slot", "pitch", "dur", "vel")
OCTUPLE_ATTR_SIZES: Tuple[int, ...] = (16, 192, 4, 128, 312, 32)  # sum = 684


@dataclass(frozen=True)
class OctupleVocabSpec:
    """
    Immutable vocabulary specification for `Octuple-MMT`:
    - Shared special & instrument IDs: `0..132` (`token_offset = 133`)
    - Disjoint 6-attribute slices:
        0 (`bar`):   `133..148` (`16`)
        1 (`pos`):   `149..340` (`192`)
        2 (`slot`):  `341..344` (`4`)
        3 (`pitch`): `345..472` (`128`)
        4 (`dur`):   `473..784` (`312`)
        5 (`vel`):   `785..816` (`32`)
    - Total global vocabulary size = `817`.
    """

    token_offset: int = INST_RVQ_TOKEN_OFFSET
    attr_names: Tuple[str, ...] = OCTUPLE_ATTR_NAMES
    attr_sizes: Tuple[int, ...] = OCTUPLE_ATTR_SIZES

    @property
    def tuple_size(self) -> int:
        return len(self.attr_sizes)

    @property
    def attr_offsets(self) -> Tuple[int, ...]:
        offsets: List[int] = []
        cur = int(self.token_offset)
        for sz in self.attr_sizes:
            offsets.append(cur)
            cur += int(sz)
        return tuple(offsets)

    @property
    def attr_limits(self) -> Tuple[int, ...]:
        return tuple(off + int(sz) for off, sz in zip(self.attr_offsets, self.attr_sizes))

    @property
    def total_attr_tokens(self) -> int:
        return sum(int(sz) for sz in self.attr_sizes)

    @property
    def vocab_size(self) -> int:
        return int(self.token_offset) + self.total_attr_tokens

    @property
    def max_attr_size(self) -> int:
        return max(int(sz) for sz in self.attr_sizes)

    @property
    def max_head_vocab_size(self) -> int:
        return int(self.token_offset) + self.max_attr_size


DEFAULT_OCTUPLE_SPEC = OctupleVocabSpec()

REMI_ATTR_NAMES: Tuple[str, ...] = ("bar", "pos", "slot", "pitch", "dur", "vel")
REMI_ATTR_SIZES: Tuple[int, ...] = (1, 192, 4, 128, 312, 32)  # sum = 669


@dataclass(frozen=True)
class REMIVocabSpec:
    """
    Immutable vocabulary specification for `Multitrack-REMI`:
    - Shared special & instrument IDs: `0..132` (`token_offset = 133`)
    - Disjoint 1D event slices:
        0 (`bar`):   `133`      (`1` token: `<BAR>`)
        1 (`pos`):   `134..325` (`192` tokens: `<POS_0..191>`)
        2 (`slot`):  `326..329` (`4` tokens: `<SLOT_0..3>`)
        3 (`pitch`): `330..457` (`128` tokens: `<PITCH_0..127>`)
        4 (`dur`):   `458..769` (`312` tokens: `<DUR_0..311>`)
        5 (`vel`):   `770..801` (`32` tokens: `<VEL_0..31>`)
    - Total global vocabulary size = `802`.
    """

    token_offset: int = INST_RVQ_TOKEN_OFFSET
    attr_names: Tuple[str, ...] = REMI_ATTR_NAMES
    attr_sizes: Tuple[int, ...] = REMI_ATTR_SIZES
    max_bars: int = 16

    def __post_init__(self) -> None:
        if len(self.attr_sizes) != 6 or int(self.attr_sizes[0]) != 1:
            raise ValueError(
                f"REMIVocabSpec expects 6 attr_sizes starting with bar_count=1, got {self.attr_sizes}"
            )
        if any(int(s) <= 0 for s in self.attr_sizes) or int(self.max_bars) <= 0:
            raise ValueError(
                f"REMIVocabSpec attr_sizes ({self.attr_sizes}) and max_bars ({self.max_bars}) must be positive."
            )

    @property
    def attr_offsets(self) -> Tuple[int, ...]:
        offsets: List[int] = []
        cur = int(self.token_offset)
        for sz in self.attr_sizes:
            offsets.append(cur)
            cur += int(sz)
        return tuple(offsets)

    @property
    def attr_limits(self) -> Tuple[int, ...]:
        return tuple(off + int(sz) for off, sz in zip(self.attr_offsets, self.attr_sizes))

    @property
    def bar_token_id(self) -> int:
        return int(self.token_offset)

    @property
    def pos_offset(self) -> int:
        return int(self.token_offset) + int(self.attr_sizes[0])

    @property
    def pos_size(self) -> int:
        return int(self.attr_sizes[1])

    @property
    def slot_offset(self) -> int:
        return self.pos_offset + self.pos_size

    @property
    def slot_size(self) -> int:
        return int(self.attr_sizes[2])

    @property
    def pitch_offset(self) -> int:
        return self.slot_offset + self.slot_size

    @property
    def pitch_size(self) -> int:
        return int(self.attr_sizes[3])

    @property
    def dur_offset(self) -> int:
        return self.pitch_offset + self.pitch_size

    @property
    def dur_size(self) -> int:
        return int(self.attr_sizes[4])

    @property
    def vel_offset(self) -> int:
        return self.dur_offset + self.dur_size

    @property
    def vel_size(self) -> int:
        return int(self.attr_sizes[5])

    @property
    def total_attr_tokens(self) -> int:
        return sum(int(sz) for sz in self.attr_sizes)

    @property
    def vocab_size(self) -> int:
        return int(self.token_offset) + self.total_attr_tokens


DEFAULT_REMI_SPEC = REMIVocabSpec()


class AbstractSymbolicSegmentTokenizer(ABC):
    """
    Strategy interface for 16-bar segment symbolic music tokenizers.
    Allows new paradigms (e.g., `Multitrack-REMI`) to plug into corpus conversion,
    dataset loading, and unified evaluation without modifying existing classes (OCP).
    """

    @property
    @abstractmethod
    def tokenizer_key(self) -> str: ...

    @property
    @abstractmethod
    def model_mode(self) -> str: ...

    @property
    @abstractmethod
    def unit_width(self) -> int: ...

    @property
    def step_unit_width(self) -> int:
        return self.unit_width

    @property
    @abstractmethod
    def token_offset(self) -> int: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    def max_segment_bars(self) -> int:
        return 16

    @abstractmethod
    def encode_canonical_segment_notes(
        self,
        canonical_notes: np.ndarray,
        num_chunks: int,
    ) -> np.ndarray:
        """Encode `int32 [M, 6]` `(bar, pos, slot, pitch, dur_steps, velocity)` into `uint16` flat body tokens."""

    def encode_song_segment(
        self,
        notes_arr: np.ndarray,
        chunk_start: int,
        chunk_end: int,
    ) -> np.ndarray:
        """Slice `[chunk_start, chunk_end)` from `notes_arr` and encode into `uint16` body tokens."""
        seg_notes = extract_segment_canonical_notes(
            notes_arr, chunk_start, chunk_end, max_bars=self.max_segment_bars
        )
        return self.encode_canonical_segment_notes(seg_notes, num_chunks=int(chunk_end - chunk_start))

    @staticmethod
    def filter_notes_by_slots_and_horizon(
        notes_arr: np.ndarray,
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> np.ndarray:
        """Filter `int32 [M, 5]` `(slot, start, end, pitch, velocity)` against active slots and `[0, num_chunks * 24)`."""
        if notes_arr is None or len(notes_arr) == 0:
            return np.zeros((0, 5), dtype=np.int32)
        arr = np.asarray(notes_arr, dtype=np.int32)
        slots = arr[:, 0]
        starts = arr[:, 1]
        ends = arr[:, 2]
        pitches = arr[:, 3]
        vels = np.clip(arr[:, 4], 1, 127)

        keep = (slots >= 0) & (slots < MAX_TRACKS) & (pitches >= 0) & (pitches < 128) & (starts >= 0)
        if slot_codes is not None and len(slot_codes) >= NUM_SLOT_TOKENS:
            active_lut = np.asarray([int(c) != SLOT_CODE_EMPTY for c in slot_codes[:NUM_SLOT_TOKENS]], dtype=bool)
            keep = keep & active_lut[np.clip(slots, 0, NUM_SLOT_TOKENS - 1)]

        if num_chunks is not None and int(num_chunks) > 0:
            max_step = int(num_chunks) * CHUNK_STEPS
            keep = keep & (starts < max_step)
            ends = np.minimum(max_step, ends)

        keep = keep & (ends > starts)
        if not np.any(keep):
            return np.zeros((0, 5), dtype=np.int32)
        return np.stack([slots[keep], starts[keep], ends[keep], pitches[keep], vels[keep]], axis=1).astype(
            np.int32, copy=False
        )

    @staticmethod
    def _deoverlap_same_pitch_notes_for_midi(notes_arr: np.ndarray) -> np.ndarray:
        """Vectorized O(M log M) same-(slot, pitch) collision merge & sustain clipping prior to SMF writing."""
        if len(notes_arr) <= 1:
            return notes_arr
        slots = notes_arr[:, 0].astype(np.int64)
        starts = notes_arr[:, 1].astype(np.int64)
        pitches = notes_arr[:, 3].astype(np.int64)
        step_span = max(1, int(starts.max()) + 1)
        key = (slots * 128 + pitches) * step_span + starts
        order = np.argsort(key, kind="mergesort")
        s_key = key[order]
        s_notes = notes_arr[order].copy()

        if len(s_key) > 1 and np.any(s_key[1:] == s_key[:-1]):
            first_idx = np.flatnonzero(np.concatenate(([True], s_key[1:] != s_key[:-1])))
            merged_ends = np.maximum.reduceat(s_notes[:, 2], first_idx)
            merged_vels = np.maximum.reduceat(s_notes[:, 4], first_idx)
            s_notes = s_notes[first_idx].copy()
            s_notes[:, 2] = merged_ends
            s_notes[:, 4] = merged_vels

        if len(s_notes) > 1:
            same_sp = (s_notes[:-1, 0] == s_notes[1:, 0]) & (s_notes[:-1, 3] == s_notes[1:, 3])
            if np.any(same_sp):
                s_notes[:-1, 2] = np.where(
                    same_sp,
                    np.minimum(s_notes[:-1, 2], s_notes[1:, 1]),
                    s_notes[:-1, 2],
                )

        chrono = np.lexsort((s_notes[:, 3], s_notes[:, 0], s_notes[:, 1]))
        return s_notes[chrono]

    @classmethod
    def extract_notes_from_chunks_f32(
        cls,
        chunks_f32: np.ndarray,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Extract `int32 [M, 5]` `(slot, start, end, pitch, velocity)` from `[N, 8, 24, 128]` or `[N, 8, 128, 24]`."""
        from mortm.utils.convert_pianoroll import TOTAL_CHANNELS, _flat_to_quantized_notes

        n_chunks = int(chunks_f32.shape[0])
        if n_chunks <= 0:
            return np.zeros((0, 5), dtype=np.int32)
        arr = chunks_f32
        if arr.ndim == 4 and arr.shape[1:] == (TOTAL_CHANNELS, 128, CHUNK_STEPS):
            arr = arr.transpose(0, 1, 3, 2)
        flat = arr.transpose(1, 0, 2, 3).reshape(TOTAL_CHANNELS, -1, 128)
        note_tuples = _flat_to_quantized_notes(flat)
        notes_arr = (
            np.asarray(note_tuples, dtype=np.int32)
            if note_tuples
            else np.zeros((0, 5), dtype=np.int32)
        )
        return cls.filter_notes_by_slots_and_horizon(notes_arr, slot_codes=slot_codes, num_chunks=n_chunks)

    def render_notes_to_chunks_u8(
        self,
        notes_arr: np.ndarray,
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        filtered = self.filter_notes_by_slots_and_horizon(
            notes_arr, slot_codes=slot_codes, num_chunks=int(num_chunks)
        )
        return rasterize_notes_to_chunks_u8(int(num_chunks), filtered)

    def render_notes_to_chunks_float(
        self,
        notes_arr: np.ndarray,
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        filtered = self.filter_notes_by_slots_and_horizon(
            notes_arr, slot_codes=slot_codes, num_chunks=int(num_chunks)
        )
        return rasterize_notes_to_chunks_float(int(num_chunks), filtered)

    def render_notes_to_midi(
        self,
        notes_arr: np.ndarray,
        output_path: Union[str, Path],
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> np.ndarray:
        eff_slots = (
            np.asarray(slot_codes, dtype=np.int16)
            if slot_codes is not None
            else np.asarray([SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY], dtype=np.int16)
        )
        filtered = self.filter_notes_by_slots_and_horizon(
            notes_arr, slot_codes=eff_slots, num_chunks=num_chunks
        )
        clean_notes = self._deoverlap_same_pitch_notes_for_midi(filtered)
        meta = slot_codes_to_roll_meta(eff_slots)
        note_tuples = [
            (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]))
            for r in clean_notes
        ]
        _notes_to_midi(note_tuples, meta, output_path)
        return eff_slots

    @abstractmethod
    def decode_tokens_to_notes(
        self,
        seq: Union[np.ndarray, Tensor, Sequence[int]],
        num_chunks: Optional[int] = None,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Decode token sequence back into `(notes_arr, slot_codes)`:
        - `notes_arr`: `int32 [M, 5]` `(slot, start, end, pitch, velocity)`
        - `slot_codes`: `int16 [4]` extracted from `<INST_...>` prefix (or `slot_codes` override if given).
        """

    def decode_tokens_to_chunks_u8(
        self,
        seq: Union[np.ndarray, Tensor, Sequence[int]],
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        notes_arr, eff_slots = self.decode_tokens_to_notes(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )
        chunks_u8 = rasterize_notes_to_chunks_u8(int(num_chunks), notes_arr)
        return chunks_u8, eff_slots

    def decode_tokens_to_chunks_float(
        self,
        seq: Union[np.ndarray, Tensor, Sequence[int]],
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        notes_arr, eff_slots = self.decode_tokens_to_notes(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )
        chunks_f32 = rasterize_notes_to_chunks_float(int(num_chunks), notes_arr)
        return chunks_f32, eff_slots

    def decode_tokens_to_midi(
        self,
        seq: Union[np.ndarray, Tensor, Sequence[int]],
        output_path: Union[str, Path],
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> np.ndarray:
        notes_arr, extracted_slots = self.decode_tokens_to_notes(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )
        return self.render_notes_to_midi(
            notes_arr,
            output_path=output_path,
            slot_codes=extracted_slots,
            num_chunks=num_chunks,
        )


class OctupleMMTTokenizer(AbstractSymbolicSegmentTokenizer):
    """
    6-attribute Compound-Word / MMT Tokenizer (`bar, pos, slot, pitch, dur, vel`)
    sharing prefix IDs `0..132` (`vocab_size = 817`).
    """

    def __init__(
        self,
        vocab_spec: OctupleVocabSpec = DEFAULT_OCTUPLE_SPEC,
        grid_spec: SymbolicGridSpec = DEFAULT_GRID_SPEC,
    ) -> None:
        self.vocab_spec = vocab_spec
        self.grid_spec = grid_spec
        self._attr_offsets_np = np.asarray(self.vocab_spec.attr_offsets, dtype=np.int32)
        self._attr_sizes_np = np.asarray(self.vocab_spec.attr_sizes, dtype=np.int32)
        self._dur_lut = self.grid_spec.build_dur_lut()

    @property
    def tokenizer_key(self) -> str:
        return "octuple_mmt"

    @property
    def model_mode(self) -> str:
        return "octuple"

    @property
    def unit_width(self) -> int:
        return self.vocab_spec.tuple_size

    @property
    def token_offset(self) -> int:
        return self.vocab_spec.token_offset

    @property
    def vocab_size(self) -> int:
        return self.vocab_spec.vocab_size

    @property
    def max_segment_bars(self) -> int:
        return int(self.vocab_spec.attr_sizes[0])

    def encode_canonical_segment_notes(
        self,
        canonical_notes: np.ndarray,
        num_chunks: int = 128,
    ) -> np.ndarray:
        """
        Encode canonical segment notes `int32 [M, 6]` `(bar, pos, slot, pitch, dur_steps, velocity)`
        into flat `uint16 [6 * M]` global token IDs.
        """
        if len(canonical_notes) == 0:
            return np.zeros((0,), dtype=np.uint16)

        max_bar_idx = self.vocab_spec.attr_sizes[0] - 1
        bars = np.clip(canonical_notes[:, 0], 0, max_bar_idx)
        poss = np.clip(canonical_notes[:, 1], 0, self.vocab_spec.attr_sizes[1] - 1)
        slots = np.clip(canonical_notes[:, 2], 0, self.vocab_spec.attr_sizes[2] - 1)
        pitches = np.clip(canonical_notes[:, 3], 0, self.vocab_spec.attr_sizes[3] - 1)
        dur_bins = self.grid_spec.quantize_dur_np(canonical_notes[:, 4])
        vel_bins = self.grid_spec.quantize_vel_np(canonical_notes[:, 5])

        local_cols = np.stack([bars, poss, slots, pitches, dur_bins, vel_bins], axis=1)
        global_tuples = local_cols + self._attr_offsets_np[None, :]
        return np.ascontiguousarray(global_tuples.reshape(-1), dtype=np.uint16)

    def build_full_sequence(
        self,
        body_tokens_u16: np.ndarray,
        slot_codes: Optional[Sequence[int]] = None,
        add_bos: bool = True,
        add_inst: bool = True,
        add_eos: bool = True,
    ) -> np.ndarray:
        """Build a full 1D `int64` sequence (`<BOS>*6 + <INST_0..3>*6 + body + <EOS>*6`)."""
        w = self.unit_width
        parts: List[np.ndarray] = []
        if add_bos:
            parts.append(np.full(w, BOS_TOKEN_ID, dtype=np.int64))
        if add_inst:
            codes = (
                list(slot_codes)
                if slot_codes is not None
                else [SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
            )
            inst_ids = np.asarray([slot_code_to_token_id(int(c)) for c in codes[:NUM_SLOT_TOKENS]], dtype=np.int64)
            parts.append(np.repeat(inst_ids, w))
        if body_tokens_u16.size > 0:
            parts.append(np.asarray(body_tokens_u16, dtype=np.int64).reshape(-1))
        if add_eos:
            parts.append(np.full(w, EOS_TOKEN_ID, dtype=np.int64))
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    def encode_notes_window_prompt(
        self,
        notes_arr: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        slot_codes: Sequence[int],
        append_eos: bool = False,
    ) -> Tensor:
        """
        Encode global `notes_arr` (`int32 [M, 5]` `(slot, start, end, pitch, velocity)`) over window
        `[chunk_start, chunk_end)` (`<= max_segment_bars * 8` chunks) into an `Octuple` prompt tensor `[1, S_soft, 6]`.
        """
        c0 = int(chunk_start)
        c1 = int(chunk_end)
        body_u16 = (
            self.encode_song_segment(notes_arr, c0, c1)
            if c1 > c0
            else np.zeros((0,), dtype=np.uint16)
        )
        seq_1d = self.build_full_sequence(
            body_u16,
            slot_codes=slot_codes,
            add_bos=True,
            add_inst=True,
            add_eos=append_eos,
        )
        return torch.from_numpy(seq_1d.reshape(1, -1, self.unit_width))

    def encode_midi_prompt(
        self,
        midi_path: Union[str, Path],
        prompt_chunks: int = 32,
        append_eos: bool = False,
    ) -> Tuple[Tensor, np.ndarray]:
        """
        Encode the first `prompt_chunks` chunks (e.g. `32 chunks = 4 bars`) of a MIDI file into
        an `Octuple` prompt tensor `[1, S_soft, 6]` ready for `OctupleTransformer.generate(prompt=...)`.
        """
        num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(midi_path)
        eff_chunks = max(0, min(int(num_chunks), int(prompt_chunks)))
        filtered_notes = self.filter_notes_by_slots_and_horizon(
            notes_arr, slot_codes=slot_codes.tolist(), num_chunks=eff_chunks
        )
        seq_3d = self.encode_notes_window_prompt(
            filtered_notes, 0, eff_chunks, slot_codes=slot_codes.tolist(), append_eos=append_eos
        )
        return seq_3d, slot_codes

    def encode_chunks_f32_prompt(
        self,
        chunks_f32: np.ndarray,
        slot_codes: Sequence[int],
        append_eos: bool = False,
    ) -> Tensor:
        """
        Encode an in-memory float32 piano-roll `[N_chunks, 8, 128, 24]` (or `[N_chunks, 8, 24, 128]`)
        into an `Octuple` prompt tensor `[1, S_soft, 6]` without temporary files.
        """
        n_chunks = max(0, int(chunks_f32.shape[0]))
        notes_arr = self.extract_notes_from_chunks_f32(chunks_f32, slot_codes=slot_codes)
        return self.encode_notes_window_prompt(
            notes_arr, 0, n_chunks, slot_codes=slot_codes, append_eos=append_eos
        )

    def decode_tokens_to_notes(
        self,
        seq: Union[np.ndarray, Tensor, Sequence[int]],
        num_chunks: Optional[int] = None,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(seq, Tensor):
            arr = seq.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            arr = np.asarray(seq, dtype=np.int64)

        w = self.unit_width
        flat = arr.reshape(-1)
        usable = (flat.size // w) * w
        if usable == 0:
            eff_slots_empty = np.asarray(slot_codes, dtype=np.int16) if slot_codes is not None else None
            return np.zeros((0, 5), dtype=np.int32), eff_slots_empty
        rows = flat[:usable].reshape(-1, w)

        # 1. Extract 4-slot instrument codes from prefix rows if present (override if explicitly supplied)
        col0 = rows[:, 0]
        inst_mask = (col0 >= INST_PROGRAM_OFFSET) & (col0 <= INST_EMPTY_TOKEN_ID)
        extracted_slots: Optional[np.ndarray] = None
        if np.count_nonzero(inst_mask) >= NUM_SLOT_TOKENS:
            inst_ids = col0[inst_mask][:NUM_SLOT_TOKENS]
            extracted_slots = np.asarray([token_id_to_slot_code(int(tid)) for tid in inst_ids], dtype=np.int16)
        eff_slots: Optional[np.ndarray] = (
            np.asarray(slot_codes, dtype=np.int16) if slot_codes is not None else extracted_slots
        )

        # 2. Truncate at first <EOS> row (if present)
        eos_positions = np.flatnonzero(np.any(rows == EOS_TOKEN_ID, axis=1))
        if len(eos_positions) > 0:
            rows = rows[: int(eos_positions[0])]

        if len(rows) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        # 3. Keep rows where all 6 columns fall inside their respective attribute ranges
        low = self._attr_offsets_np[None, :]
        high = low + self._attr_sizes_np[None, :]
        valid_note_rows = np.all((rows >= low) & (rows < high), axis=1)
        note_rows = rows[valid_note_rows]
        if len(note_rows) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        local_vals = (note_rows - low).astype(np.int32, copy=False)
        bars = local_vals[:, 0]
        poss = local_vals[:, 1]
        slots = local_vals[:, 2]
        pitches = local_vals[:, 3]
        durs = self.grid_spec.dequantize_dur_np(local_vals[:, 4], lut=self._dur_lut)
        vels = self.grid_spec.dequantize_vel_np(local_vals[:, 5])

        # Filter out notes targeting inactive SLOT_CODE_EMPTY tracks so piano-roll and MIDI outputs are 100% consistent
        if eff_slots is not None and len(eff_slots) >= NUM_SLOT_TOKENS:
            active_slot_lut = eff_slots[:NUM_SLOT_TOKENS] != SLOT_CODE_EMPTY
            slot_in_range = (slots >= 0) & (slots < NUM_SLOT_TOKENS)
            keep_active = slot_in_range & active_slot_lut[np.clip(slots, 0, NUM_SLOT_TOKENS - 1)]
            bars = bars[keep_active]
            poss = poss[keep_active]
            slots = slots[keep_active]
            pitches = pitches[keep_active]
            durs = durs[keep_active]
            vels = vels[keep_active]

        if len(slots) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        starts = bars * STEPS_PER_BAR + poss
        ends = starts + durs

        if num_chunks is not None and int(num_chunks) > 0:
            max_step = int(num_chunks) * CHUNK_STEPS
            in_bounds = starts < max_step
            slots = slots[in_bounds]
            starts = starts[in_bounds]
            ends = np.minimum(max_step, ends[in_bounds])
            pitches = pitches[in_bounds]
            vels = vels[in_bounds]

        if len(starts) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        ends = np.maximum(ends, starts + 1)
        notes_arr = np.stack([slots, starts, ends, pitches, vels], axis=1).astype(np.int32, copy=False)
        return notes_arr, eff_slots


class MultitrackREMITokenizer(AbstractSymbolicSegmentTokenizer):
    """
    1D Flat Event `Multitrack-REMI` Tokenizer sharing prefix IDs `0..132` (`vocab_size = 802`):
      `<BOS> <INST_0..3> (<BAR> (<POS_p> (<SLOT_s> <PITCH_k> <DUR_d> <VEL_v>)+)*)+ <EOS>`
    - Emits `<BAR>` for every bar `0 .. B_seg - 1` (`B_seg = ceil(num_chunks / 8)`), including rest bars.
    - Emits `<POS_0..191>` once per unique onset position within each bar, followed by chord notes
      in canonical `(slot, pitch)` order (`4` tokens/note).
    - Provides exact $O(M + B_{\text{seg}})$ vectorized encoding, decoding, and zero-loss binary transcoding
      directly from `Octuple-MMT` segment tokens.
    """

    def __init__(
        self,
        vocab_spec: REMIVocabSpec = DEFAULT_REMI_SPEC,
        grid_spec: SymbolicGridSpec = DEFAULT_GRID_SPEC,
        octuple_spec: OctupleVocabSpec = DEFAULT_OCTUPLE_SPEC,
    ) -> None:
        self.vocab_spec = vocab_spec
        self.grid_spec = grid_spec
        self.octuple_spec = octuple_spec
        self._dur_lut = self.grid_spec.build_dur_lut()
        self._octuple_offsets_np = np.asarray(self.octuple_spec.attr_offsets, dtype=np.int32)
        self._octuple_sizes_np = np.asarray(self.octuple_spec.attr_sizes, dtype=np.int32)

    @property
    def tokenizer_key(self) -> str:
        return "multitrack_remi"

    @property
    def model_mode(self) -> str:
        return "remi"

    @property
    def unit_width(self) -> int:
        return 1

    @property
    def token_offset(self) -> int:
        return self.vocab_spec.token_offset

    @property
    def vocab_size(self) -> int:
        return self.vocab_spec.vocab_size

    @property
    def max_segment_bars(self) -> int:
        return int(self.vocab_spec.max_bars)

    def compute_segment_bars(self, num_chunks: int) -> int:
        c = int(num_chunks)
        if c <= 0:
            return 0
        chunks_per_bar = STEPS_PER_BAR // CHUNK_STEPS
        return min(self.max_segment_bars, (c + chunks_per_bar - 1) // chunks_per_bar)

    def _scatter_remi_body(
        self,
        bars: np.ndarray,
        poss: np.ndarray,
        slots: np.ndarray,
        pitches: np.ndarray,
        dur_bins: np.ndarray,
        vel_bins: np.ndarray,
        num_chunks: int,
        out_dtype: np.dtype,
    ) -> np.ndarray:
        """
        Pure $O(M + B_{\text{seg}})$ vectorized scatter into 1D `Multitrack-REMI` body tokens
        with zero Python note loops.
        """
        b_seg = self.compute_segment_bars(num_chunks)
        m = int(len(bars))
        if m == 0:
            return np.full(b_seg, self.vocab_spec.bar_token_id, dtype=out_dtype)

        pos_sz = self.vocab_spec.pos_size
        slot_sz = self.vocab_spec.slot_size
        pitch_sz = self.vocab_spec.pitch_size

        # Defensive canonical sort check: O(M) fast path when already sorted, O(M log M) fallback otherwise
        if m > 1:
            coord = (
                ((bars.astype(np.int64) * pos_sz + poss.astype(np.int64)) * slot_sz + slots.astype(np.int64))
                * pitch_sz
                + pitches.astype(np.int64)
            )
            if np.any(coord[1:] < coord[:-1]):
                order = np.argsort(coord, kind="mergesort")
                bars = bars[order]
                poss = poss[order]
                slots = slots[order]
                pitches = pitches[order]
                dur_bins = dur_bins[order]
                vel_bins = vel_bins[order]

        b_seg = max(b_seg, int(bars[-1]) + 1)

        is_new_pos = np.empty(m, dtype=bool)
        is_new_pos[0] = True
        if m > 1:
            is_new_pos[1:] = (bars[1:] != bars[:-1]) | (poss[1:] != poss[:-1])

        pos_cum = np.cumsum(is_new_pos, dtype=np.int64)
        num_pos = int(pos_cum[-1])
        l_body = b_seg + num_pos + 4 * m

        out = np.full(l_body, self.vocab_spec.bar_token_id, dtype=out_dtype)
        note_out_idx = (bars.astype(np.int64) + 1) + pos_cum + 4 * np.arange(m, dtype=np.int64)

        out[note_out_idx[is_new_pos] - 1] = self.vocab_spec.pos_offset + poss[is_new_pos]
        out[note_out_idx] = self.vocab_spec.slot_offset + slots
        out[note_out_idx + 1] = self.vocab_spec.pitch_offset + pitches
        out[note_out_idx + 2] = self.vocab_spec.dur_offset + dur_bins
        out[note_out_idx + 3] = self.vocab_spec.vel_offset + vel_bins
        return out

    def encode_canonical_segment_notes(
        self,
        canonical_notes: np.ndarray,
        num_chunks: int = 128,
    ) -> np.ndarray:
        """
        Encode canonical segment notes `int32 [M, 6]` `(bar, pos, slot, pitch, dur_steps, velocity)`
        into flat `uint16 [L_body]` `Multitrack-REMI` body token IDs.
        """
        if len(canonical_notes) == 0:
            return self._scatter_remi_body(
                bars=np.zeros((0,), dtype=np.int32),
                poss=np.zeros((0,), dtype=np.int32),
                slots=np.zeros((0,), dtype=np.int32),
                pitches=np.zeros((0,), dtype=np.int32),
                dur_bins=np.zeros((0,), dtype=np.int32),
                vel_bins=np.zeros((0,), dtype=np.int32),
                num_chunks=num_chunks,
                out_dtype=np.dtype(np.uint16),
            )

        max_bar_idx = self.max_segment_bars - 1
        bars = np.clip(canonical_notes[:, 0], 0, max_bar_idx)
        poss = np.clip(canonical_notes[:, 1], 0, self.vocab_spec.pos_size - 1)
        slots = np.clip(canonical_notes[:, 2], 0, self.vocab_spec.slot_size - 1)
        pitches = np.clip(canonical_notes[:, 3], 0, self.vocab_spec.pitch_size - 1)
        dur_bins = np.clip(
            self.grid_spec.quantize_dur_np(canonical_notes[:, 4]), 0, self.vocab_spec.dur_size - 1
        )
        vel_bins = np.clip(
            self.grid_spec.quantize_vel_np(canonical_notes[:, 5]), 0, self.vocab_spec.vel_size - 1
        )
        return self._scatter_remi_body(
            bars=bars,
            poss=poss,
            slots=slots,
            pitches=pitches,
            dur_bins=dur_bins,
            vel_bins=vel_bins,
            num_chunks=num_chunks,
            out_dtype=np.dtype(np.uint16),
        )

    def transcode_octuple_body_to_remi(
        self,
        octuple_flat_tokens: np.ndarray,
        num_chunks: int = 128,
        out_dtype: Union[np.dtype, type] = np.int64,
    ) -> np.ndarray:
        """
        Losslessly transcode `Octuple-MMT` flat body tokens (`[6 * M]`) + `num_chunks` into
        1D `Multitrack-REMI` body tokens (`[L_body]`) in $O(M + B_{\text{seg}})$ vectorized NumPy.
        """
        target_dtype = np.dtype(out_dtype)
        flat = np.asarray(octuple_flat_tokens).reshape(-1)
        tuple_w = self.octuple_spec.tuple_size
        if flat.size == 0:
            return np.full(self.compute_segment_bars(num_chunks), self.vocab_spec.bar_token_id, dtype=target_dtype)
        if flat.size % tuple_w != 0:
            raise ValueError(
                f"Octuple body token count ({flat.size}) must be divisible by tuple_size ({tuple_w})."
            )

        rows = flat.astype(np.int32, copy=False).reshape(-1, tuple_w)
        local_vals = rows - self._octuple_offsets_np[None, :]
        bars = np.clip(local_vals[:, 0], 0, self.max_segment_bars - 1)
        poss = np.clip(local_vals[:, 1], 0, self.vocab_spec.pos_size - 1)
        slots = np.clip(local_vals[:, 2], 0, self.vocab_spec.slot_size - 1)
        pitches = np.clip(local_vals[:, 3], 0, self.vocab_spec.pitch_size - 1)
        dur_bins = np.clip(local_vals[:, 4], 0, self.vocab_spec.dur_size - 1)
        vel_bins = np.clip(local_vals[:, 5], 0, self.vocab_spec.vel_size - 1)
        return self._scatter_remi_body(
            bars=bars,
            poss=poss,
            slots=slots,
            pitches=pitches,
            dur_bins=dur_bins,
            vel_bins=vel_bins,
            num_chunks=num_chunks,
            out_dtype=target_dtype,
        )

    def build_full_sequence(
        self,
        body_tokens: np.ndarray,
        slot_codes: Optional[Sequence[int]] = None,
        add_bos: bool = True,
        add_inst: bool = True,
        add_eos: bool = True,
    ) -> np.ndarray:
        """Build a full 1D `int64` REMI sequence (`<BOS> + <INST_0..3> + body + <EOS>`)."""
        parts: List[np.ndarray] = []
        if add_bos:
            parts.append(np.asarray([BOS_TOKEN_ID], dtype=np.int64))
        if add_inst:
            codes = (
                list(slot_codes)
                if slot_codes is not None
                else [SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
            )
            inst_ids = np.asarray([slot_code_to_token_id(int(c)) for c in codes[:NUM_SLOT_TOKENS]], dtype=np.int64)
            parts.append(inst_ids)
        if body_tokens.size > 0:
            parts.append(np.asarray(body_tokens, dtype=np.int64).reshape(-1))
        if add_eos:
            parts.append(np.asarray([EOS_TOKEN_ID], dtype=np.int64))
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    def encode_notes_window_prompt(
        self,
        notes_arr: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        slot_codes: Sequence[int],
        append_eos: bool = False,
    ) -> Tensor:
        """
        Encode global `notes_arr` (`int32 [M, 5]` `(slot, start, end, pitch, velocity)`) over window
        `[chunk_start, chunk_end)` (`<= max_segment_bars * 8` chunks) into a 2D REMI prompt tensor `[1, S]`.
        """
        c0 = int(chunk_start)
        c1 = int(chunk_end)
        body_u16 = (
            self.encode_song_segment(notes_arr, c0, c1)
            if c1 > c0
            else np.zeros((0,), dtype=np.uint16)
        )
        seq_1d = self.build_full_sequence(
            body_u16,
            slot_codes=slot_codes,
            add_bos=True,
            add_inst=True,
            add_eos=append_eos,
        )
        return torch.from_numpy(seq_1d.reshape(1, -1))

    def encode_midi_prompt(
        self,
        midi_path: Union[str, Path],
        prompt_chunks: int = 32,
        append_eos: bool = False,
    ) -> Tuple[Tensor, np.ndarray]:
        """
        Encode the first `prompt_chunks` chunks (e.g. `32 chunks = 4 bars`) of a MIDI file into
        a REMI prompt tensor `[1, S]` ready for `REMITransformer.generate(prompt=...)`.
        """
        num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(midi_path)
        eff_chunks = max(0, min(int(num_chunks), int(prompt_chunks)))
        filtered_notes = self.filter_notes_by_slots_and_horizon(
            notes_arr, slot_codes=slot_codes.tolist(), num_chunks=eff_chunks
        )
        seq_2d = self.encode_notes_window_prompt(
            filtered_notes, 0, eff_chunks, slot_codes=slot_codes.tolist(), append_eos=append_eos
        )
        return seq_2d, slot_codes

    def encode_chunks_f32_prompt(
        self,
        chunks_f32: np.ndarray,
        slot_codes: Sequence[int],
        append_eos: bool = False,
    ) -> Tensor:
        """
        Encode an in-memory float32 piano-roll `[N_chunks, 8, 128, 24]` (or `[N_chunks, 8, 24, 128]`)
        into a REMI prompt tensor `[1, S]` without temporary files.
        """
        n_chunks = max(0, int(chunks_f32.shape[0]))
        notes_arr = self.extract_notes_from_chunks_f32(chunks_f32, slot_codes=slot_codes)
        return self.encode_notes_window_prompt(
            notes_arr, 0, n_chunks, slot_codes=slot_codes, append_eos=append_eos
        )

    def decode_tokens_to_notes(
        self,
        seq: Union[np.ndarray, Tensor, Sequence[int]],
        num_chunks: Optional[int] = None,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(seq, Tensor):
            arr = seq.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            arr = np.asarray(seq, dtype=np.int64)

        flat = arr.reshape(-1)
        if flat.size == 0:
            eff_slots_empty = np.asarray(slot_codes, dtype=np.int16) if slot_codes is not None else None
            return np.zeros((0, 5), dtype=np.int32), eff_slots_empty

        # 1. Truncate at first <EOS> (if present) prior to prefix/body parsing
        eos_positions = np.flatnonzero(flat == EOS_TOKEN_ID)
        if len(eos_positions) > 0:
            flat = flat[: int(eos_positions[0])]

        # 2. Extract 4-slot instrument codes from prefix tokens if present (override if explicitly supplied)
        inst_mask = (flat >= INST_PROGRAM_OFFSET) & (flat <= INST_EMPTY_TOKEN_ID)
        extracted_slots: Optional[np.ndarray] = None
        if np.count_nonzero(inst_mask) >= NUM_SLOT_TOKENS:
            inst_ids = flat[inst_mask][:NUM_SLOT_TOKENS]
            extracted_slots = np.asarray([token_id_to_slot_code(int(tid)) for tid in inst_ids], dtype=np.int16)
        eff_slots: Optional[np.ndarray] = (
            np.asarray(slot_codes, dtype=np.int16) if slot_codes is not None else extracted_slots
        )

        n_tok = int(flat.size)
        if n_tok < 4:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        bar_id = self.vocab_spec.bar_token_id
        pos_lo = self.vocab_spec.pos_offset
        slot_lo = self.vocab_spec.slot_offset
        pitch_lo = self.vocab_spec.pitch_offset
        dur_lo = self.vocab_spec.dur_offset
        vel_lo = self.vocab_spec.vel_offset
        vocab_hi = self.vocab_spec.vocab_size

        # 3. Vectorized O(L) propagation of current (bar, pos) context at every token index
        is_bar = flat == bar_id
        is_pos = (flat >= pos_lo) & (flat < slot_lo)
        bar_at_t = np.cumsum(is_bar, dtype=np.int32) - 1

        idx_grid = np.arange(n_tok, dtype=np.int64)
        last_evt_idx = np.maximum.accumulate(np.where(is_bar | is_pos, idx_grid, -1))
        has_evt = last_evt_idx >= 0
        safe_evt_idx = np.maximum(0, last_evt_idx)
        evt_tok = flat[safe_evt_idx]
        pos_at_t = np.where(has_evt & (evt_tok >= pos_lo) & (evt_tok < slot_lo), evt_tok - pos_lo, -1).astype(
            np.int32, copy=False
        )

        # 4. Non-overlapping 4-tuple note detection: (<SLOT>, <PITCH>, <DUR>, <VEL>)
        t0 = flat[:-3]
        t1 = flat[1:-2]
        t2 = flat[2:-1]
        t3 = flat[3:]
        valid_note = (
            (bar_at_t[:-3] >= 0)
            & (bar_at_t[:-3] < self.max_segment_bars)
            & (pos_at_t[:-3] >= 0)
            & (t0 >= slot_lo)
            & (t0 < pitch_lo)
            & (t1 >= pitch_lo)
            & (t1 < dur_lo)
            & (t2 >= dur_lo)
            & (t2 < vel_lo)
            & (t3 >= vel_lo)
            & (t3 < vocab_hi)
        )
        note_idx = np.flatnonzero(valid_note)
        if len(note_idx) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        bars = bar_at_t[note_idx]
        poss = pos_at_t[note_idx]
        slots = (flat[note_idx] - slot_lo).astype(np.int32, copy=False)
        pitches = (flat[note_idx + 1] - pitch_lo).astype(np.int32, copy=False)
        dur_bins = (flat[note_idx + 2] - dur_lo).astype(np.int32, copy=False)
        vel_bins = (flat[note_idx + 3] - vel_lo).astype(np.int32, copy=False)

        durs = self.grid_spec.dequantize_dur_np(dur_bins, lut=self._dur_lut)
        vels = self.grid_spec.dequantize_vel_np(vel_bins)

        if eff_slots is not None and len(eff_slots) >= NUM_SLOT_TOKENS:
            active_slot_lut = eff_slots[:NUM_SLOT_TOKENS] != SLOT_CODE_EMPTY
            slot_in_range = (slots >= 0) & (slots < NUM_SLOT_TOKENS)
            keep_active = slot_in_range & active_slot_lut[np.clip(slots, 0, NUM_SLOT_TOKENS - 1)]
            bars = bars[keep_active]
            poss = poss[keep_active]
            slots = slots[keep_active]
            pitches = pitches[keep_active]
            durs = durs[keep_active]
            vels = vels[keep_active]

        if len(slots) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        starts = bars * STEPS_PER_BAR + poss
        ends = starts + durs

        if num_chunks is not None and int(num_chunks) > 0:
            max_step = int(num_chunks) * CHUNK_STEPS
            in_bounds = starts < max_step
            slots = slots[in_bounds]
            starts = starts[in_bounds]
            ends = np.minimum(max_step, ends[in_bounds])
            pitches = pitches[in_bounds]
            vels = vels[in_bounds]

        if len(starts) == 0:
            return np.zeros((0, 5), dtype=np.int32), eff_slots

        ends = np.maximum(ends, starts + 1)
        notes_arr = np.stack([slots, starts, ends, pitches, vels], axis=1).astype(np.int32, copy=False)
        return notes_arr, eff_slots


SYMBOLIC_TOKENIZER_REGISTRY: Dict[str, Callable[[], AbstractSymbolicSegmentTokenizer]] = {
    "octuple": OctupleMMTTokenizer,
    "octuple_mmt": OctupleMMTTokenizer,
    "remi": MultitrackREMITokenizer,
    "multitrack_remi": MultitrackREMITokenizer,
}


def register_symbolic_tokenizer(
    key: str,
    builder: Callable[[], AbstractSymbolicSegmentTokenizer],
) -> None:
    SYMBOLIC_TOKENIZER_REGISTRY[str(key).strip().lower()] = builder


def build_symbolic_tokenizer(key: str) -> AbstractSymbolicSegmentTokenizer:
    norm_key = str(key).strip().lower()
    if norm_key not in SYMBOLIC_TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown symbolic tokenizer '{key}'. Available: {sorted(SYMBOLIC_TOKENIZER_REGISTRY.keys())}"
        )
    return SYMBOLIC_TOKENIZER_REGISTRY[norm_key]()
