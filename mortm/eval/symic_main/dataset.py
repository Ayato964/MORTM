from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mortm.utils.convert_pianoroll import (
    CHUNKS_PER_BAR,
    CHUNK_STEPS,
    STEPS_PER_BAR,
    extract_quantized_notes_and_slots,
)
from mortm.utils.symbolic_tokenizers import extract_segment_canonical_notes

from .domain import DEFAULT_DENSITY_TIERS, DensityTierSpec, TestSegmentItem


class GmdSegmentRepository:
    """
    Streaming repository over `data/gmd/eval.json` yielding canonical 16-bar `TestSegmentItem`
    instances with O(1) memory overhead.
    """

    def __init__(self, eval_json_path: str | Path, segment_chunks: int = 128) -> None:
        self._eval_json_path = Path(eval_json_path)
        if not self._eval_json_path.is_file():
            raise FileNotFoundError(f"Evaluation manifest not found: {self._eval_json_path}")
        if segment_chunks <= 0 or segment_chunks % CHUNKS_PER_BAR != 0:
            raise ValueError(
                f"segment_chunks must be a positive multiple of {CHUNKS_PER_BAR}, got {segment_chunks}"
            )
        self._segment_chunks = int(segment_chunks)
        self._segment_bars = self._segment_chunks // CHUNKS_PER_BAR

    def _iter_manifest_paths(self) -> Iterator[Tuple[int, str]]:
        with self._eval_json_path.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
        if not isinstance(entries, list):
            raise ValueError(f"Expected JSON list in {self._eval_json_path}, got {type(entries).__name__}")
        for idx, raw in enumerate(entries):
            path_str = raw if isinstance(raw, str) else str(raw.get("path", ""))
            if path_str:
                yield idx, path_str

    def iter_16bar_segments(
        self,
        max_songs: Optional[int] = None,
        one_per_song: bool = False,
    ) -> Iterator[TestSegmentItem]:
        """
        Lazily extract non-overlapping 16-bar (`128`-chunk) segments from unseen GMD MIDI files.
        """
        songs_yielded = 0
        seg_counter = 0
        for song_idx, midi_path in self._iter_manifest_paths():
            if max_songs is not None and songs_yielded >= int(max_songs):
                break
            path_obj = Path(midi_path)
            if not path_obj.is_file():
                continue
            try:
                num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(path_obj)
            except Exception:
                continue

            if num_chunks < self._segment_chunks or len(notes_arr) == 0:
                continue

            produced_from_song = False
            slot_tuple = tuple(int(c) for c in slot_codes.tolist())
            for c0 in range(0, num_chunks - self._segment_chunks + 1, self._segment_chunks):
                c1 = c0 + self._segment_chunks
                canonical = extract_segment_canonical_notes(
                    notes_arr,
                    chunk_start=c0,
                    chunk_end=c1,
                    max_bars=self._segment_bars,
                )
                if len(canonical) == 0:
                    continue
                item = TestSegmentItem(
                    item_id=f"seg_{seg_counter:04d}_s{song_idx:04d}_c{c0:04d}",
                    song_index=song_idx,
                    midi_path=str(path_obj),
                    chunk_start=c0,
                    chunk_end=c1,
                    slot_codes=slot_tuple,
                    canonical_notes=canonical,
                )
                seg_counter += 1
                produced_from_song = True
                yield item
                if one_per_song:
                    break

            if produced_from_song:
                songs_yielded += 1

    def select_ensemble_continuation_set(
        self,
        limit: int = 200,
        prompt_chunks: int = 32,
        min_active_tracks: int = 3,
        min_prompt_notes: int = 8,
        min_cont_melodic_notes: int = 8,
    ) -> List[TestSegmentItem]:
        """
        Deterministically select `limit` unseen 3-to-4 track ensemble 16-bar segments
        (strictly at most 1 segment per distinct GMD test song) for §5.2 4->16 bar continuation.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        prompt_step_limit = int(prompt_chunks) * CHUNK_STEPS
        prompt_bar_limit = int(prompt_chunks) // CHUNKS_PER_BAR
        selected: List[TestSegmentItem] = []

        for song_idx, midi_path in self._iter_manifest_paths():
            if len(selected) >= limit:
                break
            path_obj = Path(midi_path)
            if not path_obj.is_file():
                continue
            try:
                num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(path_obj)
            except Exception:
                continue

            if num_chunks < self._segment_chunks or len(notes_arr) == 0:
                continue

            slot_tuple = tuple(int(c) for c in slot_codes.tolist())
            for c0 in range(0, num_chunks - self._segment_chunks + 1, self._segment_chunks):
                c1 = c0 + self._segment_chunks
                canonical = extract_segment_canonical_notes(
                    notes_arr,
                    chunk_start=c0,
                    chunk_end=c1,
                    max_bars=self._segment_bars,
                )
                if len(canonical) == 0:
                    continue

                cand = TestSegmentItem(
                    item_id=f"song_{len(selected):04d}_gmd{song_idx:04d}",
                    song_index=song_idx,
                    midi_path=str(path_obj),
                    chunk_start=c0,
                    chunk_end=c1,
                    slot_codes=slot_tuple,
                    canonical_notes=canonical,
                )
                if (
                    cand.active_slot_count < min_active_tracks
                    or cand.sounding_track_count < min_active_tracks
                ):
                    continue

                starts = canonical[:, 0] * STEPS_PER_BAR + canonical[:, 1]
                ends = starts + canonical[:, 4]
                in_prompt = (starts < prompt_step_limit) & (ends > 0)
                if int(np.sum(in_prompt)) < min_prompt_notes:
                    continue

                in_cont_onset = starts >= prompt_step_limit
                cont_melodic = in_cont_onset & (canonical[:, 2] > 0)
                if int(np.sum(cont_melodic)) < min_cont_melodic_notes:
                    continue

                cont_bars = np.unique(canonical[in_cont_onset, 0])
                if len(cont_bars[cont_bars >= prompt_bar_limit]) < 2:
                    continue

                selected.append(cand)
                break

        if len(selected) < limit:
            raise RuntimeError(
                f"Requested {limit} valid 3-4 track 16-bar songs, but only found {len(selected)}."
            )
        return selected

    def collect_context_and_stratified_profile_sets(
        self,
        max_scan_songs: int = 200,
        per_tier: int = 8,
        tiers: Sequence[DensityTierSpec] = DEFAULT_DENSITY_TIERS,
    ) -> Tuple[List[TestSegmentItem], Dict[str, List[TestSegmentItem]]]:
        """
        Scan non-overlapping 16-bar segments across the first `max_scan_songs` unseen songs
        (1..4 tracks) for §5.1 context scaling statistics, and stratify up to `per_tier`
        representative segments per `Notes/Bar` density tier for hardware latency profiling.
        """
        all_segments: List[TestSegmentItem] = []
        buckets: Dict[str, List[TestSegmentItem]] = {t.tier_id: [] for t in tiers}

        for seg in self.iter_16bar_segments(max_songs=max_scan_songs, one_per_song=False):
            all_segments.append(seg)
            npb = seg.notes_per_bar
            for tier in tiers:
                if tier.contains(npb) and len(buckets[tier.tier_id]) < per_tier:
                    buckets[tier.tier_id].append(seg)
                    break

        # If extreme high-density tiers (< per_tier in first 200 songs) still have room, top up lazily
        if any(len(buckets[t.tier_id]) < per_tier for t in tiers):
            seen_keys = {(s.song_index, s.chunk_start) for s in all_segments}
            for seg in self.iter_16bar_segments(max_songs=1500, one_per_song=False):
                if (seg.song_index, seg.chunk_start) in seen_keys:
                    continue
                npb = seg.notes_per_bar
                for tier in tiers:
                    if tier.contains(npb) and len(buckets[tier.tier_id]) < per_tier:
                        buckets[tier.tier_id].append(seg)
                        break
                if all(len(buckets[t.tier_id]) >= per_tier for t in tiers):
                    break

        return all_segments, buckets
