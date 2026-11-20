from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import muspy
import numpy as np

from mortm.utils.convert_pianoroll import (
    BEATS_PER_BAR,
    CHUNKS_PER_BAR,
    CHUNK_STEPS,
    STEPS_PER_BAR,
    _notes_to_midi,
    extract_quantized_notes_and_slots,
    slot_codes_to_roll_meta,
)

from .domain import MusPySongMetrics, TestSegmentItem


STEPS_PER_BEAT: int = STEPS_PER_BAR // BEATS_PER_BAR  # 48


@dataclass(frozen=True)
class PairedMetricSummary:
    """Immutable summary of corpus mean, std, and paired per-song absolute error vs Ground Truth."""

    valid_pairs: int
    total_songs: int
    corpus_mean: float
    corpus_std: float
    gt_mean: float
    paired_mae: float
    paired_mae_ci95_low: float
    paired_mae_ci95_high: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "valid_pairs": self.valid_pairs,
            "total_songs": self.total_songs,
            "corpus_mean": round(self.corpus_mean, 4),
            "corpus_std": round(self.corpus_std, 4),
            "gt_mean": round(self.gt_mean, 4),
            "paired_mae": round(self.paired_mae, 4),
            "paired_mae_ci95_low": round(self.paired_mae_ci95_low, 4),
            "paired_mae_ci95_high": round(self.paired_mae_ci95_high, 4),
        }


class StandardizedMidiSliceService:
    """
    Single-Source-of-Truth MIDI export and continuation-window slicing service (§3 & §5.2.B).
    """

    @staticmethod
    def export_ground_truth_16bar(segment: TestSegmentItem, output_16bar_midi: Path) -> Path:
        output_16bar_midi.parent.mkdir(parents=True, exist_ok=True)
        notes_arr = segment.to_segment_5col_notes()
        meta = slot_codes_to_roll_meta(segment.slot_codes)
        tuples = [
            (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]))
            for r in notes_arr
            if int(r[2]) > int(r[1])
        ]
        _notes_to_midi(sorted(tuples), meta, output_16bar_midi)
        return output_16bar_midi

    @staticmethod
    def slice_16bar_midi_to_continuation_midi(
        midi_16bar_path: Path,
        output_cont_midi: Path,
        slot_codes: Sequence[int],
        prompt_chunks: int = 32,
        total_chunks: int = 128,
    ) -> Tuple[int, int]:
        """
        Extract notes attacked within `[prompt_chunks * 24, total_chunks * 24)` from any
        standardized 16-bar `.mid` file, shift timestamps so bar 5 begins at `step = 0`,
        and save the 12-bar continuation `.mid`. Returns `(total_notes, melodic_notes)`.
        """
        output_cont_midi.parent.mkdir(parents=True, exist_ok=True)
        _, notes_arr, _ = extract_quantized_notes_and_slots(midi_16bar_path)
        start_step = int(prompt_chunks) * CHUNK_STEPS
        end_step = int(total_chunks) * CHUNK_STEPS
        horizon_steps = max(1, end_step - start_step)

        cont_tuples: List[Tuple[int, int, int, int, int]] = []
        melodic_count = 0
        if len(notes_arr) > 0:
            mask = (notes_arr[:, 1] >= start_step) & (notes_arr[:, 1] < end_step)
            for row in notes_arr[mask]:
                slot = int(row[0])
                s_loc = int(row[1]) - start_step
                e_loc = min(horizon_steps, int(row[2]) - start_step)
                pitch = int(row[3])
                vel = int(row[4])
                if e_loc > s_loc:
                    cont_tuples.append((slot, s_loc, e_loc, pitch, vel))
                    if slot > 0:
                        melodic_count += 1

        meta = slot_codes_to_roll_meta(slot_codes)
        _notes_to_midi(sorted(cont_tuples), meta, output_cont_midi)
        return len(cont_tuples), melodic_count


class MusPyContinuationEvaluator:
    """
    Computes official `MusPy` objective metrics (`Pitch Class Entropy`, `Scale Consistency`,
    `Groove Consistency`) on 12-bar continuation MIDIs and paired per-song errors vs Ground Truth (§5.2.C).
    """

    def __init__(self, continuation_bars: int = 12, bootstrap_samples: int = 2000, seed: int = 42) -> None:
        self._continuation_bars = max(2, int(continuation_bars))
        self._max_tick = self._continuation_bars * STEPS_PER_BAR - 1
        self._bootstrap_samples = max(100, int(bootstrap_samples))
        self._seed = int(seed)

    def evaluate_single_continuation_midi(
        self,
        cont_midi_path: Path,
        note_count: int,
        melodic_note_count: int,
    ) -> MusPySongMetrics:
        if note_count == 0 or not cont_midi_path.is_file():
            return MusPySongMetrics(
                pitch_class_entropy=math.nan,
                scale_consistency=math.nan,
                groove_consistency=math.nan,
                note_count=note_count,
                melodic_note_count=melodic_note_count,
            )

        music = muspy.read_midi(str(cont_midi_path))
        music.adjust_resolution(target=STEPS_PER_BEAT)

        # Prevent a note ending on the exact final barline (`time + duration == 12 * 192`)
        # from creating an artificial empty 13th measure inside `muspy.groove_consistency`.
        for track in music.tracks:
            for note in track.notes:
                if note.time > self._max_tick:
                    note.time = self._max_tick
                if note.time + note.duration > self._max_tick:
                    note.duration = max(1, self._max_tick - note.time)

        has_any_notes = any(len(t.notes) > 0 for t in music.tracks)
        if not has_any_notes:
            return MusPySongMetrics(
                pitch_class_entropy=math.nan,
                scale_consistency=math.nan,
                groove_consistency=math.nan,
                note_count=0,
                melodic_note_count=0,
            )

        pce = float(muspy.pitch_class_entropy(music))
        sc = float(muspy.scale_consistency(music))
        gc = float(muspy.groove_consistency(music, measure_resolution=STEPS_PER_BAR))
        return MusPySongMetrics(
            pitch_class_entropy=pce,
            scale_consistency=sc,
            groove_consistency=gc,
            note_count=note_count,
            melodic_note_count=melodic_note_count,
        )

    def summarize_paired_errors(
        self,
        gt_metrics: Sequence[MusPySongMetrics],
        gen_metrics: Sequence[MusPySongMetrics],
    ) -> Dict[str, PairedMetricSummary]:
        if len(gt_metrics) != len(gen_metrics):
            raise ValueError(
                f"GT ({len(gt_metrics)}) and Gen ({len(gen_metrics)}) metric counts must match."
            )

        rng = np.random.default_rng(self._seed)
        total = len(gt_metrics)
        results: Dict[str, PairedMetricSummary] = {}

        for attr in ("pitch_class_entropy", "scale_consistency", "groove_consistency"):
            gt_vals: List[float] = []
            gen_vals: List[float] = []
            abs_errs: List[float] = []
            for gt_m, gen_m in zip(gt_metrics, gen_metrics):
                gv = float(getattr(gt_m, attr))
                mv = float(getattr(gen_m, attr))
                if math.isfinite(gv) and math.isfinite(mv):
                    gt_vals.append(gv)
                    gen_vals.append(mv)
                    abs_errs.append(abs(mv - gv))

            if not abs_errs:
                results[attr] = PairedMetricSummary(
                    valid_pairs=0,
                    total_songs=total,
                    corpus_mean=math.nan,
                    corpus_std=math.nan,
                    gt_mean=math.nan,
                    paired_mae=math.nan,
                    paired_mae_ci95_low=math.nan,
                    paired_mae_ci95_high=math.nan,
                )
                continue

            err_arr = np.asarray(abs_errs, dtype=np.float64)
            gen_arr = np.asarray(gen_vals, dtype=np.float64)
            gt_arr = np.asarray(gt_vals, dtype=np.float64)
            paired_mae = float(np.mean(err_arr))

            if len(err_arr) >= 2:
                boot_idx = rng.integers(0, len(err_arr), size=(self._bootstrap_samples, len(err_arr)))
                boot_means = np.mean(err_arr[boot_idx], axis=1)
                ci_low, ci_high = (
                    float(np.percentile(boot_means, 2.5)),
                    float(np.percentile(boot_means, 97.5)),
                )
            else:
                ci_low = paired_mae
                ci_high = paired_mae

            results[attr] = PairedMetricSummary(
                valid_pairs=len(abs_errs),
                total_songs=total,
                corpus_mean=float(np.mean(gen_arr)),
                corpus_std=float(np.std(gen_arr)),
                gt_mean=float(np.mean(gt_arr)),
                paired_mae=paired_mae,
                paired_mae_ci95_low=ci_low,
                paired_mae_ci95_high=ci_high,
            )

        return results


class Clamp2FmdService:
    """
    Reusable CLaMP 2 Fréchet Music Distance evaluator (§5.2.C).
    Caches reference Gaussian statistics `(mu_ref, sigma_ref)` so multiple candidate
    models can be scored against Ground Truth without redundant feature extraction.
    """

    def __init__(self, verbose: bool = False) -> None:
        from frechet_music_distance import FrechetMusicDistance

        self._fmd = FrechetMusicDistance(
            feature_extractor="clamp2",
            gaussian_estimator="mle",
            verbose=verbose,
        )
        self._cached_ref_dir: str | None = None
        self._cached_ref_stats: Tuple[np.ndarray, np.ndarray] | None = None

    def score_directory(self, reference_midi_dir: Path, generated_midi_dir: Path) -> float:
        ref_str = str(reference_midi_dir.resolve())
        gen_str = str(generated_midi_dir.resolve())
        if hasattr(self._fmd, "_get_dataset_stats") and hasattr(self._fmd, "_compute_fmd"):
            if self._cached_ref_dir != ref_str or self._cached_ref_stats is None:
                self._cached_ref_stats = self._fmd._get_dataset_stats(ref_str)
                self._cached_ref_dir = ref_str
            mean_ref, cov_ref = self._cached_ref_stats
            mean_gen, cov_gen = self._fmd._get_dataset_stats(gen_str)
            return float(self._fmd._compute_fmd(mean_ref, mean_gen, cov_ref, cov_gen))
        return float(self._fmd.score(ref_str, gen_str))
