from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from mortm.utils.mortm5_infer import select_device

from .dataset import GmdSegmentRepository
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
from .evaluators import (
    Clamp2FmdService,
    MusPyContinuationEvaluator,
    PairedMetricSummary,
    StandardizedMidiSliceService,
)
from .strategies import (
    ParadigmEvaluationStrategy,
    ParadigmStrategyRegistry,
    create_default_strategy_registry,
    synchronize_device,
)


def _release_accelerator_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _summarize_step_distribution(
    steps_list: Sequence[int],
    kv_bytes_per_step: int,
) -> Dict[str, float | int]:
    if not steps_list:
        return {}
    arr = np.asarray(steps_list, dtype=np.float64)
    mib_scale = float(kv_bytes_per_step) / float(1024 * 1024)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    s_max = float(np.max(arr))
    s_mean = float(np.mean(arr))
    return {
        "n_segments": len(steps_list),
        "mean_steps": round(s_mean, 2),
        "p50_steps": int(round(p50)),
        "p90_steps": int(round(p90)),
        "p95_steps": int(round(p95)),
        "p99_steps": int(round(p99)),
        "max_steps": int(round(s_max)),
        "mean_kv_mib": round((s_mean - 1.0) * mib_scale, 3),
        "p50_kv_mib": round((p50 - 1.0) * mib_scale, 3),
        "p90_kv_mib": round((p90 - 1.0) * mib_scale, 3),
        "p95_kv_mib": round((p95 - 1.0) * mib_scale, 3),
        "p99_kv_mib": round((p99 - 1.0) * mib_scale, 3),
        "max_kv_mib": round((s_max - 1.0) * mib_scale, 3),
    }


def _summarize_telemetry_batch(records: Sequence[RolloutTelemetry]) -> Dict[str, float | int]:
    if not records:
        return {}
    ms_bar = np.asarray([r.ms_per_bar for r in records], dtype=np.float64)
    dec_ms_bar = np.asarray([r.decode_ms_per_bar for r in records], dtype=np.float64)
    bps = np.asarray([r.bars_per_sec for r in records], dtype=np.float64)
    gen_steps = np.asarray([r.generated_steps for r in records], dtype=np.float64)
    tot_steps = np.asarray([r.total_steps for r in records], dtype=np.float64)
    kv_peak = np.asarray([r.peak_active_kv_mib for r in records], dtype=np.float64)

    bar_traj_summary: Dict[str, float] = {}
    for b_mark in (4, 8, 12, 16):
        vals = [
            mib
            for r in records
            for (b_i, mib) in r.kv_bar_trajectory_mib
            if b_i == b_mark
        ]
        if vals:
            bar_traj_summary[f"bar_{b_mark}_kv_mib"] = round(float(np.mean(vals)), 3)

    return {
        "count": len(records),
        "mean_generated_steps": round(float(np.mean(gen_steps)), 2),
        "mean_total_steps": round(float(np.mean(tot_steps)), 2),
        "p90_total_steps": int(round(float(np.percentile(tot_steps, 90)))),
        "max_total_steps": int(round(float(np.max(tot_steps)))),
        "mean_ms_per_bar": round(float(np.mean(ms_bar)), 2),
        "p50_ms_per_bar": round(float(np.percentile(ms_bar, 50)), 2),
        "p90_ms_per_bar": round(float(np.percentile(ms_bar, 90)), 2),
        "mean_decode_ms_per_bar": round(float(np.mean(dec_ms_bar)), 2),
        "mean_bars_per_sec": round(float(np.mean(bps)), 3),
        "mean_peak_kv_mib": round(float(np.mean(kv_peak)), 3),
        "p90_peak_kv_mib": round(float(np.percentile(kv_peak, 90)), 3),
        "max_peak_kv_mib": round(float(np.max(kv_peak)), 3),
        **bar_traj_summary,
    }


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Immutable top-level execution configuration for the Stage-3 Benchmark Runner."""

    eval_json: str = "data/gmd/eval.json"
    output_dir: str = "out/symic_main/benchmark_eval"
    num_continuation_songs: int = 200
    prompt_chunks: int = 32
    segment_chunks: int = 128
    context_scan_songs: int = 200
    latency_per_tier: int = 8
    seed: int = 42
    device: Optional[str] = None
    overwrite: bool = False
    run_eval1: bool = True
    run_eval2: bool = True


class SymicStage3BenchmarkRunner:
    """
    Coordinates §5.1 (Systems & Context Efficiency Scaling) and §5.2 (4->16 Bar Prompted
    Continuation: CLaMP 2 FMD + Paired MusPy PCE/SC/GC) across Stage-3 candidates.
    """

    def __init__(
        self,
        config: BenchmarkRunConfig,
        candidates: Sequence[ModelCandidateSpec] = DEFAULT_STAGE3_CANDIDATES,
        sampling: SamplingSpec = DEFAULT_SAMPLING_SPEC,
        tiers: Sequence[DensityTierSpec] = DEFAULT_DENSITY_TIERS,
        registry: Optional[ParadigmStrategyRegistry] = None,
    ) -> None:
        self._cfg = config
        self._candidates = tuple(candidates)
        self._sampling = sampling
        self._tiers = tuple(tiers)
        self._registry = registry or create_default_strategy_registry()
        self._device = select_device(config.device)
        self._out_root = Path(config.output_dir)
        self._repo = GmdSegmentRepository(
            eval_json_path=config.eval_json,
            segment_chunks=config.segment_chunks,
        )

    def run(self) -> Dict[str, object]:
        self._out_root.mkdir(parents=True, exist_ok=True)
        print(
            f"[Stage-3 Benchmark] Device={self._device} | Candidates={[c.key for c in self._candidates]} "
            f"| Songs={self._cfg.num_continuation_songs} | Sampling={self._sampling.to_dict()}",
            flush=True,
        )

        all_ctx_segs: List[TestSegmentItem] = []
        tier_buckets: Dict[str, List[TestSegmentItem]] = {}
        if self._cfg.run_eval1:
            print("[Eval-1] Scanning unseen GMD 16-bar segments & density tiers...", flush=True)
            all_ctx_segs, tier_buckets = self._repo.collect_context_and_stratified_profile_sets(
                max_scan_songs=self._cfg.context_scan_songs,
                per_tier=self._cfg.latency_per_tier,
                tiers=self._tiers,
            )
            tier_counts = {k: len(v) for k, v in tier_buckets.items()}
            print(
                f"[Eval-1] Collected {len(all_ctx_segs)} segments (1..4 trk) | Stratified tiers={tier_counts}",
                flush=True,
            )

        cont_segments: List[TestSegmentItem] = []
        gt_16bar_dir = self._out_root / "midis" / "ground_truth" / "16bar"
        gt_cont_dir = self._out_root / "midis" / "ground_truth" / "cont12bar"
        muspy_eval = MusPyContinuationEvaluator(
            continuation_bars=(self._cfg.segment_chunks - self._cfg.prompt_chunks) // 8,
            seed=self._cfg.seed,
        )
        gt_muspy_list: List[MusPySongMetrics] = []

        if self._cfg.run_eval2:
            print(
                f"[Eval-2] Selecting {self._cfg.num_continuation_songs} unseen 3-4 track ensemble songs...",
                flush=True,
            )
            cont_segments = self._repo.select_ensemble_continuation_set(
                limit=self._cfg.num_continuation_songs,
                prompt_chunks=self._cfg.prompt_chunks,
            )
            manifest_path = self._out_root / "eval2_manifest.json"
            manifest_path.write_text(
                json.dumps([s.metadata_dict() for s in cont_segments], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            for seg in cont_segments:
                gt_16_p = gt_16bar_dir / f"{seg.item_id}.mid"
                gt_c_p = gt_cont_dir / f"{seg.item_id}.mid"
                if self._cfg.overwrite or not gt_16_p.is_file():
                    StandardizedMidiSliceService.export_ground_truth_16bar(seg, gt_16_p)
                n_tot, n_mel = StandardizedMidiSliceService.slice_16bar_midi_to_continuation_midi(
                    midi_16bar_path=gt_16_p,
                    output_cont_midi=gt_c_p,
                    slot_codes=seg.slot_codes,
                    prompt_chunks=self._cfg.prompt_chunks,
                    total_chunks=self._cfg.segment_chunks,
                )
                gt_muspy_list.append(
                    muspy_eval.evaluate_single_continuation_midi(gt_c_p, n_tot, n_mel)
                )
            print(
                f"[Eval-2] Exported {len(cont_segments)} Ground-Truth 16-bar & 12-bar MIDIs.",
                flush=True,
            )

        eval1_results: Dict[str, object] = {}
        eval2_model_outputs: Dict[str, Dict[str, object]] = {}

        for cand in self._candidates:
            print(f"\n=== Loading Candidate: {cand.display_name} ({cand.key}) ===", flush=True)
            strategy = self._registry.create(cand, self._device)
            try:
                if self._cfg.run_eval1:
                    eval1_results[cand.key] = self._execute_eval1_for_model(
                        strategy=strategy,
                        all_ctx_segs=all_ctx_segs,
                        tier_buckets=tier_buckets,
                    )
                if self._cfg.run_eval2:
                    eval2_model_outputs[cand.key] = self._execute_eval2_generation_for_model(
                        strategy=strategy,
                        cont_segments=cont_segments,
                        muspy_eval=muspy_eval,
                        gt_muspy_list=gt_muspy_list,
                    )
            finally:
                del strategy
                _release_accelerator_memory(self._device)

        if self._cfg.run_eval2:
            print("\n[Eval-2] Computing CLaMP 2 Fréchet Music Distance (FMD)...", flush=True)
            fmd_service = Clamp2FmdService(verbose=False)
            for cand in self._candidates:
                gen_16_dir = self._out_root / "midis" / cand.key / "16bar"
                t_fmd = time.perf_counter()
                fmd_val = fmd_service.score_directory(gt_16bar_dir, gen_16_dir)
                dt_fmd = time.perf_counter() - t_fmd
                eval2_model_outputs[cand.key]["fmd_clamp2"] = round(fmd_val, 4)
                print(
                    f"  -> {cand.display_name}: FMD (CLaMP 2) = {fmd_val:.4f} ({dt_fmd:.1f}s)",
                    flush=True,
                )

        gt_summary = self._build_gt_muspy_summary(gt_muspy_list) if gt_muspy_list else {}
        final_payload: Dict[str, object] = {
            "device": str(self._device),
            "sampling_protocol": self._sampling.to_dict(),
            "num_continuation_songs": self._cfg.num_continuation_songs,
            "prompt_bars": self._cfg.prompt_chunks // 8,
            "total_bars": self._cfg.segment_chunks // 8,
            "candidates": [
                {
                    "key": c.key,
                    "display_name": c.display_name,
                    "paradigm": c.paradigm,
                    "config_path": c.config_path,
                    "checkpoint_path": c.checkpoint_path,
                }
                for c in self._candidates
            ],
            "evaluation_1_systems_efficiency": eval1_results,
            "evaluation_2_continuation": {
                "ground_truth_reference": gt_summary,
                "models": eval2_model_outputs,
            },
        }

        json_path = self._out_root / "symic_stage3_results.json"
        json_path.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_md = self._render_markdown_report(final_payload)
        md_path = self._out_root / "symic_stage3_report.md"
        md_path.write_text(report_md, encoding="utf-8")
        print(f"\n[Done] Results saved to:\n  - {json_path}\n  - {md_path}", flush=True)
        return final_payload

    def _execute_eval1_for_model(
        self,
        strategy: ParadigmEvaluationStrategy,
        all_ctx_segs: Sequence[TestSegmentItem],
        tier_buckets: Dict[str, List[TestSegmentItem]],
    ) -> Dict[str, object]:
        kv_bytes = int(round(strategy.active_kv_mib(1) * 1024 * 1024))
        all_steps: List[int] = []
        ens_steps: List[int] = []
        bar_trajs_ens: Dict[int, List[float]] = {4: [], 8: [], 12: [], 16: []}

        for seg in all_ctx_segs:
            s_len, traj = strategy.count_16bar_ar_steps_and_bar_checkpoints(seg)
            all_steps.append(s_len)
            if seg.active_slot_count >= 3 and seg.sounding_track_count >= 3:
                ens_steps.append(s_len)
                for b_mark, mib in traj:
                    bar_trajs_ens[b_mark].append(mib)

        # Warmup 2 segments before wall-clock hardware latency profiling
        warmup_pool = [s for lst in tier_buckets.values() for s in lst][:2]
        for w_seg in warmup_pool:
            _ = strategy.profile_ground_truth_horizon_rollout(
                w_seg, sampling=self._sampling, prompt_chunks=self._cfg.prompt_chunks
            )
        synchronize_device(self._device)

        tier_profiles: Dict[str, object] = {}
        for tier in self._tiers:
            segs = tier_buckets.get(tier.tier_id, [])
            if not segs:
                continue
            telems: List[RolloutTelemetry] = []
            npbs: List[float] = []
            for seg in segs:
                npbs.append(seg.notes_per_bar)
                telems.append(
                    strategy.profile_ground_truth_horizon_rollout(
                        seg,
                        sampling=self._sampling,
                        prompt_chunks=self._cfg.prompt_chunks,
                    )
                )
            summary = _summarize_telemetry_batch(telems)
            summary["tier_label"] = tier.label
            summary["mean_notes_per_bar"] = round(float(np.mean(npbs)), 2)
            tier_profiles[tier.tier_id] = summary
            print(
                f"  [Eval-1 | {strategy.spec.key} | {tier.tier_id}] "
                f"Notes/Bar={summary['mean_notes_per_bar']:.1f} | "
                f"Steps={summary['mean_total_steps']:.1f} | "
                f"Latency={summary['mean_ms_per_bar']:.1f} ms/bar ({summary['mean_bars_per_sec']:.2f} bars/s) | "
                f"KV={summary['mean_peak_kv_mib']:.2f} MiB",
                flush=True,
            )

        return {
            "display_name": strategy.spec.display_name,
            "kv_bytes_per_step": kv_bytes,
            "context_all_1to4_tracks": _summarize_step_distribution(all_steps, kv_bytes),
            "context_ensemble_3to4_tracks": {
                **_summarize_step_distribution(ens_steps, kv_bytes),
                "mean_kv_trajectory_mib": {
                    f"bar_{b}": round(float(np.mean(vals)), 3)
                    for b, vals in bar_trajs_ens.items()
                    if vals
                },
            },
            "density_tier_latency_profiles": tier_profiles,
        }

    def _execute_eval2_generation_for_model(
        self,
        strategy: ParadigmEvaluationStrategy,
        cont_segments: Sequence[TestSegmentItem],
        muspy_eval: MusPyContinuationEvaluator,
        gt_muspy_list: Sequence[MusPySongMetrics],
    ) -> Dict[str, object]:
        key = strategy.spec.key
        gen_16_dir = self._out_root / "midis" / key / "16bar"
        gen_cont_dir = self._out_root / "midis" / key / "cont12bar"
        telem_cache_path = self._out_root / "midis" / key / "telemetry_cache.json"
        gen_16_dir.mkdir(parents=True, exist_ok=True)
        gen_cont_dir.mkdir(parents=True, exist_ok=True)

        cached_telem: Dict[str, Dict[str, object]] = {}
        if not self._cfg.overwrite and telem_cache_path.is_file():
            try:
                cached_telem = json.loads(telem_cache_path.read_text(encoding="utf-8"))
            except Exception:
                cached_telem = {}

        # Warmup 1 run if any fresh generation will occur
        needs_gen = any(
            self._cfg.overwrite
            or not (gen_16_dir / f"{s.item_id}.mid").is_file()
            or s.item_id not in cached_telem
            for s in cont_segments
        )
        if needs_gen and cont_segments:
            _ = strategy.profile_ground_truth_horizon_rollout(
                cont_segments[0],
                sampling=self._sampling,
                prompt_chunks=self._cfg.prompt_chunks,
            )
            synchronize_device(self._device)

        telems: List[RolloutTelemetry] = []
        gen_muspy_list: List[MusPySongMetrics] = []
        t_start = time.perf_counter()

        for idx, seg in enumerate(cont_segments):
            out_16 = gen_16_dir / f"{seg.item_id}.mid"
            out_cont = gen_cont_dir / f"{seg.item_id}.mid"
            song_seed = self._cfg.seed + idx

            if not self._cfg.overwrite and out_16.is_file() and seg.item_id in cached_telem:
                rec = cached_telem[seg.item_id]
                traj_raw = rec.get("kv_bar_trajectory_mib", [])
                telem = RolloutTelemetry(
                    prompt_steps=int(rec["prompt_steps"]),
                    generated_steps=int(rec["generated_steps"]),
                    total_steps=int(rec["total_steps"]),
                    prefill_ms=float(rec["prefill_ms"]),
                    decode_ms=float(rec["decode_ms"]),
                    total_ms=float(rec["total_ms"]),
                    generated_bars=float(rec["generated_bars"]),
                    peak_active_kv_mib=float(rec["peak_active_kv_mib"]),
                    kv_bar_trajectory_mib=tuple(
                        (int(x["bar"]), float(x["kv_mib"])) for x in traj_raw
                    ),
                )
            else:
                telem = strategy.generate_continuation_to_midi(
                    segment=seg,
                    output_16bar_midi=out_16,
                    sampling=self._sampling,
                    prompt_chunks=self._cfg.prompt_chunks,
                    seed=song_seed,
                )
                cached_telem[seg.item_id] = telem.to_dict()
                if (idx + 1) % 20 == 0 or (idx + 1) == len(cont_segments):
                    telem_cache_path.write_text(
                        json.dumps(cached_telem, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

            telems.append(telem)
            n_tot, n_mel = StandardizedMidiSliceService.slice_16bar_midi_to_continuation_midi(
                midi_16bar_path=out_16,
                output_cont_midi=out_cont,
                slot_codes=seg.slot_codes,
                prompt_chunks=self._cfg.prompt_chunks,
                total_chunks=self._cfg.segment_chunks,
            )
            gen_muspy_list.append(
                muspy_eval.evaluate_single_continuation_midi(out_cont, n_tot, n_mel)
            )

            if (idx + 1) % 25 == 0 or (idx + 1) == len(cont_segments):
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [Eval-2 | {key}] {idx + 1}/{len(cont_segments)} songs "
                    f"({elapsed:.1f}s elapsed | last={telem.ms_per_bar:.1f} ms/bar, "
                    f"steps={telem.total_steps}, notes_cont={n_tot})",
                    flush=True,
                )

        telem_cache_path.write_text(
            json.dumps(cached_telem, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paired = muspy_eval.summarize_paired_errors(gt_muspy_list, gen_muspy_list)
        cont_note_counts = [m.note_count for m in gen_muspy_list]
        non_empty_songs = sum(1 for m in gen_muspy_list if m.note_count > 0)
        melodic_valid_songs = sum(1 for m in gen_muspy_list if m.has_valid_pce)

        return {
            "display_name": strategy.spec.display_name,
            "non_empty_continuation_songs": non_empty_songs,
            "melodic_valid_songs": melodic_valid_songs,
            "mean_continuation_notes": round(float(np.mean(cont_note_counts)), 2),
            "live_rollout_telemetry": _summarize_telemetry_batch(telems),
            "muspy_paired": {k: v.to_dict() for k, v in paired.items()},
        }

    @staticmethod
    def _build_gt_muspy_summary(gt_list: Sequence[MusPySongMetrics]) -> Dict[str, object]:
        pce = [m.pitch_class_entropy for m in gt_list if m.has_valid_pce]
        sc = [m.scale_consistency for m in gt_list if m.has_valid_sc]
        gc_vals = [m.groove_consistency for m in gt_list if m.has_valid_gc]
        notes = [m.note_count for m in gt_list]
        return {
            "num_songs": len(gt_list),
            "mean_continuation_notes": round(float(np.mean(notes)), 2),
            "pitch_class_entropy_mean": round(float(np.mean(pce)), 4),
            "pitch_class_entropy_std": round(float(np.std(pce)), 4),
            "scale_consistency_mean": round(float(np.mean(sc)), 4),
            "scale_consistency_std": round(float(np.std(sc)), 4),
            "groove_consistency_mean": round(float(np.mean(gc_vals)), 4),
            "groove_consistency_std": round(float(np.std(gc_vals)), 4),
        }

    def _render_markdown_report(self, payload: Dict[str, object]) -> str:
        lines: List[str] = [
            "# SyMIc Stage-3 Benchmark Evaluation Report",
            "",
            f"- **Device**: `{payload['device']}`",
            f"- **Continuation Task**: `{payload['prompt_bars']} -> {payload['total_bars']} bars` (`N = {payload['num_continuation_songs']}` unseen 3–4 track ensemble songs)",
            f"- **Uniform Sampling (§5.2.B)**: `{json.dumps(payload['sampling_protocol'])}`",
            "",
        ]

        eval1 = payload.get("evaluation_1_systems_efficiency", {})
        if isinstance(eval1, dict) and eval1:
            lines.extend(
                [
                    "## 1. Systems & Context Efficiency Scaling (§5.1)",
                    "",
                    "### 1.1 16-Bar Context Length & Active KV-Cache Footprint (3–4 Track Ensemble Segments)",
                    "",
                    "| Model | Mean $S$ | p50 $S$ | p90 $S$ | p99 $S$ | Max $S$ | Mean KV (MiB) | p90 KV (MiB) | Max KV (MiB) |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for cand in self._candidates:
                m1 = eval1.get(cand.key, {})
                if not isinstance(m1, dict):
                    continue
                ens = m1.get("context_ensemble_3to4_tracks", {})
                if not isinstance(ens, dict) or not ens:
                    continue
                lines.append(
                    f"| **{m1.get('display_name', cand.key)}** | `{ens['mean_steps']}` | `{ens['p50_steps']}` | "
                    f"`{ens['p90_steps']}` | `{ens['p99_steps']}` | `{ens['max_steps']}` | "
                    f"`{ens['mean_kv_mib']}` | `{ens['p90_kv_mib']}` | `{ens['max_kv_mib']}` |"
                )

            lines.extend(
                [
                    "",
                    "### 1.2 Hardware Latency (`ms/bar`, `bars/sec`) & KV Cache Across `Notes/Bar` Density Tiers",
                    "",
                    "| Density Tier | Avg Notes/Bar | Model | Total Steps $S$ | Latency (`ms/bar`) ↓ | Throughput (`bars/s`) ↑ | Peak KV (MiB) ↓ |",
                    "| :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
                ]
            )
            for tier in self._tiers:
                for cand in self._candidates:
                    m1 = eval1.get(cand.key, {})
                    if not isinstance(m1, dict):
                        continue
                    tp = m1.get("density_tier_latency_profiles", {}).get(tier.tier_id)
                    if not isinstance(tp, dict):
                        continue
                    lines.append(
                        f"| {tier.label} | `{tp['mean_notes_per_bar']}` | **{m1.get('display_name', cand.key)}** | "
                        f"`{tp['mean_total_steps']}` | **`{tp['mean_ms_per_bar']}`** | "
                        f"`{tp['mean_bars_per_sec']}` | `{tp['mean_peak_kv_mib']}` |"
                    )
            lines.append("")

        eval2 = payload.get("evaluation_2_continuation", {})
        if isinstance(eval2, dict) and eval2.get("models"):
            gt = eval2.get("ground_truth_reference", {})
            models_map = eval2["models"]
            lines.extend(
                [
                    "## 2. 4→16 Bar Prompted Continuation Objective Benchmark (§5.2)",
                    "",
                    "### 2.1 Primary Objective Metrics (`FMD [CLaMP 2]` ↓ & Per-Song Paired `MusPy` MAE ↓)",
                    "",
                    "| Model | **FMD (CLaMP 2)** ↓ | **PCE Paired MAE** ↓ (95% CI) | **SC Paired MAE** ↓ (95% CI) | **GC Paired MAE** ↓ (95% CI) | Live Latency (`ms/bar`) ↓ | Peak KV (MiB) ↓ |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for cand in self._candidates:
                md = models_map.get(cand.key)
                if not isinstance(md, dict):
                    continue
                mp = md["muspy_paired"]
                pce = mp["pitch_class_entropy"]
                sc = mp["scale_consistency"]
                gc_m = mp["groove_consistency"]
                tl = md["live_rollout_telemetry"]
                lines.append(
                    f"| **{md['display_name']}** | **`{md.get('fmd_clamp2', 'N/A')}`** | "
                    f"`{pce['paired_mae']:.4f}` (`[{pce['paired_mae_ci95_low']:.4f}, {pce['paired_mae_ci95_high']:.4f}]`) | "
                    f"`{sc['paired_mae']:.4f}` (`[{sc['paired_mae_ci95_low']:.4f}, {sc['paired_mae_ci95_high']:.4f}]`) | "
                    f"`{gc_m['paired_mae']:.4f}` (`[{gc_m['paired_mae_ci95_low']:.4f}, {gc_m['paired_mae_ci95_high']:.4f}]`) | "
                    f"`{tl['mean_ms_per_bar']}` | `{tl['mean_peak_kv_mib']}` |"
                )

            lines.extend(
                [
                    "",
                    "### 2.2 Corpus-Level Continuation Distribution vs Ground Truth (Bars 5–16)",
                    "",
                    "| Source / Model | Valid Songs | Avg Cont. Notes | Pitch Class Entropy ($H_{PC}$) | Scale Consistency ($SC$) | Groove Consistency ($GC$) |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |",
                    f"| **Ground Truth (Human)** | `{gt.get('num_songs', 200)}/{gt.get('num_songs', 200)}` | `{gt.get('mean_continuation_notes')}` | "
                    f"`{gt.get('pitch_class_entropy_mean'):.4f} ± {gt.get('pitch_class_entropy_std'):.4f}` | "
                    f"`{gt.get('scale_consistency_mean'):.4f} ± {gt.get('scale_consistency_std'):.4f}` | "
                    f"`{gt.get('groove_consistency_mean'):.4f} ± {gt.get('groove_consistency_std'):.4f}` |",
                ]
            )
            for cand in self._candidates:
                md = models_map.get(cand.key)
                if not isinstance(md, dict):
                    continue
                mp = md["muspy_paired"]
                pce = mp["pitch_class_entropy"]
                sc = mp["scale_consistency"]
                gc_m = mp["groove_consistency"]
                lines.append(
                    f"| **{md['display_name']}** | `{md['melodic_valid_songs']}/{pce['total_songs']}` | `{md['mean_continuation_notes']}` | "
                    f"`{pce['corpus_mean']:.4f} ± {pce['corpus_std']:.4f}` | "
                    f"`{sc['corpus_mean']:.4f} ± {sc['corpus_std']:.4f}` | "
                    f"`{gc_m['corpus_mean']:.4f} ± {gc_m['corpus_std']:.4f}` |"
                )
            lines.append("")

        return "\n".join(lines)
