from __future__ import annotations

import argparse
from typing import Optional, Sequence
import warnings

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

from .domain import DEFAULT_SAMPLING_SPEC, DEFAULT_STAGE3_CANDIDATES, SamplingSpec
from .runner import BenchmarkRunConfig, SymicStage3BenchmarkRunner


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MORTM5 Stage-3 Benchmark: Systems Efficiency (§5.1) & 4->16 Bar Continuation (§5.2)"
    )
    parser.add_argument("--eval_json", type=str, default="data/gmd/eval.json")
    parser.add_argument("--output_dir", type=str, default="out/symic_main/benchmark_eval")
    parser.add_argument("--num_songs", type=int, default=200)
    parser.add_argument("--prompt_chunks", type=int, default=32)
    parser.add_argument("--segment_chunks", type=int, default=128)
    parser.add_argument("--context_scan_songs", type=int, default=200)
    parser.add_argument("--latency_per_tier", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING_SPEC.temperature)
    parser.add_argument("--fine_temperature", type=float, default=DEFAULT_SAMPLING_SPEC.fine_temperature)
    parser.add_argument("--top_p", type=float, default=DEFAULT_SAMPLING_SPEC.top_p)
    parser.add_argument("--top_k", type=int, default=DEFAULT_SAMPLING_SPEC.top_k)
    parser.add_argument("--min_p", type=float, default=DEFAULT_SAMPLING_SPEC.min_p)
    parser.add_argument("--repetition_penalty", type=float, default=DEFAULT_SAMPLING_SPEC.repetition_penalty)
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model keys")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_eval1", action="store_true")
    parser.add_argument("--skip_eval2", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sampling = SamplingSpec(
        temperature=args.temperature,
        fine_temperature=args.fine_temperature,
        top_p=args.top_p,
        fine_top_p=args.top_p,
        top_k=args.top_k,
        fine_top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )
    candidates = DEFAULT_STAGE3_CANDIDATES
    if args.models:
        wanted = {m.strip() for m in args.models}
        candidates = tuple(c for c in DEFAULT_STAGE3_CANDIDATES if c.key in wanted)
        if not candidates:
            raise ValueError(
                f"No matching candidates for {args.models}. Available: {[c.key for c in DEFAULT_STAGE3_CANDIDATES]}"
            )

    run_cfg = BenchmarkRunConfig(
        eval_json=args.eval_json,
        output_dir=args.output_dir,
        num_continuation_songs=args.num_songs,
        prompt_chunks=args.prompt_chunks,
        segment_chunks=args.segment_chunks,
        context_scan_songs=args.context_scan_songs,
        latency_per_tier=args.latency_per_tier,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        run_eval1=not args.skip_eval1,
        run_eval2=not args.skip_eval2,
    )
    runner = SymicStage3BenchmarkRunner(
        config=run_cfg,
        candidates=candidates,
        sampling=sampling,
    )
    runner.run()


if __name__ == "__main__":
    main()
