"""
CLI Entrypoint for MORTM5 MidiVision M4 Codebook Layer-wise Decoding Video Generator.

Usage examples:
  # Default 4-track GMD sample (8 bars, Empty -> Noise -> L1 -> L1+2 -> L1+2+3 -> L1..4)
  .venv/bin/python -m mortm.video.render_m4_codebook_video

  # Specific MIDI file (e.g., Sample2.mid or custom MIDI) + GIF & stage MIDI exports
  .venv/bin/python -m mortm.video.render_m4_codebook_video \
      --midi data/generate/Sample2.mid \
      --out out/video/m4_sample2_layers.mp4 \
      --gif out/video/m4_sample2_layers.gif \
      --artifacts-dir out/video/sample2_stages
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*")

# Ensure repository root is importable when invoked directly as a script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mortm.video.decoder_engine import MidiSampleContext, TorchMidiVisionDecodeService
from mortm.video.domain import AnimationConfig, BarWindow, Resolution
from mortm.video.exporters import (
    AvFoundationMp4Sink,
    CompositeFrameSink,
    FrameStreamSink,
    PreviewGifSink,
    StageSnapshotArtifactSink,
)
from mortm.video.pipeline import CodebookAnimationPipeline
from mortm.video.renderer import PianoRollDashboardRenderer
from mortm.video.strategies import build_default_strategy_registry


_DEFAULT_MIDI_CANDIDATES: tuple[str, ...] = (
    "data/gmd/training/all-instruments-with-drums/9/4438331384f79f370621e974d9c4ecec.mid",
    "data/gmd/training/all-instruments-with-drums/11/41643df2f05c719730f570c47a4c40b6.mid",
    "data/generate/Sample2.mid",
)


def resolve_default_midi(repo_root: Path) -> Path:
    for rel in _DEFAULT_MIDI_CANDIDATES:
        candidate = repo_root / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No default MIDI sample found under data/. Please specify --midi <path.mid>.")


def parse_cli_args(argv: Sequence[str] | None = None) -> AnimationConfig:
    registry = build_default_strategy_registry()
    parser = argparse.ArgumentParser(
        description="Render M4 MidiVision RVQ Codebook layer-by-layer CNN reconstruction video."
    )
    parser.add_argument(
        "--midi",
        type=Path,
        default=None,
        help="Input MIDI file path (defaults to an 8-bar 4-track sample in data/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "out" / "video" / "m4_codebook_layers_8bars.mp4",
        help="Output H.264 MP4 video path.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Optional output animated GIF preview path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_REPO_ROOT / "out" / "video" / "stages",
        help="Directory to export per-stage keyframe PNGs and reconstructed .mid files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="noise_to_fine",
        choices=registry.available_names(),
        help="Latent progression strategy.",
    )
    parser.add_argument("--bars", type=int, default=8, help="Number of opening bars to decode (default: 8).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    parser.add_argument("--width", type=int, default=1920, help="Video width in pixels (even integer).")
    parser.add_argument("--height", type=int, default=1080, help="Video height in pixels (even integer).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for latent prior noise.")
    parser.add_argument(
        "--mask-inactive-slots",
        action="store_true",
        help="Zero-mask slots that have no notes in ground truth.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "models" / "mortm5" / "m4_deeprvq_g4l4_d128_k2048.json",
        help="Path to M4 MidiVision JSON config.",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=_REPO_ROOT
        / "out"
        / "models"
        / "mortm5"
        / "m4_deeprvq_g4l4_d128_k2048"
        / "MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth",
        help="Path to M4 MidiVision .pth weights.",
    )

    ns = parser.parse_args(argv)
    midi_path = ns.midi if ns.midi is not None else resolve_default_midi(_REPO_ROOT)

    return AnimationConfig(
        midi_path=midi_path.resolve(),
        output_mp4_path=ns.out.resolve(),
        output_gif_path=ns.gif.resolve() if ns.gif is not None else None,
        export_stage_midis_dir=ns.artifacts_dir.resolve() if ns.artifacts_dir is not None else None,
        model_config_path=ns.model_config.resolve(),
        model_ckpt_path=ns.model_ckpt.resolve(),
        strategy_name=ns.mode,
        bars=ns.bars,
        fps=ns.fps,
        resolution=Resolution(width=ns.width, height=ns.height),
        seed=ns.seed,
        mask_inactive_slots=bool(ns.mask_inactive_slots),
    )


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_cli_args(argv)
    registry = build_default_strategy_registry()
    strategy = registry.get(cfg.strategy_name)

    print("=" * 78)
    print("MidiVision M4 Codebook Layer-wise Decoding Video Generator")
    print(f"  Input MIDI   : {cfg.midi_path}")
    print(f"  Opening Bars : {cfg.bars} bars ({cfg.bars * 8} chunks / {cfg.bars * 192} steps)")
    print(f"  Strategy     : {cfg.strategy_name}")
    print(f"  Resolution   : {cfg.resolution.width}x{cfg.resolution.height} @ {cfg.fps} fps")
    print(f"  Output MP4   : {cfg.output_mp4_path}")
    if cfg.output_gif_path is not None:
        print(f"  Output GIF   : {cfg.output_gif_path}")
    if cfg.export_stage_midis_dir is not None:
        print(f"  Stage Exports: {cfg.export_stage_midis_dir}")
    print("=" * 78)

    bar_window = BarWindow(num_bars=cfg.bars)
    sample = MidiSampleContext.from_midi_file(cfg.midi_path, bar_window=bar_window)
    decode_service = TorchMidiVisionDecodeService(
        model_config_path=cfg.model_config_path,
        model_ckpt_path=cfg.model_ckpt_path,
    )
    print(f"  Compute Dev  : {decode_service.device} (GT notes in {cfg.bars} bars: {len(sample.gt_notes)})")

    sinks: list[FrameStreamSink] = [
        AvFoundationMp4Sink(
            output_path=cfg.output_mp4_path,
            resolution=cfg.resolution,
            fps=cfg.fps,
        )
    ]
    if cfg.output_gif_path is not None:
        sinks.append(PreviewGifSink(output_path=cfg.output_gif_path, source_fps=cfg.fps))
    if cfg.export_stage_midis_dir is not None:
        sinks.append(StageSnapshotArtifactSink(output_dir=cfg.export_stage_midis_dir, sample=sample))

    composite_sink = CompositeFrameSink(sinks)
    pipeline = CodebookAnimationPipeline(
        decode_service=decode_service,
        strategy=strategy,
        renderer_factory=lambda s, d: PianoRollDashboardRenderer(
            sample=s,
            decomposition=d,
            resolution=cfg.resolution,
            fps=cfg.fps,
        ),
        decode_batch_size=8,
    )

    def _report_progress(done: int, total: int, elapsed: float) -> None:
        if done == total or done % 64 == 0:
            rate = done / max(elapsed, 1e-6)
            print(f"  [{done:4d}/{total:4d}] ({done * 100.0 / total:5.1f}%) | {rate:5.1f} frames/s | {elapsed:5.1f}s")

    total_rendered = pipeline.execute(
        sample=sample,
        sink=composite_sink,
        fps=cfg.fps,
        noise_seed=cfg.seed,
        mask_inactive_slots=cfg.mask_inactive_slots,
        progress_callback=_report_progress,
    )
    print(f"Completed: {total_rendered} frames -> {cfg.output_mp4_path}")


if __name__ == "__main__":
    main()
