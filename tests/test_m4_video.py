from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pytest
import torch

from mortm.utils.convert_pianoroll import RollMeta, SlotInfo
from mortm.video.decoder_engine import CodebookDecomposition, LatentDecodeService, MidiSampleContext
from mortm.video.domain import (
    DEFAULT_SLOT_PALETTE,
    BarWindow,
    DecodedFrameState,
    PitchWindow,
    QuantizedNote,
    Resolution,
    SlotMetrics,
    StageMetrics,
    TrackSlotSpec,
)
from mortm.video.exporters import FrameStreamSink
from mortm.video.pipeline import CodebookAnimationPipeline
from mortm.video.renderer import FrameRenderer
from mortm.video.strategies import build_default_strategy_registry


def _make_dummy_decomposition(n_chunks: int = 8, dim: int = 128) -> CodebookDecomposition:
    z0 = torch.zeros((n_chunks, dim), dtype=torch.float32)
    layers = tuple(torch.full((n_chunks, dim), float(i + 1), dtype=torch.float32) for i in range(4))
    return CodebookDecomposition(
        z_silent=z0 - 1.0,
        z_origin=z0,
        z_noise=torch.randn((n_chunks, dim), dtype=torch.float32),
        z_continuous=sum(layers),
        layer_latents=layers,
        layer_mean_norms=(8.0, 4.5, 3.2, 2.5),
        codebook_indices=torch.zeros((4, n_chunks, 4), dtype=torch.long),
    )


class _StubDecodeService(LatentDecodeService):
    def decompose(self, sample: MidiSampleContext, noise_seed: int = 42) -> CodebookDecomposition:
        return _make_dummy_decomposition(n_chunks=sample.bar_window.num_chunks, dim=16)

    def decode_batch(
        self,
        latent_batch: Sequence[torch.Tensor],
        sample: MidiSampleContext,
        z_reference: torch.Tensor,
        mask_inactive_slots: bool = False,
    ) -> list[tuple[np.ndarray, Tuple[QuantizedNote, ...], StageMetrics]]:
        out = []
        steps = sample.bar_window.total_steps
        for z in latent_batch:
            val = float(torch.sigmoid(z.mean()).item())
            probs = np.full((8, steps, 128), val, dtype=np.float32)
            sm = tuple(SlotMetrics(s, val, val, 0.1) for s in range(4))
            metrics = StageMetrics(val, val, val, val, val, val, 0.1, 0.2, 10, sm)
            out.append((probs, (), metrics))
        return out


class _StubRenderer(FrameRenderer):
    @property
    def resolution(self) -> Resolution:
        return Resolution(640, 360)

    def render_frame(self, state: DecodedFrameState) -> np.ndarray:
        return np.zeros((360, 640, 4), dtype=np.uint8)


class _RecordingSink(FrameStreamSink):
    def __init__(self) -> None:
        self.states: list[DecodedFrameState] = []
        self.closed = False

    def open(self) -> None:
        self.states.clear()

    def write_frame(self, rgba_frame: np.ndarray, state: DecodedFrameState) -> None:
        self.states.append(state)

    def close(self, had_error: bool = False) -> None:
        self.closed = True


def test_resolution_enforces_even_dimensions() -> None:
    res = Resolution(1920, 1080)
    assert res.byte_size_rgba == 1920 * 1080 * 4
    with pytest.raises(ValueError):
        Resolution(1921, 1080)


def test_bar_window_geometry() -> None:
    bw = BarWindow(num_bars=8)
    assert bw.num_chunks == 64
    assert bw.total_steps == 1536


def test_pitch_window_validation() -> None:
    pw = PitchWindow(24, 84)
    assert pw.span == 61
    with pytest.raises(ValueError):
        PitchWindow(90, 60)


def test_codebook_cumulative_latent_exact_sum() -> None:
    decomp = _make_dummy_decomposition()
    assert torch.allclose(decomp.cumulative_latent(0), decomp.z_origin)
    assert torch.allclose(decomp.cumulative_latent(2), decomp.layer_latents[0] + decomp.layer_latents[1])
    assert torch.allclose(decomp.cumulative_latent(4), decomp.z_continuous)


def test_all_registered_strategies_yield_consistent_frame_counts() -> None:
    registry = build_default_strategy_registry()
    decomp = _make_dummy_decomposition()
    for name in registry.available_names():
        strat = registry.get(name)
        steps = list(strat.iter_steps(decomp, fps=10))
        assert len(steps) > 0
        assert steps[0].frame_index == 0
        assert steps[-1].frame_index == len(steps) - 1
        assert all(st.total_frames == len(steps) for st in steps)


def test_pipeline_populates_delta_reference_across_all_stages() -> None:
    bw = BarWindow(num_bars=1)
    slots = [SlotInfo(slot=i, is_drum=(i == 0), program=0, name=f"s{i}", source_track=i, source_channel=i, note_count=1) for i in range(4)]
    meta = RollMeta(ticks_per_beat=480, steps_per_bar=192, beats_per_bar=4, max_tracks=4, features_per_track=2, slots=slots)
    specs = tuple(TrackSlotSpec(i, i == 0, 0, f"s{i}", True, DEFAULT_SLOT_PALETTE[i]) for i in range(4))
    sample = MidiSampleContext(
        source_path=Path("dummy.mid"),
        bar_window=bw,
        roll_meta=meta,
        gt_chunks=np.zeros((8, 8, 128, 24), dtype=np.float32),
        gt_flat=np.zeros((8, 192, 128), dtype=np.float32),
        gt_notes=(),
        slot_specs=specs,
        pitch_window=PitchWindow(24, 84),
    )

    strat = build_default_strategy_registry().get("noise_to_fine")
    sink = _RecordingSink()
    pipeline = CodebookAnimationPipeline(
        decode_service=_StubDecodeService(),
        strategy=strat,
        renderer_factory=lambda s, d: _StubRenderer(),
        decode_batch_size=4,
    )
    count = pipeline.execute(sample=sample, sink=sink, fps=4)
    assert count == len(sink.states)
    assert sink.closed
    assert all(st.prev_probs_flat is not None for st in sink.states)


def test_dashboard_renderer_enforces_90_10_color_budget() -> None:
    from mortm.video.domain import StageDescriptor
    from mortm.video.renderer import PianoRollDashboardRenderer

    bw = BarWindow(num_bars=2)
    slots = [
        SlotInfo(slot=i, is_drum=(i == 0), program=i * 8, name=f"s{i}", source_track=i, source_channel=i, note_count=4)
        for i in range(4)
    ]
    meta = RollMeta(ticks_per_beat=480, steps_per_bar=192, beats_per_bar=4, max_tracks=4, features_per_track=2, slots=slots)
    specs = tuple(TrackSlotSpec(i, i == 0, i * 8, f"s{i}", True, DEFAULT_SLOT_PALETTE[i]) for i in range(4))
    gt_notes = (
        QuantizedNote(slot=0, start_step=0, end_step=24, pitch=36, velocity=100),
        QuantizedNote(slot=1, start_step=48, end_step=96, pitch=60, velocity=90),
    )
    sample = MidiSampleContext(
        source_path=Path("dummy.mid"),
        bar_window=bw,
        roll_meta=meta,
        gt_chunks=np.zeros((16, 8, 128, 24), dtype=np.float32),
        gt_flat=np.zeros((8, bw.total_steps, 128), dtype=np.float32),
        gt_notes=gt_notes,
        slot_specs=specs,
        pitch_window=PitchWindow(24, 84),
    )
    decomp = _make_dummy_decomposition(n_chunks=bw.num_chunks, dim=128)
    renderer = PianoRollDashboardRenderer(sample=sample, decomposition=decomp, resolution=Resolution(1280, 720), fps=30)

    sm = tuple(SlotMetrics(s, 0.9, 0.95, 0.05) for s in range(4))
    state = DecodedFrameState(
        frame_index=15,
        total_frames=60,
        descriptor=StageDescriptor(
            stage_id="02_layer1",
            badge_text="Layer 1",
            title_ja="Layer 1 — リズム骨格と主要和音",
            subtitle_ja="ノイズが一掃され、ドラムの主拍と基本コードの大枠が立ち現れます。",
            active_layers_mask=(True, False, False, False),
            highlight_layer_index=0,
            progress_in_stage=0.5,
            timeline_progress=0.25,
        ),
        probs_flat=np.zeros((8, bw.total_steps, 128), dtype=np.float32),
        prev_probs_flat=np.zeros((8, bw.total_steps, 128), dtype=np.float32),
        notes=gt_notes,
        metrics=StageMetrics(0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.05, 0.1, 2, sm),
    )
    rgba = renderer.render_frame(state)
    assert rgba.shape == (720, 1280, 4)
    assert rgba.dtype == np.uint8

    # Verify >= 90% of pixels are low-chroma neutral (max(R,G,B) - min(R,G,B) <= 16)
    rgb = rgba[:, :, :3].astype(np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    neutral_ratio = float((chroma <= 16).mean())
    assert neutral_ratio >= 0.90

