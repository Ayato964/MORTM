from __future__ import annotations

import itertools
import time
from typing import Callable, Iterator, TypeVar

import numpy as np
import torch

from .decoder_engine import CodebookDecomposition, LatentDecodeService, MidiSampleContext
from .domain import DecodedFrameState
from .exporters import FrameStreamSink
from .renderer import FrameRenderer
from .strategies import LatentTrajectoryStep, TrajectoryStrategy


T = TypeVar("T")


def _chunked_iterator(iterable: Iterator[T], chunk_size: int) -> Iterator[list[T]]:
    """Yield bounded-size chunks from an iterator in O(chunk_size) memory."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


class CodebookAnimationPipeline:
    """Coordinates RVQ decomposition, batched CNN decoding, frame rendering, and streaming export."""

    def __init__(
        self,
        decode_service: LatentDecodeService,
        strategy: TrajectoryStrategy,
        renderer_factory: Callable[[MidiSampleContext, CodebookDecomposition], FrameRenderer],
        decode_batch_size: int = 8,
    ) -> None:
        if decode_batch_size <= 0:
            raise ValueError(f"decode_batch_size must be positive, got {decode_batch_size}")
        self._decode_service = decode_service
        self._strategy = strategy
        self._renderer_factory = renderer_factory
        self._decode_batch_size = decode_batch_size

    def execute(
        self,
        sample: MidiSampleContext,
        sink: FrameStreamSink,
        fps: int = 30,
        noise_seed: int = 42,
        mask_inactive_slots: bool = False,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> int:
        t_start = time.perf_counter()
        decomp = self._decode_service.decompose(sample, noise_seed=noise_seed)
        renderer = self._renderer_factory(sample, decomp)

        # Pre-decode delta reference latents keyed deterministically by stage_id (O(1) lookup)
        waypoints = self._strategy.build_waypoints(decomp)
        ref_latents_by_stage: dict[str, torch.Tensor] = {
            wp.stage_id: wp.delta_reference_latent
            for wp in waypoints
            if wp.delta_reference_latent is not None
        }

        ref_probs_by_stage: dict[str, np.ndarray] = {}
        if ref_latents_by_stage:
            stage_ids = list(ref_latents_by_stage.keys())
            ref_latents = [ref_latents_by_stage[sid] for sid in stage_ids]
            decoded_refs = self._decode_service.decode_batch(
                latent_batch=ref_latents,
                sample=sample,
                z_reference=decomp.z_continuous,
                mask_inactive_slots=mask_inactive_slots,
            )
            for sid, (probs_flat, _, _) in zip(stage_ids, decoded_refs):
                ref_probs_by_stage[sid] = probs_flat

        rendered_count = 0
        step_stream = self._strategy.iter_steps(decomp, fps=fps)

        with sink:
            for step_batch in _chunked_iterator(step_stream, self._decode_batch_size):
                latents = [st.latent for st in step_batch]
                decoded_batch = self._decode_service.decode_batch(
                    latent_batch=latents,
                    sample=sample,
                    z_reference=decomp.z_continuous,
                    mask_inactive_slots=mask_inactive_slots,
                )

                for step, (probs_flat, notes, metrics) in zip(step_batch, decoded_batch):
                    prev_probs = ref_probs_by_stage.get(step.descriptor.stage_id)
                    frame_state = DecodedFrameState(
                        frame_index=step.frame_index,
                        total_frames=step.total_frames,
                        descriptor=step.descriptor,
                        probs_flat=probs_flat,
                        prev_probs_flat=prev_probs,
                        notes=notes,
                        metrics=metrics,
                    )
                    rgba_frame = renderer.render_frame(frame_state)
                    sink.write_frame(rgba_frame, frame_state)
                    rendered_count += 1

                if progress_callback is not None and step_batch:
                    elapsed = time.perf_counter() - t_start
                    progress_callback(rendered_count, step_batch[-1].total_frames, elapsed)

        return rendered_count
