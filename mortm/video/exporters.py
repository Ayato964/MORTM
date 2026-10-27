from __future__ import annotations

import glob
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from mortm.utils.convert_pianoroll import CHUNK_STEPS, TOTAL_CHANNELS, chunks_to_midi

from .decoder_engine import MidiSampleContext
from .domain import DecodedFrameState, Resolution


class FrameStreamSink(ABC):
    """Context-managed streaming sink consuming rendered RGBA frames in O(1) memory."""

    def __enter__(self) -> "FrameStreamSink":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close(had_error=(exc_type is not None))

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def write_frame(self, rgba_frame: np.ndarray, state: DecodedFrameState) -> None:
        ...

    @abstractmethod
    def close(self, had_error: bool = False) -> None:
        ...


class AvFoundationMp4Sink(FrameStreamSink):
    """Streams raw RGBA frames directly into macOS native Hardware H.264 MP4 encoder."""

    def __init__(
        self,
        output_path: Path,
        resolution: Resolution,
        fps: int,
        bitrate_bps: int = 14_000_000,
    ) -> None:
        self._output_path = output_path
        self._resolution = resolution
        self._fps = fps
        self._bitrate_bps = bitrate_bps
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def _ensure_native_binary() -> Path:
        native_dir = Path(__file__).resolve().parent / "native"
        bin_path = native_dir / "rgba_to_mp4"
        src_path = native_dir / "rgba_to_mp4.m"

        if bin_path.is_file() and (not src_path.is_file() or bin_path.stat().st_mtime >= src_path.stat().st_mtime):
            return bin_path

        if not src_path.is_file():
            raise FileNotFoundError(f"Native encoder source missing: {src_path}")

        sdk_candidates = sorted(
            glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk"),
            reverse=True,
        )
        # Try default clang first, then explicit SDK fallbacks (handles beta CLT SDK mismatches)
        compile_attempts: list[list[str]] = [[]] + [["-isysroot", sdk] for sdk in sdk_candidates]
        last_err = ""
        for extra_flags in compile_attempts:
            cmd = [
                "clang",
                "-O3",
                "-fobjc-arc",
                *extra_flags,
                str(src_path),
                "-framework", "Foundation",
                "-framework", "AVFoundation",
                "-framework", "CoreMedia",
                "-framework", "CoreVideo",
                "-framework", "Accelerate",
                "-o", str(bin_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and bin_path.is_file():
                return bin_path
            last_err = res.stderr.strip()

        raise RuntimeError(f"Failed to compile native AVFoundation MP4 encoder:\n{last_err}")

    def open(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        bin_path = self._ensure_native_binary()
        cmd = [
            str(bin_path),
            str(self._output_path),
            str(self._resolution.width),
            str(self._resolution.height),
            str(self._fps),
            str(self._bitrate_bps),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, rgba_frame: np.ndarray, state: DecodedFrameState) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("AvFoundationMp4Sink is not open.")
        expected_shape = (self._resolution.height, self._resolution.width, 4)
        if rgba_frame.shape != expected_shape:
            raise ValueError(f"Expected RGBA frame {expected_shape}, got {rgba_frame.shape}")
        self._proc.stdin.write(memoryview(np.ascontiguousarray(rgba_frame, dtype=np.uint8)))

    def close(self, had_error: bool = False) -> None:
        if self._proc is None:
            return
        stderr_txt = ""
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            if self._proc.stderr is not None:
                stderr_txt = self._proc.stderr.read().decode("utf-8", errors="replace")
            ret = self._proc.wait(timeout=30)
            if ret != 0 and not had_error:
                raise RuntimeError(f"MP4 encoder exited with code {ret}: {stderr_txt}")
        finally:
            self._proc = None


class PreviewGifSink(FrameStreamSink):
    """Downsampled stride-subsampled animated GIF sink for lightweight web/markdown previews."""

    def __init__(
        self,
        output_path: Path,
        source_fps: int,
        frame_stride: int = 3,
        preview_width: int = 960,
    ) -> None:
        self._output_path = output_path
        self._source_fps = source_fps
        self._stride = max(1, frame_stride)
        self._preview_width = preview_width
        self._frames: list[Image.Image] = []

    def open(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._frames.clear()

    def write_frame(self, rgba_frame: np.ndarray, state: DecodedFrameState) -> None:
        if state.frame_index % self._stride != 0 and state.frame_index != state.total_frames - 1:
            return
        h, w, _ = rgba_frame.shape
        target_h = int(round(h * (self._preview_width / float(w))))
        rgb_img = Image.fromarray(rgba_frame[..., :3], mode="RGB").resize(
            (self._preview_width, target_h),
            resample=Image.Resampling.BILINEAR,
        )
        # Quantize immediately to 8-bit palette (1/3 memory of RGB)
        self._frames.append(rgb_img.quantize(colors=224, method=Image.Quantize.FASTOCTREE))

    def close(self, had_error: bool = False) -> None:
        if had_error or not self._frames:
            self._frames.clear()
            return
        frame_duration_ms = max(20, int(round(1000.0 * self._stride / float(self._source_fps))))
        first, rest = self._frames[0], self._frames[1:]
        first.save(
            str(self._output_path),
            save_all=True,
            append_images=rest,
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
        )
        self._frames.clear()


class StageSnapshotArtifactSink(FrameStreamSink):
    """Exports per-stage keyframe PNGs and reconstructed MIDI files at hold anchors."""

    def __init__(self, output_dir: Path, sample: MidiSampleContext) -> None:
        self._output_dir = output_dir
        self._sample = sample
        self._exported_stages: set[str] = set()

    def open(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._exported_stages.clear()
        # Also export Ground-Truth 8-bar MIDI for direct listening comparison
        gt_midi_path = self._output_dir / "99_ground_truth_8bars.mid"
        chunks_to_midi(self._sample.gt_chunks, self._sample.roll_meta, gt_midi_path)

    def write_frame(self, rgba_frame: np.ndarray, state: DecodedFrameState) -> None:
        stage_id = state.descriptor.stage_id
        if stage_id in self._exported_stages:
            return
        # Capture once when stage reaches hold phase (>= 60% into stage)
        if state.descriptor.progress_in_stage < 0.65 and state.frame_index < state.total_frames - 1:
            return

        self._exported_stages.add(stage_id)
        png_path = self._output_dir / f"{stage_id}.png"
        Image.fromarray(rgba_frame, mode="RGBA").save(str(png_path))

        midi_path = self._output_dir / f"{stage_id}.mid"
        # Convert probs_flat [8, steps, 128] -> [N_chunks, 8, 24, 128]
        n_chunks = self._sample.bar_window.num_chunks
        chunks_time_pitch = (
            state.probs_flat.reshape(TOTAL_CHANNELS, n_chunks, CHUNK_STEPS, 128)
            .transpose(1, 0, 2, 3)
            .copy()
        )
        chunks_to_midi(chunks_time_pitch, self._sample.roll_meta, midi_path)

    def close(self, had_error: bool = False) -> None:
        pass


class CompositeFrameSink(FrameStreamSink):
    """Composite pattern fan-out sink forwarding each streamed frame to multiple sinks."""

    def __init__(self, sinks: Sequence[FrameStreamSink]) -> None:
        self._sinks = tuple(sinks)

    def open(self) -> None:
        for sink in self._sinks:
            sink.open()

    def write_frame(self, rgba_frame: np.ndarray, state: DecodedFrameState) -> None:
        for sink in self._sinks:
            sink.write_frame(rgba_frame, state)

    def close(self, had_error: bool = False) -> None:
        first_err: Exception | None = None
        for sink in self._sinks:
            try:
                sink.close(had_error=had_error)
            except Exception as exc:
                if first_err is None:
                    first_err = exc
        if first_err is not None and not had_error:
            raise first_err
