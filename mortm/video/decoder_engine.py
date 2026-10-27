from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from mortm.models.modules.config import MORTM5Args
from mortm.models.mortm5 import MidiVision
from mortm.utils.convert_pianoroll import (
    CHUNK_STEPS,
    TOTAL_CHANNELS,
    RollMeta,
    _flat_to_quantized_notes,
    midi_to_chunks,
)

from .domain import (
    DEFAULT_SLOT_PALETTE,
    BarWindow,
    PitchWindow,
    QuantizedNote,
    SlotMetrics,
    StageMetrics,
    TrackSlotSpec,
)


@dataclass(frozen=True)
class MidiSampleContext:
    """Encapsulates the 8-bar ground-truth MIDI sample and its metadata."""
    source_path: Path
    bar_window: BarWindow
    roll_meta: RollMeta
    gt_chunks: np.ndarray          # [N_chunks, 8, 128, 24] float32 in [0, 1]
    gt_flat: np.ndarray            # [8, total_steps, 128] float32 in [0, 1]
    gt_notes: Tuple[QuantizedNote, ...]
    slot_specs: Tuple[TrackSlotSpec, ...]
    pitch_window: PitchWindow

    @classmethod
    def from_midi_file(cls, midi_path: Path, bar_window: BarWindow) -> "MidiSampleContext":
        if not midi_path.is_file():
            raise FileNotFoundError(f"MIDI sample not found: {midi_path}")

        raw_chunks, meta = midi_to_chunks(midi_path)
        target_chunks = bar_window.num_chunks

        if raw_chunks.shape[0] >= target_chunks:
            sliced = raw_chunks[:target_chunks]
        else:
            pad_len = target_chunks - raw_chunks.shape[0]
            pad = np.zeros((pad_len, *raw_chunks.shape[1:]), dtype=np.float32)
            sliced = np.concatenate([raw_chunks, pad], axis=0)

        # raw_chunks is [N, 8, 24, 128]; MidiVision expects [N, 8, 128, 24]
        gt_chunks = np.ascontiguousarray(sliced.transpose(0, 1, 3, 2), dtype=np.float32)
        gt_flat = np.ascontiguousarray(
            sliced.transpose(1, 0, 2, 3).reshape(TOTAL_CHANNELS, bar_window.total_steps, 128),
            dtype=np.float32,
        )

        raw_notes = _flat_to_quantized_notes(gt_flat)
        gt_notes = tuple(
            QuantizedNote(slot=s, start_step=st, end_step=ed, pitch=p, velocity=v)
            for s, st, ed, p, v in raw_notes
        )

        slot_specs = []
        for s_idx in range(4):
            s_info = meta.slots[s_idx]
            active_in_window = bool(np.any(gt_flat[s_idx * 2 + 1] > 0.0))
            slot_specs.append(
                TrackSlotSpec(
                    slot_index=s_idx,
                    is_drum=bool(s_info.is_drum or s_idx == 0),
                    program=int(s_info.program),
                    raw_name=str(s_info.name),
                    is_active=active_in_window,
                    color=DEFAULT_SLOT_PALETTE[s_idx],
                )
            )

        pitch_window = PitchWindow.from_ground_truth_roll(gt_flat)
        return cls(
            source_path=midi_path,
            bar_window=bar_window,
            roll_meta=meta,
            gt_chunks=gt_chunks,
            gt_flat=gt_flat,
            gt_notes=gt_notes,
            slot_specs=tuple(slot_specs),
            pitch_window=pitch_window,
        )


@dataclass(frozen=True)
class CodebookDecomposition:
    """Holds decomposed Grouped-RVQ layer latent vectors for an encoded MIDI excerpt."""
    z_silent: torch.Tensor                        # [N_chunks, D]
    z_origin: torch.Tensor                        # [N_chunks, D] (zeros)
    z_noise: torch.Tensor                         # [N_chunks, D] (unit-norm prior noise)
    z_continuous: torch.Tensor                    # [N_chunks, D] (unquantized encoder latent)
    layer_latents: Tuple[torch.Tensor, ...]       # length L, each [N_chunks, D]
    layer_mean_norms: Tuple[float, ...]           # mean L2 norm ||c_l||_2 per chunk
    codebook_indices: torch.Tensor                # [G, N_chunks, L]

    @property
    def num_layers(self) -> int:
        return len(self.layer_latents)

    def cumulative_latent(self, num_active_layers: int) -> torch.Tensor:
        """Return sum_{l=0}^{num_active_layers - 1} c_l, or z_origin when 0."""
        if not (0 <= num_active_layers <= self.num_layers):
            raise ValueError(f"num_active_layers must be in 0..{self.num_layers}, got {num_active_layers}")
        if num_active_layers == 0:
            return self.z_origin
        acc = self.layer_latents[0]
        for l_idx in range(1, num_active_layers):
            acc = acc + self.layer_latents[l_idx]
        return acc

    def reverse_cumulative_latent(self, count_from_fine: int) -> torch.Tensor:
        """Return sum from finest layer L-1 backwards (e.g., c_4 -> c_4+c_3 -> ...)."""
        if not (0 <= count_from_fine <= self.num_layers):
            raise ValueError(f"count_from_fine must be in 0..{self.num_layers}, got {count_from_fine}")
        if count_from_fine == 0:
            return self.z_origin
        start_idx = self.num_layers - count_from_fine
        acc = self.layer_latents[start_idx]
        for l_idx in range(start_idx + 1, self.num_layers):
            acc = acc + self.layer_latents[l_idx]
        return acc


class LatentDecodeService(ABC):
    """Abstract interface for decomposing MIDI into RVQ layers and decoding latents via CNN."""

    @abstractmethod
    def decompose(self, sample: MidiSampleContext, noise_seed: int = 42) -> CodebookDecomposition:
        ...

    @abstractmethod
    def decode_batch(
        self,
        latent_batch: Sequence[torch.Tensor],
        sample: MidiSampleContext,
        z_reference: torch.Tensor,
        mask_inactive_slots: bool = False,
    ) -> list[tuple[np.ndarray, Tuple[QuantizedNote, ...], StageMetrics]]:
        """Batch-decode multiple frame latents [N_chunks, D] in a single GPU pass."""
        ...


class TorchMidiVisionDecodeService(LatentDecodeService):
    """Concrete PyTorch/MPS implementation backed by MidiVision (ResNetEncoder + GroupedRVQ + ResNetDecoder)."""

    def __init__(self, model_config_path: Path, model_ckpt_path: Path, device: torch.device | None = None) -> None:
        if not model_config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        if not model_ckpt_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")

        self._device = device or self._auto_device()
        self._args = MORTM5Args(str(model_config_path))
        self._model = MidiVision(self._args, sync_codebook=False).to(self._device).eval()

        raw_sd = torch.load(str(model_ckpt_path), map_location="cpu", weights_only=False)
        state_dict = raw_sd.get("model_state_dict", raw_sd) if isinstance(raw_sd, dict) else raw_sd
        self._model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_layers(self) -> int:
        return int(self._model.num_quantizers)

    @property
    def vq_groups(self) -> int:
        return int(self._model.vq_groups)

    @property
    def codebook_size(self) -> int:
        return int(self._model.codebook_size)

    @torch.inference_mode()
    def decompose(self, sample: MidiSampleContext, noise_seed: int = 42) -> CodebookDecomposition:
        x = torch.from_numpy(sample.gt_chunks).to(device=self._device, dtype=torch.float32)
        x_silent = torch.zeros_like(x)

        z_silent = self._model._encode_latent(x_silent)
        z_continuous = self._model._encode_latent(x)
        z_origin = torch.zeros_like(z_continuous)

        # Deterministic sphere-normalized prior noise matching pre_vq_norm geometry
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(noise_seed))
        raw_gauss = torch.randn(
            (x.size(0), self._model.vq_groups, self._model.codebook_dim),
            generator=gen,
            dtype=torch.float32,
        ).to(self._device)
        z_noise = self._model.pre_vq_norm(raw_gauss).reshape_as(z_continuous) * 0.72

        _, all_indices, _, all_codes = self._model.rvq(z_continuous, return_all_codes=True)
        layer_latents = []
        layer_norms = []
        for l_idx in range(self._model.num_quantizers):
            group_vecs = [
                rvq.project_out(codes[l_idx])
                for rvq, codes in zip(self._model.rvq.rvqs, all_codes)
            ]
            c_l = torch.cat(group_vecs, dim=-1)
            layer_latents.append(c_l)
            mean_norm = float(torch.norm(c_l, dim=-1).mean().item())
            layer_norms.append(mean_norm)

        return CodebookDecomposition(
            z_silent=z_silent,
            z_origin=z_origin,
            z_noise=z_noise,
            z_continuous=z_continuous,
            layer_latents=tuple(layer_latents),
            layer_mean_norms=tuple(layer_norms),
            codebook_indices=all_indices,
        )

    @torch.inference_mode()
    def decode_batch(
        self,
        latent_batch: Sequence[torch.Tensor],
        sample: MidiSampleContext,
        z_reference: torch.Tensor,
        mask_inactive_slots: bool = False,
    ) -> list[tuple[np.ndarray, Tuple[QuantizedNote, ...], StageMetrics]]:
        if not latent_batch:
            return []

        b_frames = len(latent_batch)
        n_chunks = latent_batch[0].size(0)
        stacked_z = torch.stack(list(latent_batch), dim=0)  # [B_frames, N_chunks, D]
        # ResNetDecoder natively accepts 3D [B_frames, N_chunks, D] -> [B_frames, N_chunks, 8, 128, 24]
        logits_5d = self._model.decoder(stacked_z)
        probs_5d = torch.sigmoid(logits_5d)

        if mask_inactive_slots:
            for s_spec in sample.slot_specs:
                if not s_spec.is_active:
                    probs_5d[:, :, s_spec.slot_index * 2 : s_spec.slot_index * 2 + 2] = 0.0

        gt_x = torch.from_numpy(sample.gt_chunks).to(device=self._device, dtype=torch.float32)
        ref_norm = float(torch.norm(z_reference).item()) + 1e-8
        rel_errors = (
            torch.norm(z_reference.unsqueeze(0) - stacked_z, dim=(-2, -1)) / ref_norm
        ).cpu().numpy()

        probs_np = probs_5d.cpu().numpy()
        results: list[tuple[np.ndarray, Tuple[QuantizedNote, ...], StageMetrics]] = []

        for f_idx in range(b_frames):
            frame_probs_chunks = probs_np[f_idx]  # [N_chunks, 8, 128, 24]
            probs_flat = np.ascontiguousarray(
                frame_probs_chunks.transpose(1, 0, 3, 2).reshape(
                    TOTAL_CHANNELS, n_chunks * CHUNK_STEPS, 128
                ),
                dtype=np.float32,
            )
            raw_notes = _flat_to_quantized_notes(probs_flat)
            notes = tuple(
                QuantizedNote(slot=s, start_step=st, end_step=ed, pitch=p, velocity=v)
                for s, st, ed, p, v in raw_notes
            )
            metrics = self._compute_metrics(
                pred_chunks=probs_5d[f_idx],
                gt_chunks=gt_x,
                rel_q_err=float(rel_errors[f_idx]),
                note_count=len(notes),
            )
            results.append((probs_flat, notes, metrics))

        return results

    @staticmethod
    def _compute_metrics(
        pred_chunks: torch.Tensor,
        gt_chunks: torch.Tensor,
        rel_q_err: float,
        note_count: int,
    ) -> StageMetrics:
        pred_active = (pred_chunks[:, 1::2] >= 0.5).float()
        targ_active = gt_chunks[:, 1::2]
        tp = float((pred_active * targ_active).sum().item())
        fp = float((pred_active * (1.0 - targ_active)).sum().item())
        fn = float(((1.0 - pred_active) * targ_active).sum().item())
        note_prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        note_rec = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        note_f1 = (2.0 * note_prec * note_rec / (note_prec + note_rec + 1e-8)) if (note_prec + note_rec) > 0 else 0.0

        pred_vel = pred_chunks[:, 0::2]
        targ_vel = gt_chunks[:, 0::2]
        pred_onset = (pred_vel >= 0.5).float()
        targ_onset = (targ_vel >= 0.5).float()
        otp = float((pred_onset * targ_onset).sum().item())
        ofp = float((pred_onset * (1.0 - targ_onset)).sum().item())
        ofn = float(((1.0 - pred_onset) * targ_onset).sum().item())
        onset_prec = otp / (otp + ofp + 1e-8) if (otp + ofp) > 0 else 0.0
        onset_rec = otp / (otp + ofn + 1e-8) if (otp + ofn) > 0 else 0.0
        onset_f1 = (
            2.0 * onset_prec * onset_rec / (onset_prec + onset_rec + 1e-8)
            if (onset_prec + onset_rec) > 0
            else 0.0
        )

        onset_sum = float(targ_onset.sum().item())
        vel_mae = (
            float(((pred_vel.clamp(0.5, 1.0) - targ_vel).abs() * 2.0 * targ_onset).sum().item())
            / (onset_sum + 1e-5)
            if onset_sum > 0
            else 0.0
        )

        slot_metrics = []
        for s_idx in range(4):
            pa = pred_active[:, s_idx]
            ta = targ_active[:, s_idx]
            stp = float((pa * ta).sum().item())
            sfp = float((pa * (1.0 - ta)).sum().item())
            sfn = float(((1.0 - pa) * ta).sum().item())
            sf1 = (2.0 * stp) / (2.0 * stp + sfp + sfn + 1e-8) if (2.0 * stp + sfp + sfn) > 0 else 0.0

            po = pred_onset[:, s_idx]
            to = targ_onset[:, s_idx]
            sotp = float((po * to).sum().item())
            sofp = float((po * (1.0 - to)).sum().item())
            sofn = float(((1.0 - po) * to).sum().item())
            sof1 = (2.0 * sotp) / (2.0 * sotp + sofp + sofn + 1e-8) if (2.0 * sotp + sofp + sofn) > 0 else 0.0

            so_sum = float(to.sum().item())
            svmae = (
                float(((pred_vel[:, s_idx].clamp(0.5, 1.0) - targ_vel[:, s_idx]).abs() * 2.0 * to).sum().item())
                / (so_sum + 1e-5)
                if so_sum > 0
                else 0.0
            )
            slot_metrics.append(SlotMetrics(slot_index=s_idx, note_f1=sf1, onset_f1=sof1, velocity_mae=svmae))

        return StageMetrics(
            note_f1=note_f1,
            note_precision=note_prec,
            note_recall=note_rec,
            onset_f1=onset_f1,
            onset_precision=onset_prec,
            onset_recall=onset_rec,
            velocity_mae=vel_mae,
            relative_quant_error=rel_q_err,
            decoded_note_count=note_count,
            slot_metrics=tuple(slot_metrics),
        )
