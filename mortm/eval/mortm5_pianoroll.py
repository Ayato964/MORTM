"""
MORTM5 Unified Piano-Roll Teacher-Forced Autoregressive Benchmark (`mortm.eval.mortm5_pianoroll`).

Evaluates MORTM5 8-Way Ablation downstream autoregressive models in physical decoded
piano-roll space [N_chunks, 8, 128, 24] via MidiVision (M3 / M4), eliminating cross-vocabulary
Perplexity (PPL) comparability bias across Flat vs Grouped and Shared vs Separated models.

CLI usage:
  python -m mortm.eval.mortm5_pianoroll --eval_json data/gmd/eval.json --max_songs 150
"""

from __future__ import annotations

import abc
import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

import numpy as np
import torch
from torch import Tensor

from mortm.models.modules.attention import apply_rotary_pos_emb
from mortm.models.modules.config import MORTM5Args, NUM_SLOT_TOKENS, RVQ_TOKEN_OFFSET
from mortm.models.modules.layers import DecodeCodebookHead
from mortm.models.mortm5 import (
    AblationTransformer1,
    AblationTransformer2,
    MidiVision,
    _BaseMORTM5Transformer,
    build_mortm5_ablation_model,
)
from mortm.utils.convert_pianoroll import (
    extract_quantized_notes_and_slots,
    rasterize_notes_to_chunks_u8,
)
from mortm.utils.mortm5_infer import (
    TOKENIZER_CONFIGS,
    auto_find_checkpoint,
    select_device,
)


# ============================================================================
# 1. Domain Value Objects & Accumulators (SRP / Immutability)
# ============================================================================


@dataclass(frozen=True)
class WindowSlice:
    """Input chunk window [chunk_start, chunk_end) and evaluated suffix [eval_rel_start, chunk_end - chunk_start)."""

    chunk_start: int
    chunk_end: int
    eval_rel_start: int

    @property
    def num_window_chunks(self) -> int:
        return self.chunk_end - self.chunk_start

    @property
    def eval_global_start(self) -> int:
        return self.chunk_start + self.eval_rel_start


@dataclass(frozen=True)
class PianoRollMetrics:
    """Immutable piano-roll evaluation snapshot aligned with MidiVision §8.1."""

    num_chunks: int
    note_f1: float
    note_recall: float
    note_precision: float
    onset_f1: float
    onset_recall: float
    onset_precision: float
    vel_mae: float
    token_acc: float
    layer0_token_acc: float

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "num_chunks": self.num_chunks,
            "note_f1": round(self.note_f1 * 100.0, 2),
            "note_recall": round(self.note_recall * 100.0, 2),
            "note_precision": round(self.note_precision * 100.0, 2),
            "onset_f1": round(self.onset_f1 * 100.0, 2),
            "onset_recall": round(self.onset_recall * 100.0, 2),
            "onset_precision": round(self.onset_precision * 100.0, 2),
            "vel_mae": round(self.vel_mae, 4),
            "token_acc": round(self.token_acc * 100.0, 2),
            "layer0_token_acc": round(self.layer0_token_acc * 100.0, 2),
        }


class ConfusionAccumulator:
    """O(1)-memory micro-averaged accumulator for Sustain Note F1, Onset F1, Velocity MAE, and RVQ token accuracy."""

    def __init__(self) -> None:
        self._num_chunks: int = 0
        self._note_tp: float = 0.0
        self._note_fp: float = 0.0
        self._note_fn: float = 0.0
        self._onset_tp: float = 0.0
        self._onset_fp: float = 0.0
        self._onset_fn: float = 0.0
        self._vel_abs_err_sum: float = 0.0
        self._onset_gt_sum: float = 0.0
        self._tok_correct: int = 0
        self._tok_total: int = 0
        self._l0_correct: int = 0
        self._l0_total: int = 0

    def update(
        self,
        decoded_logits: Tensor,
        target_chunks: Tensor,
        pred_rvq_3d: Optional[Tensor] = None,
        gt_rvq_3d: Optional[Tensor] = None,
    ) -> None:
        n = int(target_chunks.size(0))
        if n == 0:
            return
        self._num_chunks += n

        # 1. Note-Active (odd channels: 1, 3, 5, 7)
        prob_active = torch.sigmoid(decoded_logits[:, 1::2].float())
        target_active = target_chunks[:, 1::2].float()
        pred_active_bin = (prob_active >= 0.5).float()

        self._note_tp += float((pred_active_bin * target_active).sum().item())
        self._note_fp += float((pred_active_bin * (1.0 - target_active)).sum().item())
        self._note_fn += float(((1.0 - pred_active_bin) * target_active).sum().item())

        # 2. Onset + Velocity (even channels: 0, 2, 4, 6; active onset cells >= 0.5)
        pred_vel = torch.sigmoid(decoded_logits[:, 0::2].float())
        target_vel = target_chunks[:, 0::2].float()
        target_onset = (target_vel >= 0.5).float()
        pred_onset_bin = (pred_vel >= 0.5).float()

        self._onset_tp += float((pred_onset_bin * target_onset).sum().item())
        self._onset_fp += float((pred_onset_bin * (1.0 - target_onset)).sum().item())
        self._onset_fn += float(((1.0 - pred_onset_bin) * target_onset).sum().item())

        onset_cnt = float(target_onset.sum().item())
        if onset_cnt > 0.0:
            vel_err = ((pred_vel.clamp(0.5, 1.0) - target_vel).abs() * 2.0 * target_onset).sum().item()
            self._vel_abs_err_sum += float(vel_err)
            self._onset_gt_sum += onset_cnt

        # 3. Discrete RVQ Token Accuracy (optional)
        if pred_rvq_3d is not None and gt_rvq_3d is not None:
            eq = pred_rvq_3d == gt_rvq_3d
            self._tok_correct += int(eq.sum().item())
            self._tok_total += int(eq.numel())
            eq_l0 = eq[:, 0, :]
            self._l0_correct += int(eq_l0.sum().item())
            self._l0_total += int(eq_l0.numel())

    def snapshot(self) -> PianoRollMetrics:
        n_rec = self._note_tp / (self._note_tp + self._note_fn) if (self._note_tp + self._note_fn) > 0 else 0.0
        n_pre = self._note_tp / (self._note_tp + self._note_fp) if (self._note_tp + self._note_fp) > 0 else 0.0
        n_f1 = (2.0 * n_pre * n_rec / (n_pre + n_rec)) if (n_pre + n_rec) > 0 else 0.0

        o_rec = self._onset_tp / (self._onset_tp + self._onset_fn) if (self._onset_tp + self._onset_fn) > 0 else 0.0
        o_pre = self._onset_tp / (self._onset_tp + self._onset_fp) if (self._onset_tp + self._onset_fp) > 0 else 0.0
        o_f1 = (2.0 * o_pre * o_rec / (o_pre + o_rec)) if (o_pre + o_rec) > 0 else 0.0

        vel_mae = (self._vel_abs_err_sum / self._onset_gt_sum) if self._onset_gt_sum > 0 else 0.0
        tok_acc = (self._tok_correct / self._tok_total) if self._tok_total > 0 else 0.0
        l0_acc = (self._l0_correct / self._l0_total) if self._l0_total > 0 else 0.0

        return PianoRollMetrics(
            num_chunks=self._num_chunks,
            note_f1=n_f1,
            note_recall=n_rec,
            note_precision=n_pre,
            onset_f1=o_f1,
            onset_recall=o_rec,
            onset_precision=o_pre,
            vel_mae=vel_mae,
            token_acc=tok_acc,
            layer0_token_acc=l0_acc,
        )


class StratifiedPianoRollStats:
    """Aggregates overall, sparse (<=2 active tracks), and dense (3..4 active tracks) metrics."""

    def __init__(self) -> None:
        self._overall = ConfusionAccumulator()
        self._sparse = ConfusionAccumulator()
        self._dense = ConfusionAccumulator()

    def update_song(
        self,
        decoded_logits: Tensor,
        target_chunks: Tensor,
        pred_rvq_3d: Optional[Tensor] = None,
        gt_rvq_3d: Optional[Tensor] = None,
    ) -> None:
        self._overall.update(decoded_logits, target_chunks, pred_rvq_3d, gt_rvq_3d)

        n = target_chunks.size(0)
        active_trks = (target_chunks.view(n, 4, 2, 128, 24) > 0.0).any(dim=(2, 3, 4)).sum(dim=1)
        sparse_mask = active_trks <= 2
        dense_mask = active_trks >= 3

        if bool(sparse_mask.any()):
            p_sp = pred_rvq_3d[sparse_mask] if pred_rvq_3d is not None else None
            g_sp = gt_rvq_3d[sparse_mask] if gt_rvq_3d is not None else None
            self._sparse.update(decoded_logits[sparse_mask], target_chunks[sparse_mask], p_sp, g_sp)

        if bool(dense_mask.any()):
            p_dn = pred_rvq_3d[dense_mask] if pred_rvq_3d is not None else None
            g_dn = gt_rvq_3d[dense_mask] if gt_rvq_3d is not None else None
            self._dense.update(decoded_logits[dense_mask], target_chunks[dense_mask], p_dn, g_dn)

    def summarize(self) -> Dict[str, PianoRollMetrics]:
        return {
            "overall": self._overall.snapshot(),
            "sparse_1_2_trk": self._sparse.snapshot(),
            "dense_3_4_trk": self._dense.snapshot(),
        }


# ============================================================================
# 2. Sliding Context Window Planner & Compact Song Representation
# ============================================================================


@dataclass(frozen=True)
class CompactValidationSong:
    """Sparse ~14KB per-song note representation that rasterizes to float32 chunks on demand."""

    song_idx: int
    midi_path: str
    num_chunks: int
    notes_arr: np.ndarray
    slot_codes: Tuple[int, int, int, int]

    def rasterize_float32(self) -> np.ndarray:
        chunks_u8 = rasterize_notes_to_chunks_u8(self.num_chunks, self.notes_arr)
        return chunks_u8.astype(np.float32) / np.float32(254.0)


class SlidingWindowPlanner:
    """Plans overlapping context windows so every chunk in a long piece is evaluated exactly once."""

    def __init__(self, max_chunks: int) -> None:
        if max_chunks <= 0:
            raise ValueError(f"max_chunks must be positive, got {max_chunks}")
        self._max_chunks = int(max_chunks)
        self._stride = max(1, self._max_chunks // 2)

    @property
    def max_chunks(self) -> int:
        return self._max_chunks

    def plan(self, total_chunks: int) -> List[WindowSlice]:
        if total_chunks <= 0:
            return []
        if total_chunks <= self._max_chunks:
            return [WindowSlice(chunk_start=0, chunk_end=total_chunks, eval_rel_start=0)]

        slices: List[WindowSlice] = []
        evaluated_until = 0
        while evaluated_until < total_chunks:
            if evaluated_until == 0:
                c_start = 0
                c_end = min(total_chunks, self._max_chunks)
                slices.append(WindowSlice(chunk_start=c_start, chunk_end=c_end, eval_rel_start=0))
                evaluated_until = c_end
            else:
                c_end = min(total_chunks, evaluated_until + self._stride)
                c_start = max(0, c_end - self._max_chunks)
                eval_rel = evaluated_until - c_start
                slices.append(WindowSlice(chunk_start=c_start, chunk_end=c_end, eval_rel_start=eval_rel))
                evaluated_until = c_end
        return slices


class ValidationSongRepository:
    """Loads compact unseen validation songs from GigaMIDI `eval.json`."""

    def __init__(self, eval_json_path: str, max_songs: int = 150) -> None:
        self._eval_json_path = eval_json_path
        self._max_songs = max_songs

    def load_compact_songs(self) -> List[CompactValidationSong]:
        with open(self._eval_json_path, "r", encoding="utf-8") as f:
            paths: List[str] = json.load(f)

        songs: List[CompactValidationSong] = []
        for raw_path in paths:
            local_p = raw_path if os.path.exists(raw_path) else raw_path.replace("/data/GMD/", "/data/gmd/")
            if not os.path.exists(local_p):
                continue
            try:
                num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(local_p)
            except Exception:
                continue
            if num_chunks < 1:
                continue
            sc_tuple = (int(slot_codes[0]), int(slot_codes[1]), int(slot_codes[2]), int(slot_codes[3]))
            songs.append(
                CompactValidationSong(
                    song_idx=len(songs),
                    midi_path=local_p,
                    num_chunks=int(num_chunks),
                    notes_arr=notes_arr,
                    slot_codes=sc_tuple,
                )
            )
            if 0 < self._max_songs <= len(songs):
                break
        return songs


# ============================================================================
# 3. Strategy Pattern for Teacher-Forced Rollout (OCP / LSP / DIP)
# ============================================================================


class RolloutStrategy(abc.ABC):
    """Abstract strategy computing both Step-level TF and Chunk-level TF via Shared-Prefix Branched Attention."""

    def __init__(self, model: _BaseMORTM5Transformer) -> None:
        self._model = model
        self._prefix_units = 1 + (NUM_SLOT_TOKENS if model.use_instrument_tokens else 0)
        self._vq_layers = int(model.vq_layers)
        self._vq_groups = int(model.vq_groups)
        self._codebook_size = int(model.codebook_size)

    @property
    @abc.abstractmethod
    def steps_per_chunk(self) -> int:
        ...

    @property
    def max_window_chunks(self) -> int:
        reserved = 2 + (NUM_SLOT_TOKENS if self._model.use_instrument_tokens else 0)
        usable = max(self.steps_per_chunk, int(self._model.args.position_length) - reserved)
        return max(1, usable // self.steps_per_chunk)

    @abc.abstractmethod
    def build_window_input(
        self,
        raw_rvq_window: Tensor,
        slot_codes: Sequence[int],
        vq_token_offset: int,
    ) -> Tensor:
        ...

    @abc.abstractmethod
    def _extract_prefill_predictions(self, logits_all: Tensor, num_chunks: int) -> Tuple[Tensor, Tensor]:
        ...

    @abc.abstractmethod
    def _embed_step_tokens(self, chunk_tf_tokens: Tensor, step_u: int) -> Tensor:
        ...

    @abc.abstractmethod
    def _project_and_update_step(self, hidden_out: Tensor, chunk_tf_tokens: Tensor, step_u: int) -> None:
        ...

    def _run_branched_intra_chunk_decoder(
        self,
        chunk_tf_tokens: Tensor,
        eval_rel_start: int,
        total_window_chunks: int,
    ) -> None:
        U = self.steps_per_chunk
        C_eval = int(chunk_tf_tokens.size(0))
        if U <= 1 or C_eval <= 0:
            return

        device = chunk_tf_tokens.device
        P = self._prefix_units

        decoder_layers = self._model.decoder.layers
        num_layers = len(decoder_layers)
        n_heads = int(self._model.args.num_heads)
        head_dim = int(self._model.args.d_model) // n_heads
        scale = head_dim ** -0.5

        T_p = P + (total_window_chunks - 1) * U
        c_indices = torch.arange(eval_rel_start, total_window_chunks, device=device, dtype=torch.long)
        base_c_pos = P + c_indices * U
        c_limits = base_c_pos.unsqueeze(1)
        t_indices = torch.arange(T_p, device=device, dtype=torch.long).unsqueeze(0)
        gt_invalid_mask = (t_indices >= c_limits).unsqueeze(0)  # [1, C_eval, T_p]

        intra_k = [
            torch.empty((C_eval, U - 1, n_heads, head_dim), device=device, dtype=torch.bfloat16)
            for _ in range(num_layers)
        ]
        intra_v = [
            torch.empty((C_eval, U - 1, n_heads, head_dim), device=device, dtype=torch.bfloat16)
            for _ in range(num_layers)
        ]

        for u in range(1, U):
            pos_ids = base_c_pos + (u - 1)
            max_pos = int(pos_ids[-1].item()) + 1
            y = self._embed_step_tokens(chunk_tf_tokens, step_u=u)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                for l_idx, layer in enumerate(decoder_layers):
                    attn_mod = layer.self_attention
                    h = layer.norm1(y)
                    qkv = attn_mod.qkv_block(q=h.unsqueeze(1))
                    q, k, v = qkv.unbind(2)
                    q = q.contiguous()
                    k = k.contiguous()
                    v = v.contiguous()

                    cos, sin = attn_mod.rotary_emb(v, seq_len=max_pos)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

                    intra_k[l_idx][:, u - 1] = k[:, 0]
                    intra_v[l_idx][:, u - 1] = v[:, 0]

                    k_gt = attn_mod.kv_cache[0][0, :T_p]
                    v_gt = attn_mod.kv_cache[1][0, :T_p]
                    q_h = q[:, 0].permute(1, 0, 2)  # [H, C_eval, D]

                    scores_gt = torch.matmul(q_h, k_gt.permute(1, 2, 0)) * scale
                    scores_gt = scores_gt.masked_fill(gt_invalid_mask, float("-inf"))

                    k_in = intra_k[l_idx][:, :u].permute(2, 0, 1, 3)
                    v_in = intra_v[l_idx][:, :u].permute(2, 0, 1, 3)
                    scores_in = (q_h.unsqueeze(2) * k_in).sum(dim=-1) * scale

                    scores_all = torch.cat([scores_gt, scores_in], dim=-1)
                    probs = torch.softmax(scores_all.float(), dim=-1).to(q.dtype)

                    out_gt = torch.matmul(probs[:, :, :T_p], v_gt.permute(1, 0, 2))
                    out_in = (probs[:, :, T_p:].unsqueeze(-1) * v_in).sum(dim=2)
                    attn_out = (out_gt + out_in).permute(1, 0, 2).reshape(C_eval, -1)
                    attn_out = attn_mod.qkv_block.comp(attn_out)

                    y = y + attn_out
                    y = y + layer.ffn(layer.norm3(y))

                hidden_out = self._model.decoder.norm(y)

            self._project_and_update_step(hidden_out, chunk_tf_tokens, step_u=u)

    @torch.inference_mode()
    def predict_window(
        self,
        raw_rvq_window: Tensor,
        slot_codes: Sequence[int],
        vq_token_offset: int = RVQ_TOKEN_OFFSET,
        eval_rel_start: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        C = int(raw_rvq_window.size(0))
        C_eval = C - int(eval_rel_start)
        L, G = self._vq_layers, self._vq_groups
        gt_seq = self.build_window_input(raw_rvq_window, slot_codes, vq_token_offset=vq_token_offset)

        self._model.eval()
        self._model.reset_kv_cache()
        try:
            logits_all = self._model.forward(gt_seq, is_causal=True, is_save_cache=True)
            step_tf_all, chunk_tf_all = self._extract_prefill_predictions(logits_all, num_chunks=C)
            step_tf_eval = step_tf_all[eval_rel_start:].contiguous()
            chunk_tf_eval = chunk_tf_all[eval_rel_start:].contiguous()
            self._run_branched_intra_chunk_decoder(
                chunk_tf_eval,
                eval_rel_start=int(eval_rel_start),
                total_window_chunks=C,
            )
        finally:
            self._model.reset_kv_cache()

        step_rvq_flat = self._model.extract_rvq_tokens(step_tf_eval.reshape(-1), target_token_offset=RVQ_TOKEN_OFFSET)
        chunk_rvq_flat = self._model.extract_rvq_tokens(chunk_tf_eval.reshape(-1), target_token_offset=RVQ_TOKEN_OFFSET)
        return step_rvq_flat.view(C_eval, L, G), chunk_rvq_flat.view(C_eval, L, G)


class FlatRolloutStrategy(RolloutStrategy):
    """Strategy for Flat autoregressive models (`AblationTransformer1`: 1 chunk = L * G steps)."""

    def __init__(self, model: AblationTransformer1) -> None:
        super().__init__(model)
        self._flat_model = model

    @property
    def steps_per_chunk(self) -> int:
        return self._vq_layers * self._vq_groups

    def build_window_input(
        self,
        raw_rvq_window: Tensor,
        slot_codes: Sequence[int],
        vq_token_offset: int,
    ) -> Tensor:
        device = raw_rvq_window.device
        C = raw_rvq_window.size(0)
        L, G, K = self._vq_layers, self._vq_groups, self._codebook_size
        rvq_3d = raw_rvq_window.view(C, L, G).to(dtype=torch.long) + (self._flat_model.token_offset - vq_token_offset)
        if self._flat_model.separate_rvq_layers:
            layer_offsets = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1) * (G * K)
            rvq_3d = rvq_3d + layer_offsets
        prefix = self._flat_model.build_prefix_tokens(slot_codes=slot_codes, batch_size=1, device=device)
        return torch.cat([prefix, rvq_3d.reshape(1, C * L * G)], dim=1)

    def _extract_prefill_predictions(self, logits_all: Tensor, num_chunks: int) -> Tuple[Tensor, Tensor]:
        P = self._prefix_units
        U = self.steps_per_chunk
        G, K = self._vq_groups, self._codebook_size
        device = logits_all.device

        rvq_logits = logits_all[0, P - 1 : P - 1 + num_chunks * U, :]
        step_tf = torch.empty((num_chunks, U), dtype=torch.long, device=device)
        for u in range(U):
            layer_idx = u // G
            group_idx = u % G
            eff_layer = layer_idx if self._flat_model.separate_rvq_layers else 0
            low = self._flat_model.token_offset + (eff_layer * G + group_idx) * K
            high = low + K
            step_tf[:, u] = low + torch.argmax(rvq_logits[u::U, low:high], dim=-1)

        chunk_tf = torch.empty_like(step_tf)
        chunk_tf[:, 0] = step_tf[:, 0]
        return step_tf, chunk_tf

    def _embed_step_tokens(self, chunk_tf_tokens: Tensor, step_u: int) -> Tensor:
        prev_tok = chunk_tf_tokens[:, step_u - 1]
        return self._flat_model.embedding(prev_tok).to(dtype=torch.bfloat16)

    def _project_and_update_step(self, hidden_out: Tensor, chunk_tf_tokens: Tensor, step_u: int) -> None:
        G, K = self._vq_groups, self._codebook_size
        step_logits = self._flat_model.Wout(hidden_out.to(self._flat_model.Wout.weight.dtype)).float()
        layer_idx = step_u // G
        group_idx = step_u % G
        eff_layer = layer_idx if self._flat_model.separate_rvq_layers else 0
        low = self._flat_model.token_offset + (eff_layer * G + group_idx) * K
        high = low + K
        chunk_tf_tokens[:, step_u] = low + torch.argmax(step_logits[:, low:high], dim=-1)


class GroupedRolloutStrategy(RolloutStrategy):
    """Strategy for Grouped Soft-Token models (`AblationTransformer2`: 1 chunk = L steps)."""

    def __init__(self, model: AblationTransformer2) -> None:
        super().__init__(model)
        self._grp_model = model

    @property
    def steps_per_chunk(self) -> int:
        return self._vq_layers

    def build_window_input(
        self,
        raw_rvq_window: Tensor,
        slot_codes: Sequence[int],
        vq_token_offset: int,
    ) -> Tensor:
        device = raw_rvq_window.device
        C = raw_rvq_window.size(0)
        L, G, K = self._vq_layers, self._vq_groups, self._codebook_size
        rvq_3d = raw_rvq_window.view(C, L, G).to(dtype=torch.long) + (self._grp_model.token_offset - vq_token_offset)
        if self._grp_model.separate_rvq_layers:
            layer_offsets = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1) * (G * K)
            rvq_3d = rvq_3d + layer_offsets
        prefix = self._grp_model.build_prefix_tokens(slot_codes=slot_codes, batch_size=1, device=device)
        return torch.cat([prefix, rvq_3d.reshape(1, C * L, G)], dim=1)

    def _extract_prefill_predictions(self, logits_all: Tensor, num_chunks: int) -> Tuple[Tensor, Tensor]:
        P = self._prefix_units
        U = self.steps_per_chunk
        G, K = self._vq_groups, self._codebook_size
        device = logits_all.device

        rvq_logits = logits_all[0, P - 1 : P - 1 + num_chunks * U, :, :]
        low = self._grp_model.token_offset
        high = low + K
        local_sampled = low + torch.argmax(rvq_logits[:, :, low:high], dim=-1)

        step_tf = torch.empty((num_chunks, U, G), dtype=torch.long, device=device)
        for u in range(U):
            step_tf[:, u, :] = self._grp_model.local_to_global_ids(local_sampled[u::U, :], layer_idx=u)

        chunk_tf = torch.empty_like(step_tf)
        chunk_tf[:, 0, :] = step_tf[:, 0, :]
        return step_tf, chunk_tf

    def _embed_step_tokens(self, chunk_tf_tokens: Tensor, step_u: int) -> Tensor:
        prev_tok = chunk_tf_tokens[:, step_u - 1, :]
        return self._grp_model._aggregate_group_embeddings(prev_tok).to(dtype=torch.bfloat16)

    def _project_and_update_step(self, hidden_out: Tensor, chunk_tf_tokens: Tensor, step_u: int) -> None:
        P = self._prefix_units
        G, K = self._vq_groups, self._codebook_size
        C = hidden_out.size(0)
        wout = self._grp_model.Wout
        if isinstance(wout, DecodeCodebookHead):
            step_logits = wout(hidden_out.to(wout.weight.dtype), start_pos=(P - 1 + step_u)).float()
        else:
            step_logits = wout(hidden_out.to(wout.weight.dtype)).float().reshape(C, G, self._grp_model.group_vocab_size)
        low = self._grp_model.token_offset
        high = low + K
        local_sampled = low + torch.argmax(step_logits[:, :, low:high], dim=-1)
        chunk_tf_tokens[:, step_u, :] = self._grp_model.local_to_global_ids(local_sampled, layer_idx=step_u)


class RolloutStrategyFactory:
    """Factory creating the matching `RolloutStrategy` for any MORTM5 downstream Transformer."""

    @staticmethod
    def create(model: _BaseMORTM5Transformer) -> RolloutStrategy:
        if isinstance(model, AblationTransformer1):
            return FlatRolloutStrategy(model)
        if isinstance(model, AblationTransformer2):
            return GroupedRolloutStrategy(model)
        raise TypeError(f"Unsupported MORTM5 Transformer type: {type(model).__name__}")


# ============================================================================
# 4. Batched MidiVision Codec Service & Benchmark Runner
# ============================================================================


class MidiVisionCodecService:
    """Mini-batched MidiVision encode/decode service keeping GPU/MPS memory bounded."""

    def __init__(self, vq_model: MidiVision, device: torch.device, chunk_batch_size: int = 256) -> None:
        self._vq = vq_model
        self._device = device
        self._bs = max(1, int(chunk_batch_size))

    @property
    def token_offset(self) -> int:
        return int(getattr(self._vq, "token_offset", RVQ_TOKEN_OFFSET))

    @property
    def num_quantizers(self) -> int:
        return int(self._vq.num_quantizers)

    @property
    def vq_groups(self) -> int:
        return int(self._vq.vq_groups)

    @torch.inference_mode()
    def encode_chunks(self, chunks_t: Tensor) -> Tensor:
        outs: List[Tensor] = []
        n = chunks_t.size(0)
        for s in range(0, n, self._bs):
            outs.append(self._vq.encode_tokens(chunks_t[s : s + self._bs], include_offset=True))
        return torch.cat(outs, dim=0)

    @torch.inference_mode()
    def decode_rvq_3d(self, rvq_3d: Tensor) -> Tensor:
        outs: List[Tensor] = []
        n = rvq_3d.size(0)
        for s in range(0, n, self._bs):
            outs.append(self._vq.decode_tokens(rvq_3d[s : s + self._bs], token_offset=RVQ_TOKEN_OFFSET))
        return torch.cat(outs, dim=0)


@dataclass(frozen=True)
class AblationTargetSpec:
    ablation_id: str
    label: str
    ckpt_path: str
    tokenizer_key: str


class AblationPianoRollBenchmark:
    """Orchestrates Oracle upper-bound and downstream Teacher-Forced piano-roll evaluations."""

    def __init__(self, device: torch.device, vq_batch_size: int = 256) -> None:
        self._device = device
        self._vq_batch_size = vq_batch_size

    def _clear_device_cache(self) -> None:
        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_codec(self, tokenizer_key: str) -> MidiVisionCodecService:
        tok_cfg = TOKENIZER_CONFIGS[tokenizer_key]
        vq_args = MORTM5Args(tok_cfg["config"])
        vq = MidiVision(vq_args, sync_codebook=False).to(self._device).eval()
        sd = torch.load(tok_cfg["ckpt"], map_location="cpu", weights_only=False)
        vq.load_state_dict(sd.get("model_state_dict", sd), strict=True)
        return MidiVisionCodecService(vq, self._device, chunk_batch_size=self._vq_batch_size)

    def evaluate_oracle(
        self,
        tokenizer_key: str,
        songs: Sequence[CompactValidationSong],
    ) -> Tuple[Dict[str, PianoRollMetrics], List[np.ndarray]]:
        codec = self._load_codec(tokenizer_key)
        stats = StratifiedPianoRollStats()
        encoded_cache: List[np.ndarray] = []
        t0 = time.time()
        try:
            for song in songs:
                gt_chunks = torch.from_numpy(song.rasterize_float32()).to(self._device)
                raw_rvq = codec.encode_chunks(gt_chunks)
                encoded_cache.append(raw_rvq.detach().cpu().numpy().astype(np.uint16))
                gt_rvq_3d = raw_rvq.view(song.num_chunks, codec.num_quantizers, codec.vq_groups)
                dec_logits = codec.decode_rvq_3d(gt_rvq_3d)
                stats.update_song(dec_logits, gt_chunks, gt_rvq_3d, gt_rvq_3d)
        finally:
            del codec
            self._clear_device_cache()
        elapsed = time.time() - t0
        res = stats.summarize()
        ov = res["overall"]
        print(
            f"[Oracle {tokenizer_key}] {ov.num_chunks:,} chunks in {elapsed:.1f}s | "
            f"Note F1={ov.note_f1*100:.2f}% | Onset F1={ov.onset_f1*100:.2f}% | VelMAE={ov.vel_mae:.4f}",
            flush=True,
        )
        return res, encoded_cache

    def evaluate_ablation_model(
        self,
        spec: AblationTargetSpec,
        songs: Sequence[CompactValidationSong],
        encoded_rvq_per_song: Sequence[np.ndarray],
    ) -> Dict[str, Dict[str, PianoRollMetrics]]:
        codec = self._load_codec(spec.tokenizer_key)
        m_args = MORTM5Args(spec.ablation_id)
        model = build_mortm5_ablation_model(m_args).to(self._device).eval()
        raw_sd = torch.load(spec.ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(raw_sd.get("model_state_dict", raw_sd), strict=True)

        strategy = RolloutStrategyFactory.create(model)
        planner = SlidingWindowPlanner(max_chunks=strategy.max_window_chunks)

        chunk_tf_stats = StratifiedPianoRollStats()
        step_tf_stats = StratifiedPianoRollStats()

        L = int(model.vq_layers)
        G = int(model.vq_groups)
        t0 = time.time()
        try:
            for idx, (song, rvq_u16) in enumerate(zip(songs, encoded_rvq_per_song), start=1):
                gt_chunks = torch.from_numpy(song.rasterize_float32()).to(self._device)
                raw_rvq = torch.from_numpy(rvq_u16.astype(np.int64)).to(self._device)
                gt_rvq_3d = raw_rvq.view(song.num_chunks, L, G)

                pred_step_parts: List[Tensor] = []
                pred_chunk_parts: List[Tensor] = []
                for w in planner.plan(song.num_chunks):
                    win_rvq = raw_rvq[w.chunk_start : w.chunk_end]
                    s_3d, c_3d = strategy.predict_window(
                        win_rvq,
                        slot_codes=song.slot_codes,
                        vq_token_offset=codec.token_offset,
                        eval_rel_start=w.eval_rel_start,
                    )
                    pred_step_parts.append(s_3d)
                    pred_chunk_parts.append(c_3d)

                pred_step_song = torch.cat(pred_step_parts, dim=0)
                pred_chunk_song = torch.cat(pred_chunk_parts, dim=0)

                dec_chunk_logits = codec.decode_rvq_3d(pred_chunk_song)
                dec_step_logits = codec.decode_rvq_3d(pred_step_song)

                chunk_tf_stats.update_song(dec_chunk_logits, gt_chunks, pred_chunk_song, gt_rvq_3d)
                step_tf_stats.update_song(dec_step_logits, gt_chunks, pred_step_song, gt_rvq_3d)

                if idx % 25 == 0 or idx == len(songs):
                    cur_ov = chunk_tf_stats.summarize()["overall"]
                    print(
                        f"  [{spec.ablation_id} {idx:>3}/{len(songs)} songs | {cur_ov.num_chunks:>6,} ch | {time.time()-t0:>5.1f}s] "
                        f"Chunk-TF Note F1={cur_ov.note_f1*100:.2f}% | Onset F1={cur_ov.onset_f1*100:.2f}% | "
                        f"VelMAE={cur_ov.vel_mae:.4f} | L0 Acc={cur_ov.layer0_token_acc*100:.2f}%",
                        flush=True,
                    )
        finally:
            del strategy, model, codec
            self._clear_device_cache()

        return {
            "chunk_tf": chunk_tf_stats.summarize(),
            "step_tf": step_tf_stats.summarize(),
        }


# ============================================================================
# 5. Presentation Layer: Markdown Report Formatter (SRP)
# ============================================================================


class MarkdownReportFormatter:
    """Formats evaluation results into GitHub Flavored Markdown tables."""

    @staticmethod
    def format_report(
        num_songs: int,
        oracle_results: Dict[str, Dict[str, PianoRollMetrics]],
        ablation_specs: Sequence[AblationTargetSpec],
        ablation_results: Dict[str, Dict[str, Dict[str, PianoRollMetrics]]],
    ) -> str:
        ref_or = oracle_results.get("m4_g4l4") or next(iter(oracle_results.values()))
        total_chunks = ref_or["overall"].num_chunks
        sparse_chunks = ref_or["sparse_1_2_trk"].num_chunks
        dense_chunks = ref_or["dense_3_4_trk"].num_chunks

        lines: List[str] = []
        lines.append("## MORTM5 統一ピアノロール復元評価（Teacher-Forced Autoregressive Decoding）")
        lines.append("")
        lines.append(
            f"- **評価データセット**: 未見 GigaMIDI 検証セット 先頭 `{num_songs}` 曲 "
            f"（計 **`{total_chunks:,}` チャンク**：1〜2トラック疎領域 `{sparse_chunks:,}` チャンク、"
            f"3〜4トラック密領域 `{dense_chunks:,}` チャンク）"
        )
        lines.append("")
        lines.append("### 1. 【主評価・完全公平比較】チャンク単位教師強制 ＋ チャンク内自己回帰生成 (`Chunk-level TF`)")
        lines.append("")
        lines.append(
            "| モデル | 構成 (`Tokenizer / Mode / Vocab`) | Note F1 (Rec / Prec) | Onset F1 (Rec / Prec) | Velocity MAE ↓ | 1〜2 Trk Note / Onset F1 | 3〜4 Trk Note / Onset F1 | RVQ L0 / 全層一致率 |"
        )
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for tok_key, title in [("m3_g8l2", "*Oracle Upper Bound (M3)*"), ("m4_g4l4", "*Oracle Upper Bound (M4)*")]:
            if tok_key not in oracle_results:
                continue
            orr = oracle_results[tok_key]
            ov, sp, dn = orr["overall"], orr["sparse_1_2_trk"], orr["dense_3_4_trk"]
            cfg_desc = "`G8-L2` 直接再構成上限" if tok_key == "m3_g8l2" else "`G4-L4` 直接再構成上限"
            lines.append(
                f"| {title} | {cfg_desc} | "
                f"`{ov.note_f1*100:.2f}%` (`{ov.note_recall*100:.1f}% / {ov.note_precision*100:.1f}%`) | "
                f"`{ov.onset_f1*100:.2f}%` (`{ov.onset_recall*100:.1f}% / {ov.onset_precision*100:.1f}%`) | "
                f"`{ov.vel_mae:.4f}` | "
                f"`{sp.note_f1*100:.2f}% / {sp.onset_f1*100:.2f}%` | "
                f"`{dn.note_f1*100:.2f}% / {dn.onset_f1*100:.2f}%` | "
                f"`100.0% / 100.0%` |"
            )

        for spec in ablation_specs:
            r = ablation_results[spec.ablation_id]["chunk_tf"]
            ov, sp, dn = r["overall"], r["sparse_1_2_trk"], r["dense_3_4_trk"]
            lines.append(
                f"| **`{spec.ablation_id}`** | {spec.label} | "
                f"**`{ov.note_f1*100:.2f}%`** (`{ov.note_recall*100:.1f}% / {ov.note_precision*100:.1f}%`) | "
                f"**`{ov.onset_f1*100:.2f}%`** (`{ov.onset_recall*100:.1f}% / {ov.onset_precision*100:.1f}%`) | "
                f"**`{ov.vel_mae:.4f}`** | "
                f"`{sp.note_f1*100:.2f}% / {sp.onset_f1*100:.2f}%` | "
                f"`{dn.note_f1*100:.2f}% / {dn.onset_f1*100:.2f}%` | "
                f"`{ov.layer0_token_acc*100:.2f}% / {ov.token_acc*100:.2f}%` |"
            )

        lines.append("")
        lines.append("### 2. 【参考評価】ステップ単位教師強制 (`Step-level 1-Step-Ahead TF`)")
        lines.append("")
        lines.append(
            "| モデル | 構成 (`Tokenizer / Mode / Vocab`) | Note F1 (Rec / Prec) | Onset F1 (Rec / Prec) | Velocity MAE ↓ | 1〜2 Trk Note / Onset F1 | 3〜4 Trk Note / Onset F1 | RVQ L0 / 全層一致率 |"
        )
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for spec in ablation_specs:
            r = ablation_results[spec.ablation_id]["step_tf"]
            ov, sp, dn = r["overall"], r["sparse_1_2_trk"], r["dense_3_4_trk"]
            lines.append(
                f"| **`{spec.ablation_id}`** | {spec.label} | "
                f"**`{ov.note_f1*100:.2f}%`** (`{ov.note_recall*100:.1f}% / {ov.note_precision*100:.1f}%`) | "
                f"**`{ov.onset_f1*100:.2f}%`** (`{ov.onset_recall*100:.1f}% / {ov.onset_precision*100:.1f}%`) | "
                f"**`{ov.vel_mae:.4f}`** | "
                f"`{sp.note_f1*100:.2f}% / {sp.onset_f1*100:.2f}%` | "
                f"`{dn.note_f1*100:.2f}% / {dn.onset_f1*100:.2f}%` | "
                f"`{ov.layer0_token_acc*100:.2f}% / {ov.token_acc*100:.2f}%` |"
            )

        return "\n".join(lines) + "\n"


def build_available_ablation_specs(ablation_ids: Sequence[str]) -> List[AblationTargetSpec]:
    specs: List[AblationTargetSpec] = []
    for raw_id in ablation_ids:
        aid = raw_id.strip()
        if not aid.startswith("#") and aid.isdigit():
            aid = f"#{aid}"
        m_args = MORTM5Args(aid)
        ckpt = auto_find_checkpoint(aid, m_args.name)
        if not ckpt or not os.path.isfile(ckpt):
            continue
        tok_key = m_args.tokenizer_key or ("m4_g4l4" if m_args.vq_groups == 4 else "m3_g8l2")
        tok_label = "M4 (G4L4)" if tok_key == "m4_g4l4" else "M3 (G8L2)"
        mode_label = "Grouped" if m_args.model_mode == "grouped" else "Flat"
        vocab_label = "Separated" if m_args.separate_rvq_layers else "Shared"
        specs.append(
            AblationTargetSpec(
                ablation_id=aid,
                label=f"`{tok_label}` / `{mode_label}` / `{vocab_label}`",
                ckpt_path=ckpt,
                tokenizer_key=tok_key,
            )
        )
    return specs


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MORTM5 Ablation Unified Piano-Roll Teacher-Forced Benchmark")
    parser.add_argument("--eval_json", type=str, default="data/gmd/eval.json", help="Validation MIDI paths JSON")
    parser.add_argument("--max_songs", type=int, default=150, help="Number of validation songs (default: 150)")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["#6", "#8", "#4", "#7"],
        help="Ablation presets to evaluate (auto-skips missing checkpoints)",
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / mps / cpu)")
    parser.add_argument("--vq_batch_size", type=int, default=256, help="MidiVision chunk mini-batch size")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON report path")
    parser.add_argument("--out_md", type=str, default=None, help="Optional output Markdown report path")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    print(f"[Eval] Device={device} | Loading up to {args.max_songs} songs from {args.eval_json} ...", flush=True)

    repo = ValidationSongRepository(args.eval_json, max_songs=args.max_songs)
    songs = repo.load_compact_songs()
    total_chunks = sum(s.num_chunks for s in songs)
    print(f"[Eval] Loaded {len(songs)} valid songs ({total_chunks:,} total chunks).", flush=True)

    specs = build_available_ablation_specs(args.ablations)
    if not specs:
        raise FileNotFoundError(f"No trained ablation checkpoints found for {args.ablations}.")

    needed_toks = sorted({s.tokenizer_key for s in specs})
    benchmark = AblationPianoRollBenchmark(device=device, vq_batch_size=args.vq_batch_size)
    oracle_results: Dict[str, Dict[str, PianoRollMetrics]] = {}
    encoded_caches: Dict[str, List[np.ndarray]] = {}
    for tok_key in needed_toks:
        res, cache = benchmark.evaluate_oracle(tok_key, songs)
        oracle_results[tok_key] = res
        encoded_caches[tok_key] = cache

    ablation_results: Dict[str, Dict[str, Dict[str, PianoRollMetrics]]] = {}
    for spec in specs:
        print(f"\n[Eval] Evaluating {spec.ablation_id} ({spec.label}) ...", flush=True)
        ablation_results[spec.ablation_id] = benchmark.evaluate_ablation_model(
            spec, songs, encoded_caches[spec.tokenizer_key]
        )

    md_text = MarkdownReportFormatter.format_report(
        num_songs=len(songs),
        oracle_results=oracle_results,
        ablation_specs=specs,
        ablation_results=ablation_results,
    )

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "num_songs": len(songs),
            "total_chunks": total_chunks,
            "oracle": {k: {sk: sv.to_dict() for sk, sv in v.items()} for k, v in oracle_results.items()},
            "ablations": {
                ab_id: {
                    mode_k: {sk: sv.to_dict() for sk, sv in mode_v.items()}
                    for mode_k, mode_v in res.items()
                }
                for ab_id, res in ablation_results.items()
            },
        }
        out_json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_md:
        out_md_path = Path(args.out_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(md_text, encoding="utf-8")

    print("\n" + "=" * 90)
    print(md_text)
    print("=" * 90)


if __name__ == "__main__":
    main()
