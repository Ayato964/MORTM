"""SyMIc-G4L4 + Depth (`AblationTransformer2` + `MidiVision`) Adapters for Unified CARL."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mortm.models.modules.attention import pad_input, unpad_input
from mortm.models.modules.config import (
    EOS_TOKEN_ID,
    MORTM5Args,
    NUM_SLOT_TOKENS,
    PAD_TOKEN_ID,
    SLOT_CODE_EMPTY,
)
from mortm.models.modules.layers import DecodeCodebookHead, MILPool
from mortm.models.mortm5 import AblationTransformer2, MidiVision, build_mortm5_ablation_model
from mortm.train.datasets import MORTM5MemmapDataset
from mortm.utils.convert_pianoroll import (
    CHUNKS_PER_BAR,
    extract_quantized_notes_and_slots,
    rasterize_notes_to_chunks_float,
)
from mortm.utils.mortm5_infer import Mortm5PromptEncoder, TOKENIZER_CONFIGS
from mortm.utils.symbolic_tokenizers import extract_segment_canonical_notes
from ..carl_even import similarity_reward
from ..domain import CARLExample, RolloutBatch, StepLogProbsKL, UnifiedSamplingConfig
from ..interfaces import (
    CARLActorAdapter,
    CARLCriticAdapter,
    CARLDataRepository,
    CARLDataSamplePool,
    GuardrailContextBuilder,
    GuardrailRule,
    GuardrailSuite,
    InMemorySamplePool,
    LazyIndexedSamplePool,
)
from ..lora_adapter import inject_attention_lora


def _extract_state_dict(raw_ckpt: Any) -> Dict[str, Any]:
    if isinstance(raw_ckpt, dict):
        if "model_state_dict" in raw_ckpt and isinstance(raw_ckpt["model_state_dict"], dict):
            return raw_ckpt["model_state_dict"]
        if "model" in raw_ckpt and isinstance(raw_ckpt["model"], dict):
            return raw_ckpt["model"]
        return raw_ckpt
    raise TypeError(f"Unsupported checkpoint payload type: {type(raw_ckpt).__name__}")


def load_symic_model(
    config_source: str | dict | MORTM5Args,
    ckpt_path: Optional[str | Path],
    device: torch.device,
) -> AblationTransformer2:
    m_args = config_source if isinstance(config_source, MORTM5Args) else MORTM5Args(config_source)
    model = build_mortm5_ablation_model(m_args).to(device)
    if not isinstance(model, AblationTransformer2):
        raise TypeError(
            f"SyMIc CARL adapter requires grouped AblationTransformer2, got {type(model).__name__}."
        )
    if ckpt_path is not None:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(raw)
        sd = {k: v for k, v in sd.items() if k != "_carl_unmerged"}
        model.load_state_dict(sd, strict=True)
    return model


def load_midivision_codec(
    tokenizer_key: str = "m4_g4l4",
    vq_config_path: Optional[str | Path] = None,
    vq_ckpt_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> MidiVision:
    dev = device or torch.device("cpu")
    if vq_config_path is None or vq_ckpt_path is None:
        if tokenizer_key not in TOKENIZER_CONFIGS:
            raise KeyError(
                f"Unknown MidiVision tokenizer_key '{tokenizer_key}'. "
                f"Available: {sorted(TOKENIZER_CONFIGS.keys())}"
            )
        info = TOKENIZER_CONFIGS[tokenizer_key]
        vq_config_path = vq_config_path or info["config"]
        vq_ckpt_path = vq_ckpt_path or info["ckpt"]

    if not os.path.isfile(vq_ckpt_path):
        raise FileNotFoundError(f"MidiVision checkpoint not found: {vq_ckpt_path}")

    vq_args = MORTM5Args(str(vq_config_path))
    vq = MidiVision(vq_args, sync_codebook=False).to(dev).eval()
    raw = torch.load(vq_ckpt_path, map_location="cpu", weights_only=False)
    vq.load_state_dict(_extract_state_dict(raw), strict=True)
    for param in vq.parameters():
        param.requires_grad_(False)
    return vq


# ======================================================================
# 1. Data Repositories (`SyMIcMemmapRepository` & `SyMIcMidiRepository`)
# ======================================================================


class SyMIcMemmapRepository(CARLDataRepository):
    """O(1)-RAM repository backed by `MORTM5MemmapDataset` (`train_tokens.bin` + `offsets` + `slots`).

    Splits strictly at the **song (`song_idx`) level** 50:50 into `ana_pool` and `gem_pool`
    so the Odd-round Critic can never memorize exact ground-truth continuations for Even-round prompts.
    """

    def __init__(
        self,
        token_dir: str | Path,
        model_args: MORTM5Args,
        split: str = "train",
        prompt_bars: int = 4,
        total_bars: int = 16,
        min_active_tracks: int = 2,
        stride_bars: Optional[int] = None,
    ) -> None:
        if prompt_bars <= 0 or total_bars <= prompt_bars:
            raise ValueError(
                f"Expected 0 < prompt_bars < total_bars, got prompt={prompt_bars}, total={total_bars}."
            )
        self._prompt_chunks = int(prompt_bars) * CHUNKS_PER_BAR
        self._total_chunks = int(total_bars) * CHUNKS_PER_BAR
        self._min_active_tracks = max(1, int(min_active_tracks))
        eff_stride = int(stride_bars) * CHUNKS_PER_BAR if stride_bars is not None else self._total_chunks

        self._dataset = MORTM5MemmapDataset.from_split_dir(
            token_dir=str(token_dir),
            split=split,
            args=model_args,
            max_chunks=self._total_chunks,
            min_chunks=self._total_chunks,
            stride_chunks=eff_stride,
            bos_eos_mode="always",
        )
        self._vq_groups = int(model_args.vq_groups)
        self._vq_layers = int(model_args.vq_layers)
        self._prefix_len = 1 + (NUM_SLOT_TOKENS if bool(model_args.use_instrument_tokens) else 0)
        self._prompt_soft_len = self._prefix_len + self._prompt_chunks * self._vq_layers
        self._full_soft_len = self._prefix_len + self._total_chunks * self._vq_layers

    def _load_window_example(self, win_idx: int) -> CARLExample:
        flat_seq = self._dataset[win_idx].numpy()
        seq_2d = flat_seq.reshape(-1, self._vq_groups)
        # Drop trailing `<EOS>*G` if present so `[0 : full_soft_len]` is strictly `[Prefix | RVQ]`.
        full_no_eos = np.ascontiguousarray(seq_2d[: self._full_soft_len], dtype=np.int64)
        prompt_2d = np.ascontiguousarray(full_no_eos[: self._prompt_soft_len], dtype=np.int64)
        song_idx = int(self._dataset.windows[win_idx, 4])
        raw_slots = self._dataset.get_window_slot_codes(win_idx)
        slot_tuple = (
            tuple(int(c) for c in raw_slots.tolist())
            if raw_slots is not None
            else (128, 0, 32, -1)
        )
        return CARLExample(
            example_id=f"symic_win_{win_idx:07d}_s{song_idx:06d}",
            song_id=song_idx,
            prompt_tokens=prompt_2d,
            human_full_tokens=full_no_eos,
            slot_codes=slot_tuple,
        )

    def load_disjoint_pools(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[CARLDataSamplePool, CARLDataSamplePool]:
        windows = self._dataset.windows
        if len(windows) == 0:
            raise RuntimeError("SyMIcMemmapRepository found 0 windows of sufficient bar length.")

        song_ids = windows[:, 4]
        if self._dataset.slots is not None and self._min_active_tracks > 1:
            active_counts = (self._dataset.slots != SLOT_CODE_EMPTY).sum(axis=1)
            valid_win_mask = active_counts[song_ids] >= self._min_active_tracks
        else:
            valid_win_mask = np.ones((len(windows),), dtype=bool)

        valid_win_indices = np.flatnonzero(valid_win_mask)
        if valid_win_indices.size < 2:
            raise RuntimeError(
                f"Need at least 2 valid windows after min_active_tracks={self._min_active_tracks} filter, "
                f"got {valid_win_indices.size}."
            )

        unique_songs = np.unique(song_ids[valid_win_indices])
        if unique_songs.size < 2:
            raise RuntimeError("Need at least 2 distinct songs to build disjoint Odd/Even pools.")

        rng = np.random.default_rng(seed)
        shuffled_songs = rng.permutation(unique_songs)
        cut = max(1, len(shuffled_songs) // 2)
        ana_song_set = set(int(s) for s in shuffled_songs[:cut].tolist())

        ana_indices: List[int] = []
        gem_indices: List[int] = []
        for w_idx in valid_win_indices.tolist():
            s_id = int(song_ids[w_idx])
            if s_id in ana_song_set:
                ana_indices.append(int(w_idx))
            else:
                gem_indices.append(int(w_idx))

        if limit is not None and int(limit) > 0:
            half_lim = max(1, int(limit) // 2)
            if len(ana_indices) > half_lim:
                ana_indices = [ana_indices[int(i)] for i in rng.choice(len(ana_indices), size=half_lim, replace=False)]
            if len(gem_indices) > half_lim:
                gem_indices = [gem_indices[int(i)] for i in rng.choice(len(gem_indices), size=half_lim, replace=False)]

        return (
            LazyIndexedSamplePool(ana_indices, self._load_window_example),
            LazyIndexedSamplePool(gem_indices, self._load_window_example),
        )


class SyMIcMidiRepository(CARLDataRepository):
    """On-the-fly MIDI -> `MidiVision` repository for local Mac MPS runs and smoke tests."""

    def __init__(
        self,
        midi_manifest_path: str | Path,
        model: AblationTransformer2,
        vq_model: MidiVision,
        device: torch.device,
        prompt_bars: int = 4,
        total_bars: int = 16,
        min_active_tracks: int = 2,
    ) -> None:
        self._manifest_path = Path(midi_manifest_path)
        if not self._manifest_path.is_file():
            raise FileNotFoundError(f"MIDI manifest not found: {self._manifest_path}")
        self._model = model
        if int(vq_model.num_quantizers) != int(model.vq_layers) or int(vq_model.vq_groups) != int(model.vq_groups):
            raise ValueError(
                f"MidiVision shape (L={vq_model.num_quantizers}, G={vq_model.vq_groups}) "
                f"does not match SyMIc model (L={model.vq_layers}, G={model.vq_groups})."
            )
        self._vq = vq_model
        self._device = device
        self._prompt_chunks = int(prompt_bars) * CHUNKS_PER_BAR
        self._total_chunks = int(total_bars) * CHUNKS_PER_BAR
        self._total_bars = int(total_bars)
        self._min_active_tracks = max(1, int(min_active_tracks))
        self._vq_layers = int(model.vq_layers)
        self._prefix_len = 1 + (NUM_SLOT_TOKENS if bool(model.use_instrument_tokens) else 0)
        self._prompt_soft_len = self._prefix_len + self._prompt_chunks * self._vq_layers

    def load_disjoint_pools(
        self,
        limit: Optional[int] = 200,
        seed: int = 42,
    ) -> Tuple[CARLDataSamplePool, CARLDataSamplePool]:
        import json

        with self._manifest_path.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(entries))
        target_total = int(limit) if limit is not None and int(limit) > 0 else len(entries)

        collected: List[CARLExample] = []
        for raw_idx in perm.tolist():
            if len(collected) >= target_total:
                break
            entry = entries[raw_idx]
            path_str = entry if isinstance(entry, str) else str(entry.get("path", ""))
            if not path_str or not os.path.isfile(path_str):
                continue
            try:
                num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(path_str)
            except Exception:
                continue
            if num_chunks < self._total_chunks or len(notes_arr) == 0:
                continue
            slot_tuple = tuple(int(c) for c in slot_codes.tolist())
            if sum(1 for c in slot_tuple if c != SLOT_CODE_EMPTY) < self._min_active_tracks:
                continue

            seg_notes = extract_segment_canonical_notes(
                notes_arr, chunk_start=0, chunk_end=self._total_chunks, max_bars=self._total_bars
            )
            if len(seg_notes) == 0:
                continue
            # Convert canonical `(bar, pos, slot, pitch, duration, velocity)` to 5-col `(slot, start, end, pitch, vel)`.
            starts = seg_notes[:, 0] * 192 + seg_notes[:, 1]
            ends = np.minimum(self._total_chunks * 24, starts + seg_notes[:, 4])
            notes_5col = np.stack(
                [seg_notes[:, 2], starts, ends, seg_notes[:, 3], seg_notes[:, 5]], axis=1
            ).astype(np.int32)

            chunks_f32 = rasterize_notes_to_chunks_float(self._total_chunks, notes_5col)
            full_seq = Mortm5PromptEncoder.encode_chunks_to_prompt(
                chunks_f32,
                vq_model=self._vq,
                model=self._model,
                slot_codes=slot_tuple,
                device=self._device,
            )  # [1, full_soft_len, G]
            full_np = full_seq[0].detach().cpu().numpy().astype(np.int64)
            prompt_np = np.ascontiguousarray(full_np[: self._prompt_soft_len], dtype=np.int64)
            collected.append(
                CARLExample(
                    example_id=f"midi_s{raw_idx:06d}",
                    song_id=int(raw_idx),
                    prompt_tokens=prompt_np,
                    human_full_tokens=full_np,
                    slot_codes=slot_tuple,
                )
            )

        if len(collected) < 2:
            raise RuntimeError(f"SyMIcMidiRepository collected only {len(collected)} valid songs (need >= 2).")
        cut = len(collected) // 2
        return InMemorySamplePool(collected[:cut]), InMemorySamplePool(collected[cut:])


# ======================================================================
# 2. Critic (`AnalysisSyMIc` + `SyMIcCriticAdapter`)
# ======================================================================


class AnalysisSyMIc(nn.Module):
    """SyMIc causal discriminator with native Attention LoRA + continuation-only `MILPool`.

    Passes `[Prefix | Prompt (4 bars) | Continuation (12 bars)]` through the causal
    SyMIc decoder so continuation tokens attend to the 4-bar prompt seam, then pools
    **strictly over continuation positions `prompt_soft_len .. full_soft_len - 1`** via
    LogSumExp MIL (`MILPool`) to eliminate any prompt-shortcut leakage.
    """

    def __init__(
        self,
        backbone: AblationTransformer2,
        prompt_soft_len: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        mil_r_init: float = 2.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prompt_soft_len = int(prompt_soft_len)
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        inject_attention_lora(
            self.backbone.decoder.layers,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        dev = self.backbone.embedding.weight.device
        self.mil = MILPool(int(self.backbone.args.d_model), r_init=mil_r_init).to(dev)

    def forward_disc(self, seq_3d: Tensor) -> Tensor:
        """Compute 1D AI/Human logits `[B]` from `[B, S_full, G]` grouped token sequences."""
        if seq_3d.dim() != 3:
            raise ValueError(f"AnalysisSyMIc expects 3D [B, S, G] input, got {tuple(seq_3d.shape)}.")
        bsz, seq_len, _ = seq_3d.shape
        if seq_len <= self.prompt_soft_len:
            raise ValueError(
                f"Sequence length ({seq_len}) must exceed prompt_soft_len ({self.prompt_soft_len})."
            )

        dev_type = seq_3d.device.type
        padding_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=seq_3d.device)
        x_emb = self.backbone._aggregate_group_embeddings(seq_3d).to(dtype=torch.bfloat16)
        x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x_emb, padding_mask)

        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
            dec_unpad = self.backbone.decoder(
                tgt=x_unpad,
                tgt_is_causal=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_s,
                batch_size=bsz,
                indices=indices,
                is_save_cache=False,
            )
        dec_out = pad_input(dec_unpad, indices, bsz, seq_len)
        cont_hidden = dec_out[:, self.prompt_soft_len :, :].to(dtype=self.mil.scorer[0].weight.dtype)

        # Vectorized rectangular LogSumExp MIL pooling (identical to MILPool.forward on equal-length bags).
        tok_scores = self.mil.scorer(cont_hidden).squeeze(-1).float()  # [B, S_cont]
        r_val = self.mil.log_r.exp().clamp(0.05, 20.0)
        n_cont = float(tok_scores.size(1))
        return (torch.logsumexp(r_val * tok_scores, dim=1) - math.log(n_cont)) / r_val


class SyMIcCriticAdapter(CARLCriticAdapter):
    """`CARLCriticAdapter` wrapping `AnalysisSyMIc`."""

    def __init__(self, ana_model: AnalysisSyMIc, device: torch.device) -> None:
        self._ana = ana_model
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        config_source: str | dict | MORTM5Args,
        ckpt_path: Optional[str | Path],
        device: torch.device,
        prompt_bars: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ) -> "SyMIcCriticAdapter":
        backbone = load_symic_model(config_source, ckpt_path, device)
        prefix_len = 1 + (NUM_SLOT_TOKENS if bool(backbone.use_instrument_tokens) else 0)
        prompt_soft_len = prefix_len + int(prompt_bars) * CHUNKS_PER_BAR * int(backbone.vq_layers)
        ana = AnalysisSyMIc(
            backbone=backbone,
            prompt_soft_len=prompt_soft_len,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        return cls(ana_model=ana, device=device)

    @property
    def critic_module(self) -> nn.Module:
        return self._ana

    @property
    def device(self) -> torch.device:
        return self._device

    def mark_trainable(self) -> Dict[str, int]:
        groups = {"lora": 0, "mil": 0}
        for name, param in self._ana.named_parameters():
            is_lora = ".lora_A" in name or ".lora_B" in name
            is_mil = name.startswith("mil.")
            param.requires_grad_(bool(is_lora or is_mil))
            if is_lora:
                groups["lora"] += param.numel()
            elif is_mil:
                groups["mil"] += param.numel()
        if groups["lora"] == 0 or groups["mil"] == 0:
            raise RuntimeError(f"SyMIcCriticAdapter missing trainable parameter groups: {groups}")
        return groups

    def score_disc_logits(
        self,
        critic_sequences: Sequence[np.ndarray],
        chunk_size: Optional[int] = None,
        grad_enabled: bool = False,
    ) -> Tensor:
        step = len(critic_sequences) if chunk_size is None or chunk_size <= 0 else int(chunk_size)
        parts: List[Tensor] = []
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            for start in range(0, len(critic_sequences), step):
                arr = np.stack(critic_sequences[start : start + step], axis=0)
                batch_t = torch.as_tensor(arr, dtype=torch.long, device=self._device)
                parts.append(self._ana.forward_disc(batch_t))
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def export_compact_state(self) -> Dict[str, Any]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self._ana.state_dict().items()
            if (".lora_A" in k) or (".lora_B" in k) or k.startswith("mil.")
        }

    def load_compact_state(self, state: Dict[str, Any]) -> None:
        cleaned = {k: v for k, v in state.items() if k != "_carl_unmerged"}
        _, unexpected = self._ana.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in SyMIc Critic state_dict: {unexpected[:5]}")


# ======================================================================
# 3. Actor (`SyMIcGroupedActorAdapter`)
# ======================================================================


class SyMIcGroupedActorAdapter(CARLActorAdapter):
    """`CARLActorAdapter` for `SyMIc-G4L4 + Depth` (`AblationTransformer2`).

    Even-round parameter policy (A2):
      - Unfreezes all 12 layers' `self_attention` AND `Wout` (`DecodeCodebookHead`:
        Depth Transformer + group/special codebook projections).
      - Keeps `embedding`, `FFN`, and `RMSNorm` frozen.
      - Runs GRPO forward passes in `.eval()` mode to eliminate `dropout=0.1` KL noise.
    """

    def __init__(
        self,
        model: AblationTransformer2,
        device: torch.device,
        prompt_bars: int = 4,
        total_bars: int = 16,
        unfreeze_mode: str = "attn_and_depth",
    ) -> None:
        if unfreeze_mode not in ("attn_and_depth", "depth_only"):
            raise ValueError(
                f"unfreeze_mode must be 'attn_and_depth' or 'depth_only', got '{unfreeze_mode}'."
            )
        self._model = model
        self._device = device
        self._prompt_chunks = int(prompt_bars) * CHUNKS_PER_BAR
        self._total_chunks = int(total_bars) * CHUNKS_PER_BAR
        self._unfreeze_mode = unfreeze_mode
        self._prefix_len = 1 + (NUM_SLOT_TOKENS if bool(model.use_instrument_tokens) else 0)
        self._prompt_soft_len = self._prefix_len + self._prompt_chunks * int(model.vq_layers)
        self._full_soft_len = self._prefix_len + self._total_chunks * int(model.vq_layers)

    @classmethod
    def from_pretrained(
        cls,
        config_source: str | dict | MORTM5Args,
        ckpt_path: Optional[str | Path],
        device: torch.device,
        prompt_bars: int = 4,
        total_bars: int = 16,
        unfreeze_mode: str = "attn_and_depth",
    ) -> "SyMIcGroupedActorAdapter":
        model = load_symic_model(config_source, ckpt_path, device)
        return cls(
            model=model,
            device=device,
            prompt_bars=prompt_bars,
            total_bars=total_bars,
            unfreeze_mode=unfreeze_mode,
        )

    @property
    def actor_module(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def prompt_soft_len(self) -> int:
        return self._prompt_soft_len

    @property
    def full_soft_len(self) -> int:
        return self._full_soft_len

    @property
    def continuation_chunks(self) -> int:
        return self._total_chunks - self._prompt_chunks

    def unfreeze_for_grpo(self) -> Tuple[int, int]:
        n_train = 0
        n_total = 0
        for name, param in self._model.named_parameters():
            n_total += param.numel()
            is_wout = name.startswith("Wout.") or name.startswith("wout.")
            is_attn = ".self_attention." in name and self._unfreeze_mode == "attn_and_depth"
            trainable = bool(is_wout or is_attn)
            param.requires_grad_(trainable)
            if trainable:
                n_train += param.numel()
        if isinstance(self._model.Wout, DecodeCodebookHead):
            self._model.Wout._legacy_untrained_depth.fill_(False)
        return n_train, n_total

    @torch.no_grad()
    def rollout_group(
        self,
        examples: Sequence[CARLExample],
        group_size: int,
        sampling: UnifiedSamplingConfig,
        chunk_size: int = 16,
        seed: Optional[int] = None,
    ) -> RolloutBatch:
        if seed is not None:
            torch.manual_seed(int(seed))

        expanded_examples: List[CARLExample] = []
        prompt_list: List[np.ndarray] = []
        for ex in examples:
            for _ in range(group_size):
                expanded_examples.append(ex)
                prompt_list.append(ex.prompt_tokens)

        all_prompts = np.stack(prompt_list, axis=0)  # [N, S_prompt, G]
        step = max(1, int(chunk_size))
        gen_chunks: List[np.ndarray] = []

        self._model.eval()
        for start in range(0, len(expanded_examples), step):
            p_batch = torch.as_tensor(
                all_prompts[start : start + step], dtype=torch.long, device=self._device
            )
            out_seq = self._model.generate(
                num_chunks=self._total_chunks,
                prompt=p_batch,
                batch_size=p_batch.size(0),
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                min_p=sampling.min_p,
                append_eos=False,
                return_flat=False,
                repetition_penalty=sampling.repetition_penalty,
                repetition_window=sampling.repetition_window,
                fine_temperature=sampling.fine_temperature,
                fine_top_k=sampling.fine_top_k,
                fine_top_p=sampling.fine_top_p,
            )  # [chunk_B, full_soft_len, G]
            gen_chunks.append(out_seq[:, : self._full_soft_len, :].detach().cpu().numpy().astype(np.int64))

        stacked_out = np.concatenate(gen_chunks, axis=0)
        gen_tuples = tuple(stacked_out[i] for i in range(stacked_out.shape[0]))
        return RolloutBatch(
            examples=tuple(expanded_examples),
            generated_sequences=gen_tuples,
            critic_sequences=gen_tuples,
            group_size=group_size,
        )

    def _forward_continuation_codebook_logprobs(
        self,
        model: AblationTransformer2,
        seq_3d: Tensor,
    ) -> Tensor:
        """Compute `[B, T_cont, G, K]` log-softmax over valid RVQ codebook entries for continuation steps."""
        inp = seq_3d[:, :-1, :]
        tgt = seq_3d[:, 1:, :]
        c_start = self._prompt_soft_len - 1  # position predicting first continuation soft-token

        bsz, tgt_len, _ = inp.shape
        padding_mask = torch.ones((bsz, tgt_len), dtype=torch.bool, device=seq_3d.device)
        x_emb = model._aggregate_group_embeddings(inp).to(dtype=torch.bfloat16)
        x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x_emb, padding_mask)

        with torch.autocast(device_type=seq_3d.device.type, dtype=torch.bfloat16):
            dec_unpad = model.decoder(
                tgt=x_unpad,
                tgt_is_causal=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_s,
                batch_size=bsz,
                indices=indices,
                is_save_cache=False,
            )
        dec_out = pad_input(dec_unpad, indices, bsz, tgt_len)
        cont_hidden = dec_out[:, c_start:, :]
        cont_targets = tgt[:, c_start:, :]

        with torch.autocast(device_type=seq_3d.device.type, enabled=False):
            if isinstance(model.Wout, DecodeCodebookHead):
                group_embs = model._extract_shifted_group_embeddings(cont_targets)
                logits = model.Wout(
                    cont_hidden.to(model.Wout.weight.dtype),
                    start_pos=c_start,
                    group_embs=group_embs,
                ).float()
            else:
                raw = model.Wout(cont_hidden.to(model.Wout.weight.dtype)).float()
                logits = raw.reshape(*cont_hidden.shape[:-1], model.vq_groups, model.group_vocab_size)

        low = int(model.token_offset)
        high = low + int(model.codebook_size)
        codebook_logits = logits[..., low:high]  # [B, T_cont, G, K]
        return F.log_softmax(codebook_logits, dim=-1)

    def forward_logprobs_and_exact_kl(
        self,
        base_actor: CARLActorAdapter,
        examples: Sequence[CARLExample],
        generated_sequences: Sequence[np.ndarray],
    ) -> StepLogProbsKL:
        if not isinstance(base_actor.actor_module, AblationTransformer2):
            raise TypeError("SyMIcGroupedActorAdapter requires an AblationTransformer2 base_actor.")

        seq_np = np.stack(generated_sequences, axis=0)
        seq_3d = torch.as_tensor(seq_np, dtype=torch.long, device=self._device)

        with torch.no_grad():
            logp_base = self._forward_continuation_codebook_logprobs(base_actor.actor_module, seq_3d)
        logp_new = self._forward_continuation_codebook_logprobs(self._model, seq_3d)

        # Extract local codebook targets `[0, K)` for continuation positions `prompt_soft_len .. full_soft_len - 1`.
        cont_tokens = seq_3d[:, self._prompt_soft_len :, :]  # [B, T_cont, G]
        local_k = (
            (cont_tokens - int(self._model.token_offset)).clamp(min=0) % int(self._model.codebook_size)
        )  # [B, T_cont, G]

        tok_logp_per_group = logp_new.gather(-1, local_k.unsqueeze(-1)).squeeze(-1)  # [B, T_cont, G]
        joint_step_logp = tok_logp_per_group.sum(dim=-1)  # [B, T_cont]

        p_new = logp_new.exp()
        kl_per_group = (p_new * (logp_new - logp_base)).sum(dim=-1)  # [B, T_cont, G]
        joint_step_kl = kl_per_group.sum(dim=-1)  # [B, T_cont]

        mask = torch.ones_like(joint_step_logp, dtype=torch.bool)
        return StepLogProbsKL(
            action_logprobs=joint_step_logp,
            exact_kl=joint_step_kl,
            action_mask=mask,
        )

    def export_checkpoint_state(self) -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        cleaned = _extract_state_dict(state)
        cleaned = {k: v for k, v in cleaned.items() if k != "_carl_unmerged"}
        self._model.load_state_dict(cleaned, strict=True)


# ======================================================================
# 4. GPU-Decoded Zero-on-Human SyMIc Guardrail Suite (§2.4)
# ======================================================================


def _build_24_diatonic_scale_masks() -> Tensor:
    """Return `[24, 12]` float32 indicator matrix for all 12 Major + 12 Minor pitch-class sets."""
    major = (0, 2, 4, 5, 7, 9, 11)
    minor = (0, 2, 3, 5, 7, 8, 10)
    rows: List[List[float]] = []
    for root in range(12):
        for intervals in (major, minor):
            row = [0.0] * 12
            for iv in intervals:
                row[(root + iv) % 12] = 1.0
            rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


_SCALE_MASKS_24X12: Tensor = _build_24_diatonic_scale_masks()


@dataclass(frozen=True)
class SyMIcDecodedContext:
    """Deduplicated GPU-decoded piano-roll statistics shared across all 4 SyMIc guardrails."""

    batch_size: int
    is_empty: Tensor
    gen_onset_counts: Tensor
    gt_onset_counts: Tensor
    gen_blank_bar_ratio: Tensor
    gt_blank_bar_ratio: Tensor
    gen_out_of_scale: Tensor
    gt_out_of_scale: Tensor
    has_melodic_gt: Tensor
    gt_copy_iou: Tensor
    gen_adj_2bar_iou: Tensor
    gt_adj_2bar_iou: Tensor


class SyMIcPianoRollContextBuilder(GuardrailContextBuilder[SyMIcDecodedContext]):
    """Decodes continuation RVQ tokens via frozen `MidiVision` once per step (O(P) for GT, O(N) for Gen)."""

    def __init__(
        self,
        vq_model: MidiVision,
        prompt_soft_len: int,
        vq_layers: int = 4,
        vq_groups: int = 4,
        token_offset: int = 133,
        decode_chunk_size: int = 256,
    ) -> None:
        if int(vq_model.num_quantizers) != int(vq_layers) or int(vq_model.vq_groups) != int(vq_groups):
            raise ValueError(
                f"MidiVision shape (L={vq_model.num_quantizers}, G={vq_model.vq_groups}) "
                f"does not match SyMIc model (L={vq_layers}, G={vq_groups})."
            )
        self._vq = vq_model
        self._prompt_soft_len = int(prompt_soft_len)
        self._vq_layers = int(vq_layers)
        self._vq_groups = int(vq_groups)
        self._token_offset = int(token_offset)
        self._decode_chunk_size = max(16, int(decode_chunk_size))

    @torch.no_grad()
    def _decode_binary_rolls(
        self,
        sequences_3d: Sequence[np.ndarray],
        slot_codes_list: Sequence[Tuple[int, ...]],
    ) -> Tuple[Tensor, Tensor]:
        """Return `(onset_mask, active_mask)` on CPU bool of shape `[M, C_cont, 4, 128, 24]`."""
        dev = next(self._vq.parameters()).device
        m_cnt = len(sequences_3d)
        stacked = np.stack([seq[self._prompt_soft_len :, :] for seq in sequences_3d], axis=0)
        cont_steps = stacked.shape[1]
        if cont_steps % self._vq_layers != 0:
            raise ValueError(
                f"Continuation soft-token count ({cont_steps}) must be divisible by vq_layers ({self._vq_layers})."
            )
        c_cont = cont_steps // self._vq_layers
        flat_chunks = stacked.reshape(m_cnt * c_cont, self._vq_layers, self._vq_groups)

        onset_parts: List[Tensor] = []
        active_parts: List[Tensor] = []
        step = self._decode_chunk_size
        for start in range(0, flat_chunks.shape[0], step):
            tok_t = torch.as_tensor(flat_chunks[start : start + step], dtype=torch.long, device=dev)
            # `logit >= 0.0` is strictly equivalent to `sigmoid(logit) >= 0.5` without exp() overhead.
            logits = self._vq.decode_tokens(tok_t, token_offset=self._token_offset)  # [b, 8, 128, 24]
            active_8 = logits >= 0.0
            on_4 = active_8[:, 0::2, :, :]
            sus_4 = active_8[:, 1::2, :, :]
            onset_parts.append(on_4.cpu())
            active_parts.append((on_4 | sus_4).cpu())

        onset_all = torch.cat(onset_parts, dim=0).view(m_cnt, c_cont, 4, 128, 24)
        active_all = torch.cat(active_parts, dim=0).view(m_cnt, c_cont, 4, 128, 24)

        # Mask out inactive instrument slots (`SLOT_CODE_EMPTY == -1`) so residual decoder noise is ignored.
        for item_idx, slots in enumerate(slot_codes_list):
            for s_i, code in enumerate(slots[:4]):
                if int(code) == SLOT_CODE_EMPTY:
                    onset_all[item_idx, :, s_i, :, :] = False
                    active_all[item_idx, :, s_i, :, :] = False
        return onset_all, active_all

    @staticmethod
    def _compute_roll_stats(
        onset_mask: Tensor,
        active_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute `(onset_counts, blank_bar_ratio, out_of_scale, has_melodic, adj_2bar_iou)`."""
        m_cnt, c_cont, _, _, _ = active_mask.shape
        onset_counts = onset_mask.sum(dim=(1, 2, 3, 4)).float()

        # 1. Blank-bar ratio
        cpb = min(CHUNKS_PER_BAR, c_cont)
        n_bars = max(1, c_cont // cpb)
        trimmed = active_mask[:, : n_bars * cpb, :, :, :].view(m_cnt, n_bars, cpb, 4, 128, 24)
        bar_has_note = trimmed.any(dim=(2, 3, 4, 5))
        blank_bar_ratio = (~bar_has_note).float().mean(dim=1)

        # 2. Melodic out-of-scale ratio (strictly excluding slot 0 = Drums, matching MusPy)
        mel_active = active_mask[:, :, 1:, :, :]  # [M, C_cont, 3, 128, 24]
        pitch_mass = mel_active.sum(dim=(1, 2, 4)).float()  # [M, 128]
        pc_hist = torch.zeros((m_cnt, 12), dtype=torch.float32)
        for p_class in range(12):
            pc_hist[:, p_class] = pitch_mass[:, p_class::12].sum(dim=1)
        mel_total = pc_hist.sum(dim=1)
        has_melodic = mel_total > 0.0
        in_scale_mass = torch.matmul(pc_hist, _SCALE_MASKS_24X12.t())  # [M, 24]
        best_in_scale = in_scale_mass.max(dim=1).values / mel_total.clamp(min=1.0)
        out_of_scale = torch.where(has_melodic, 1.0 - best_in_scale, torch.zeros_like(best_in_scale))

        # 3. Adjacent 2-bar (or 1-bar when short) self-similarity IoU to catch riff copy-looping
        win_bars = 2 if n_bars >= 4 else 1
        n_wins = n_bars // win_bars
        if n_wins >= 2:
            win_view = trimmed[:, : n_wins * win_bars].reshape(m_cnt, n_wins, -1)
            w_a = win_view[:, :-1, :]
            w_b = win_view[:, 1:, :]
            inter = (w_a & w_b).sum(dim=-1).float()
            union = (w_a | w_b).sum(dim=-1).float()
            pair_iou = torch.where(union > 0.0, inter / union.clamp(min=1.0), torch.zeros_like(union))
            adj_iou = pair_iou.mean(dim=1)
        else:
            adj_iou = torch.zeros((m_cnt,), dtype=torch.float32)

        return onset_counts, blank_bar_ratio, out_of_scale, has_melodic, adj_iou

    def build_context(self, batch: RolloutBatch) -> SyMIcDecodedContext:
        n = len(batch)
        gen_slots = [ex.slot_codes for ex in batch.examples]
        gen_onset, gen_active = self._decode_binary_rolls(batch.generated_sequences, gen_slots)
        gen_on_cnt, gen_blank, gen_oos, _, gen_adj_iou = self._compute_roll_stats(gen_onset, gen_active)
        gen_any_active = gen_active.any(dim=(1, 2, 3, 4))
        is_empty = (gen_on_cnt <= 0.0) & (~gen_any_active)

        # Deduplicate human GT decoding by `example_id` (`O(P)` instead of `O(P * G)`).
        unique_map: Dict[str, int] = {}
        unique_gt_seqs: List[np.ndarray] = []
        unique_gt_slots: List[Tuple[int, ...]] = []
        example_to_uniq_idx: List[int] = []
        for ex in batch.examples:
            u_idx = unique_map.get(ex.example_id)
            if u_idx is None:
                u_idx = len(unique_gt_seqs)
                unique_map[ex.example_id] = u_idx
                unique_gt_seqs.append(ex.human_full_tokens)
                unique_gt_slots.append(ex.slot_codes)
            example_to_uniq_idx.append(u_idx)

        gt_onset_u, gt_active_u = self._decode_binary_rolls(unique_gt_seqs, unique_gt_slots)
        gt_on_u, gt_blank_u, gt_oos_u, gt_has_mel_u, gt_adj_u = self._compute_roll_stats(
            gt_onset_u, gt_active_u
        )

        idx_t = torch.tensor(example_to_uniq_idx, dtype=torch.long)
        gt_active_expanded = gt_active_u[idx_t]  # [N, C_cont, 4, 128, 24]
        inter_gt = (gen_active & gt_active_expanded).sum(dim=(1, 2, 3, 4)).float()
        union_gt = (gen_active | gt_active_expanded).sum(dim=(1, 2, 3, 4)).float()
        gt_copy_iou = torch.where(
            union_gt > 0.0, inter_gt / union_gt.clamp(min=1.0), torch.zeros_like(union_gt)
        )

        return SyMIcDecodedContext(
            batch_size=n,
            is_empty=is_empty,
            gen_onset_counts=gen_on_cnt,
            gt_onset_counts=gt_on_u[idx_t],
            gen_blank_bar_ratio=gen_blank,
            gt_blank_bar_ratio=gt_blank_u[idx_t],
            gen_out_of_scale=gen_oos,
            gt_out_of_scale=gt_oos_u[idx_t],
            has_melodic_gt=gt_has_mel_u[idx_t],
            gt_copy_iou=gt_copy_iou,
            gen_adj_2bar_iou=gen_adj_iou,
            gt_adj_2bar_iou=gt_adj_u[idx_t],
        )


class SyMIcSilenceDropoutRule(GuardrailRule[SyMIcDecodedContext]):
    """Penalizes continuations whose blank-bar ratio exceeds `human_gt_blank_ratio + margin`."""

    def __init__(self, weight: float = 0.40, blank_margin: float = 0.15) -> None:
        self._weight = float(weight)
        self._margin = float(blank_margin)

    @property
    def name(self) -> str:
        return "silence"

    def evaluate(self, context: SyMIcDecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        allowed = torch.clamp(context.gt_blank_bar_ratio + self._margin, max=1.0)
        denom = torch.clamp(1.0 - allowed, min=0.10)
        excess = torch.clamp(context.gen_blank_bar_ratio - allowed, min=0.0) / denom
        penalties = -self._weight * torch.clamp(excess, max=1.0)
        return (
            penalties,
            {"blank_bar_ratio": float(context.gen_blank_bar_ratio.mean().item())},
        )


class SyMIcRelativeDensityRule(GuardrailRule[SyMIcDecodedContext]):
    """Penalizes severe note-density collapse (`< 0.5x`) or note-spam explosion (`> 2.0x`) vs GT."""

    def __init__(
        self,
        weight: float = 0.25,
        band_octaves: float = 1.0,
        span_octaves: float = 1.5,
    ) -> None:
        self._weight = float(weight)
        self._band = float(band_octaves)
        self._span = max(0.1, float(span_octaves))

    @property
    def name(self) -> str:
        return "density"

    def evaluate(self, context: SyMIcDecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        ratio = (context.gen_onset_counts + 1.0) / (context.gt_onset_counts + 1.0)
        oct_dev = torch.abs(torch.log2(ratio))
        violation = torch.clamp((oct_dev - self._band) / self._span, min=0.0, max=1.0)
        penalties = -self._weight * violation
        return (
            penalties,
            {
                "onset_count_mean": float(context.gen_onset_counts.mean().item()),
                "density_ratio_mean": float(ratio.mean().item()),
            },
        )


class SyMIcScaleOutlierRule(GuardrailRule[SyMIcDecodedContext]):
    """Suppresses Fine-layer (`L1..L3`) chromatic scatter exceeding `max(gt_oos, tau_floor) + margin`."""

    def __init__(
        self,
        weight: float = 0.25,
        tau_floor: float = 0.12,
        margin: float = 0.08,
    ) -> None:
        self._weight = float(weight)
        self._tau_floor = float(tau_floor)
        self._margin = float(margin)

    @property
    def name(self) -> str:
        return "scale"

    def evaluate(self, context: SyMIcDecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        floor_t = torch.full_like(context.gt_out_of_scale, self._tau_floor)
        tau = torch.maximum(context.gt_out_of_scale, floor_t) + self._margin
        denom = torch.clamp(1.0 - tau, min=0.10)
        violation = torch.clamp((context.gen_out_of_scale - tau) / denom, min=0.0, max=1.0)
        violation = torch.where(context.has_melodic_gt, violation, torch.zeros_like(violation))
        penalties = -self._weight * violation
        return (
            penalties,
            {
                "out_of_scale_mean": float(context.gen_out_of_scale.mean().item()),
                "scale_consistency_est": float((1.0 - context.gen_out_of_scale).mean().item()),
            },
        )


class SyMIcLoopAndCopyRule(GuardrailRule[SyMIcDecodedContext]):
    """Blocks (a) verbatim GT continuation copying (`IoU > 0.5`) and (b) 2-bar riff self-looping."""

    def __init__(
        self,
        weight: float = 0.50,
        loop_floor: float = 0.65,
        loop_margin: float = 0.10,
        enable_gt_copy_penalty: bool = True,
    ) -> None:
        self._weight = float(weight)
        self._loop_floor = float(loop_floor)
        self._loop_margin = float(loop_margin)
        self._enable_gt_copy = bool(enable_gt_copy_penalty)

    @property
    def name(self) -> str:
        return "loop_copy"

    def evaluate(self, context: SyMIcDecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        # (a) Reference continuation copy penalty (<= 0, matching CARL v2 `similarity_reward`)
        if self._enable_gt_copy:
            copy_pens = torch.tensor(
                [
                    similarity_reward(float(sim), weight=self._weight)
                    for sim in context.gt_copy_iou.tolist()
                ],
                dtype=torch.float32,
            )
        else:
            copy_pens = torch.zeros((context.batch_size,), dtype=torch.float32)

        # (b) Degenerate adjacent 2-bar self-loop penalty (strictly 0 on human GT)
        floor_t = torch.full_like(context.gt_adj_2bar_iou, self._loop_floor)
        loop_tau = torch.maximum(context.gt_adj_2bar_iou, floor_t) + self._loop_margin
        denom = torch.clamp(1.0 - loop_tau, min=0.10)
        loop_v = torch.clamp((context.gen_adj_2bar_iou - loop_tau) / denom, min=0.0, max=1.0)
        loop_pens = -self._weight * loop_v

        # Take the worse (more negative) of copy vs self-loop so maximum penalty stays bounded at `-weight`.
        combined = torch.minimum(copy_pens, loop_pens)
        return (
            combined,
            {
                "gt_copy_iou_mean": float(context.gt_copy_iou.mean().item()),
                "adj_2bar_iou_mean": float(context.gen_adj_2bar_iou.mean().item()),
            },
        )


def build_symic_guardrail_suite(
    vq_model: MidiVision,
    actor: SyMIcGroupedActorAdapter,
    enable_gt_copy_penalty: bool = True,
) -> GuardrailSuite[SyMIcDecodedContext]:
    raw_model = actor.actor_module
    if not isinstance(raw_model, AblationTransformer2):
        raise TypeError(f"Expected AblationTransformer2, got {type(raw_model).__name__}.")
    builder = SyMIcPianoRollContextBuilder(
        vq_model=vq_model,
        prompt_soft_len=actor.prompt_soft_len,
        vq_layers=int(raw_model.vq_layers),
        vq_groups=int(raw_model.vq_groups),
        token_offset=int(raw_model.token_offset),
    )
    return GuardrailSuite(
        context_builder=builder,
        empty_detector=lambda ctx: ctx.is_empty,
        rules=(
            SyMIcSilenceDropoutRule(),
            SyMIcRelativeDensityRule(),
            SyMIcScaleOutlierRule(),
            SyMIcLoopAndCopyRule(enable_gt_copy_penalty=enable_gt_copy_penalty),
        ),
    )
