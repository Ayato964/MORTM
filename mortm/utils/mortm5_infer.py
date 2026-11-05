"""
MORTM5 Conditional Multi-Track MIDI Inference Utilities.

Provides clean, SOLID-compliant domain configs, prompt encoders, and generation
services for MORTM5 8-Way Ablation Transformers (#1..#8) paired with MidiVision
RVQ tokenizers (M3 / M4).

CLI usage:
  python -m mortm.utils.mortm5_infer --ablation "#6" --slots drums piano sax bass --bars 8 --out out/sample.mid
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from mortm.models.modules.config import (
    MORTM5Args,
    RVQ_TOKEN_OFFSET,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
    resolve_ablation_subdir,
)
from mortm.models.mortm5 import (
    MidiVision,
    _BaseMORTM5Transformer,
    build_mortm5_ablation_model,
)
from mortm.utils.convert_pianoroll import (
    CHUNK_STEPS,
    CHUNKS_PER_BAR,
    chunks_to_midi,
    midi_to_chunks,
    slot_codes_to_roll_meta,
    slots_to_slot_codes,
)


INSTRUMENT_ALIASES: Dict[str, int] = {
    "drums": SLOT_CODE_DRUMS,
    "drum": SLOT_CODE_DRUMS,
    "empty": SLOT_CODE_EMPTY,
    "none": SLOT_CODE_EMPTY,
    "piano": 0,          # Acoustic Grand Piano
    "epiano": 4,         # Electric Piano 1
    "organ": 16,         # Drawbar Organ
    "guitar": 25,        # Acoustic Guitar (steel)
    "eguitar": 27,       # Electric Guitar (clean)
    "bass": 32,          # Acoustic Bass
    "ebass": 33,         # Electric Bass (finger)
    "violin": 40,        # Violin
    "strings": 48,       # String Ensemble 1
    "trumpet": 56,       # Trumpet
    "trombone": 57,      # Trombone
    "sax": 65,           # Alto Sax
    "altosax": 65,       # Alto Sax
    "tenorsax": 66,      # Tenor Sax
    "flute": 73,         # Flute
    "synth": 80,         # Lead 1 (square)
}

TOKENIZER_CONFIGS: Dict[str, Dict[str, str]] = {
    "m3_g8l2": {
        "config": "configs/models/mortm5/m3_finegroup_g8l2_d256_k2048.json",
        "ckpt": "out/models/mortm5/m3_finegroup_g8l2_d256_k2048/MORTM5.M3_FineGroup_G8L2_D256_K2048_latest.pth",
    },
    "m4_g4l4": {
        "config": "configs/models/mortm5/m4_deeprvq_g4l4_d128_k2048.json",
        "ckpt": "out/models/mortm5/m4_deeprvq_g4l4_d128_k2048/MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth",
    },
}


def parse_slot_spec(items: Sequence[str]) -> List[int]:
    """Parse 1..4 instrument names or GM program numbers into [Slot0, Slot1, Slot2, Slot3]."""
    if not (1 <= len(items) <= 4):
        raise ValueError(f"Expected 1..4 slot items, got {items}")

    def _to_code(raw: str) -> int:
        key = str(raw).strip().lower()
        if key in INSTRUMENT_ALIASES:
            return INSTRUMENT_ALIASES[key]
        return int(key)

    raw_codes = [_to_code(x) for x in items]

    if len(raw_codes) == 4:
        codes = raw_codes
    else:
        if raw_codes[0] in (SLOT_CODE_DRUMS, SLOT_CODE_EMPTY):
            slot0 = raw_codes[0]
            melodic = raw_codes[1:]
        else:
            slot0 = SLOT_CODE_EMPTY
            melodic = raw_codes
        if len(melodic) > 3:
            raise ValueError(f"At most 3 melodic instruments can be assigned to Slots 1..3: got {items}")
        codes = [slot0] + melodic + [SLOT_CODE_EMPTY] * (3 - len(melodic))

    if codes[0] not in (SLOT_CODE_DRUMS, SLOT_CODE_EMPTY):
        raise ValueError(f"Slot 0 must be drums (128/'drums') or empty (-1/'empty'), got {items[0]}")
    for idx in range(1, 4):
        c = codes[idx]
        if not (c == SLOT_CODE_EMPTY or 0 <= c <= 127):
            raise ValueError(f"Slot {idx} must be a MIDI Program (0..127) or -1 ('empty'), got {c}")
    return codes


def auto_find_checkpoint(ablation_id: str, model_name: str) -> Optional[str]:
    """Locate the most recently modified checkpoint for an ablation or Stage-3 model under standard project directories."""
    sub = resolve_ablation_subdir(model_name=model_name, ablation_id=ablation_id)
    roots = [
        "out/mortm5_symic_main",
        "checkpoints/mortm5_symic_main",
        "out/mortm5_ablation",
        "checkpoints/mortm5_ablation",
    ]
    search_patterns: List[str] = []
    for root in roots:
        search_patterns.extend(
            [
                f"{root}/{sub}/MORTM5.{model_name}*_latest*.pth",
                f"{root}/{sub}/MORTM5.{model_name}*.checkpoint.pt",
                f"{root}/{sub}/*.pth",
                f"{root}/{sub}/*.pt",
                f"{root}/MORTM5.{model_name}*_latest*.pth",
                f"{root}/MORTM5.{model_name}*.pth",
            ]
        )
    for pat in search_patterns:
        matches = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    return None


def select_device(requested: Optional[str] = None) -> torch.device:
    """Select PyTorch compute device (explicit override -> CUDA -> MPS -> CPU)."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class Mortm5SamplingConfig:
    """Immutable coarse-to-fine sampling parameters for MORTM5 autoregressive generation."""

    temperature: float = 0.95
    top_k: int = 40
    top_p: float = 0.95
    fine_temperature: float = 0.50
    fine_top_k: int = 8
    fine_top_p: float = 0.85
    repetition_penalty: float = 1.0
    repetition_window: int = 24
    min_notes: int = 1
    max_notes: Optional[int] = None
    sliding_window_chunks: Optional[int] = 128

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.fine_temperature < 0.0:
            raise ValueError(f"fine_temperature must be >= 0.0, got {self.fine_temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.fine_top_k < 0:
            raise ValueError(f"fine_top_k must be >= 0, got {self.fine_top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if not (0.0 < self.fine_top_p <= 1.0):
            raise ValueError(f"fine_top_p must be in (0.0, 1.0], got {self.fine_top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
        if self.repetition_window < 1:
            raise ValueError(f"repetition_window must be >= 1, got {self.repetition_window}")
        if self.sliding_window_chunks is not None and int(self.sliding_window_chunks) < 2:
            raise ValueError(f"sliding_window_chunks must be >= 2 or None, got {self.sliding_window_chunks}")


def _compute_sliding_window_step(
    cur_chunk: int,
    total_chunks: int,
    max_win_chunks: int,
) -> tuple[int, int, int]:
    """
    Compute `(win_start_chunk, ctx_chunks, step_new_chunks)` for long-horizon continuation.
    Aligns `win_start_chunk` to bar multiples (`CHUNKS_PER_BAR = 8`) whenever possible so
    bar-relative downbeat phase (`pos = 0..191`) is strictly preserved across sliding windows.
    """
    max_w = max(2, int(max_win_chunks))
    if total_chunks <= max_w:
        win_start = 0
    else:
        half_w = max(1, min(max_w - 1, max(CHUNKS_PER_BAR, max_w // 2)))
        if cur_chunk <= half_w:
            win_start = 0
        else:
            raw_start = cur_chunk - half_w
            win_start = (raw_start // CHUNKS_PER_BAR) * CHUNKS_PER_BAR
            if cur_chunk - win_start >= max_w:
                win_start = cur_chunk - (max_w - 1)
    ctx_chunks = cur_chunk - win_start
    step_new_chunks = min(total_chunks - cur_chunk, max_w - ctx_chunks)
    return win_start, ctx_chunks, step_new_chunks


def _normalize_chunks_f32_to_vq_layout(chunks_f32: np.ndarray) -> np.ndarray:
    """
    Normalize a 4D piano-roll chunk array into MidiVision's canonical `[N, 8, 128, 24]` layout.
    Accepts either `[N, 8, 24, 128]` (`midi_to_chunks` layout) or `[N, 8, 128, 24]` (`MidiVision` layout).
    """
    if chunks_f32.ndim != 4 or chunks_f32.shape[1] != 8:
        raise ValueError(
            f"Expected 4D piano-roll chunks [N, 8, 24, 128] or [N, 8, 128, 24], got shape {chunks_f32.shape}."
        )
    if chunks_f32.shape[2:] == (24, 128):
        return np.ascontiguousarray(chunks_f32.transpose(0, 1, 3, 2), dtype=np.float32)
    if chunks_f32.shape[2:] == (128, 24):
        return np.ascontiguousarray(chunks_f32, dtype=np.float32)
    raise ValueError(
        f"Expected spatial chunk dims (24, 128) or (128, 24), got {chunks_f32.shape[2:]}."
    )


class Mortm5PromptEncoder:
    """Encodes float32 piano-roll chunks into downstream Flat or Grouped prefix token sequences."""

    @staticmethod
    @torch.inference_mode()
    def encode_chunks_to_prompt(
        chunks_f32: np.ndarray,
        vq_model: MidiVision,
        model: _BaseMORTM5Transformer,
        slot_codes: Sequence[int],
        device: Optional[torch.device] = None,
        batch_size: int = 64,
    ) -> Tensor:
        target_device = device or model.embedding.weight.device
        n_chunks = int(chunks_f32.shape[0])
        prefix = model.build_prefix_tokens(slot_codes=slot_codes, batch_size=1, device=target_device)
        if n_chunks == 0:
            return prefix

        chunks_vq = _normalize_chunks_f32_to_vq_layout(chunks_f32)
        bs = max(1, int(batch_size))
        parts: List[Tensor] = []
        for start in range(0, n_chunks, bs):
            batch_np = chunks_vq[start : start + bs]
            x = torch.from_numpy(batch_np).to(device=target_device, dtype=torch.float32)
            parts.append(vq_model.encode_tokens(x, include_offset=True))
        raw_rvq = torch.cat(parts, dim=0)  # [N, L * G]

        L = int(model.vq_layers)
        G = int(model.vq_groups)
        K = int(model.codebook_size)
        rvq_3d = raw_rvq.reshape(n_chunks, L, G).to(dtype=torch.long)

        shift = int(model.token_offset) - int(getattr(vq_model, "token_offset", RVQ_TOKEN_OFFSET))
        rvq_3d = rvq_3d + shift
        if getattr(model, "separate_rvq_layers", False):
            layer_offsets = torch.arange(L, device=target_device, dtype=torch.long).view(1, L, 1) * (G * K)
            rvq_3d = rvq_3d + layer_offsets

        if getattr(model.args, "model_mode", "flat") == "grouped":
            prompt_body = rvq_3d.reshape(1, n_chunks * L, G)
        else:
            prompt_body = rvq_3d.reshape(1, n_chunks * L * G)
        return torch.cat([prefix, prompt_body], dim=1)


class Mortm5MidiGenerator:
    """High-level service coordinating MORTM5 autoregressive generation (RVQ or Symbolic) and MIDI decoding."""

    def __init__(
        self,
        vq_model: Optional[MidiVision],
        model: _BaseMORTM5Transformer,
        device: torch.device,
    ) -> None:
        self._vq_model = vq_model
        self._model = model
        self._device = device

    @property
    def vq_model(self) -> Optional[MidiVision]:
        return self._vq_model

    @property
    def model(self) -> _BaseMORTM5Transformer:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_symbolic(self) -> bool:
        return bool(getattr(self._model, "is_symbolic", False))

    @classmethod
    def from_pretrained(
        cls,
        ablation: str = "#6",
        ckpt_path: Optional[str] = None,
        vq_ckpt_path: Optional[str] = None,
        device: Optional[str | torch.device] = None,
    ) -> "Mortm5MidiGenerator":
        dev = device if isinstance(device, torch.device) else select_device(device)
        m_args = MORTM5Args(ablation)

        resolved_ckpt = ckpt_path or auto_find_checkpoint(ablation, m_args.name)
        if not resolved_ckpt or not os.path.isfile(resolved_ckpt):
            raise FileNotFoundError(
                f"MORTM5 checkpoint not found for {ablation} ({m_args.name}). Specify --ckpt explicitly."
            )

        is_symbolic = bool(
            getattr(m_args, "symbolic_vocab", False)
            or getattr(m_args, "model_mode", "flat") in ("octuple", "remi")
        )
        vq_model: Optional[MidiVision] = None
        if not is_symbolic:
            tok_key = m_args.tokenizer_key or ("m4_g4l4" if m_args.vq_groups == 4 else "m3_g8l2")
            if tok_key not in TOKENIZER_CONFIGS:
                raise KeyError(
                    f"Unknown RVQ tokenizer_key '{tok_key}'. Available: {sorted(TOKENIZER_CONFIGS.keys())}"
                )
            tok_info = TOKENIZER_CONFIGS[tok_key]
            resolved_vq_ckpt = vq_ckpt_path or tok_info["ckpt"]
            if not os.path.isfile(resolved_vq_ckpt):
                raise FileNotFoundError(f"MidiVision tokenizer checkpoint not found: {resolved_vq_ckpt}")

            vq_args = MORTM5Args(tok_info["config"])
            vq_model = MidiVision(vq_args, sync_codebook=False).to(dev).eval()
            vq_sd = torch.load(resolved_vq_ckpt, map_location="cpu", weights_only=False)
            vq_sd = vq_sd.get("model_state_dict", vq_sd) if isinstance(vq_sd, dict) else vq_sd
            vq_model.load_state_dict(vq_sd, strict=True)

        model = build_mortm5_ablation_model(m_args).to(dev).eval()
        raw_ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
        state_dict = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
        model.load_state_dict(state_dict, strict=True)

        return cls(vq_model=vq_model, model=model, device=dev)

    @torch.inference_mode()
    def _generate_symbolic_notes(
        self,
        slot_codes: Sequence[int],
        num_chunks: int,
        sampling: Optional[Mortm5SamplingConfig] = None,
        prompt_chunks_f32: Optional[np.ndarray] = None,
        prompt_midi_path: Optional[str | Path] = None,
        prompt_chunks_limit: Optional[int] = None,
    ) -> tuple[np.ndarray, int, List[int]]:
        """
        Generate `num_chunks` new symbolic music chunks in global quantized note space
        (`int32 [M, 5]` `(slot, start, end, pitch, velocity)`), supporting both single-window
        generation (`<= 16` bars, including sub-bar / half-beat prompts) and bar-aligned
        16-bar sliding-window continuation (`> 16` bars, e.g. 8-bar -> 24..32-bar rollout).
        """
        import math

        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")

        cfg = sampling or Mortm5SamplingConfig()
        tok = self._model.get_symbolic_tokenizer()
        eff_slots = [int(c) for c in slot_codes]
        prompt_len = 0
        prompt_notes = np.zeros((0, 5), dtype=np.int32)

        if prompt_midi_path is not None:
            from mortm.utils.convert_pianoroll import extract_quantized_notes_and_slots

            raw_n_chunks, raw_notes, midi_slots = extract_quantized_notes_and_slots(prompt_midi_path)
            eff_slots = [int(c) for c in midi_slots.tolist()]
            limit_c = int(prompt_chunks_limit) if prompt_chunks_limit is not None else int(raw_n_chunks)
            prompt_len = max(0, min(int(raw_n_chunks), limit_c))
            if prompt_len > 0:
                prompt_notes = tok.filter_notes_by_slots_and_horizon(
                    raw_notes, slot_codes=eff_slots, num_chunks=prompt_len
                )
        elif prompt_chunks_f32 is not None and int(prompt_chunks_f32.shape[0]) > 0:
            raw_len = int(prompt_chunks_f32.shape[0])
            limit_c = int(prompt_chunks_limit) if prompt_chunks_limit is not None else raw_len
            prompt_len = max(0, min(raw_len, limit_c))
            if prompt_len > 0:
                prompt_notes = tok.extract_notes_from_chunks_f32(
                    prompt_chunks_f32[:prompt_len], slot_codes=eff_slots
                )

        model_max_win_chunks = max(1, int(getattr(tok, "max_segment_bars", 16))) * CHUNKS_PER_BAR
        max_win_chunks = (
            min(model_max_win_chunks, max(2, int(cfg.sliding_window_chunks)))
            if cfg.sliding_window_chunks is not None
            else model_max_win_chunks
        )

        total_chunks = prompt_len + int(num_chunks)
        cur_chunk = prompt_len
        accum_parts: List[np.ndarray] = [prompt_notes]

        while cur_chunk < total_chunks:
            win_start_chunk, ctx_chunks, step_new_chunks = _compute_sliding_window_step(
                cur_chunk=cur_chunk,
                total_chunks=total_chunks,
                max_win_chunks=max_win_chunks,
            )
            win_total_chunks = ctx_chunks + step_new_chunks
            win_max_bars = max(1, math.ceil(win_total_chunks / CHUNKS_PER_BAR))
            win_min_bar = ctx_chunks // CHUNKS_PER_BAR
            win_min_pos = (ctx_chunks % CHUNKS_PER_BAR) * CHUNK_STEPS
            win_max_steps = win_total_chunks * CHUNK_STEPS

            if ctx_chunks > 0:
                cur_all_notes = np.concatenate(accum_parts, axis=0) if len(accum_parts) > 1 else accum_parts[0]
                win_prompt_seq = tok.encode_notes_window_prompt(
                    cur_all_notes,
                    chunk_start=win_start_chunk,
                    chunk_end=cur_chunk,
                    slot_codes=eff_slots,
                    append_eos=False,
                ).to(device=self._device)
            else:
                win_prompt_seq = None

            win_max_notes = (
                max(1, math.ceil(int(cfg.max_notes) * step_new_chunks / max(1, int(num_chunks))))
                if cfg.max_notes is not None and int(cfg.max_notes) > 0
                else max(256, int(step_new_chunks) * 16)
            )
            raw_min_notes = max(0, int(cfg.min_notes))
            win_min_notes = (
                0
                if raw_min_notes == 0
                else min(win_max_notes, max(1, math.ceil(raw_min_notes * step_new_chunks / max(1, int(num_chunks)))))
            )

            seq = self._model.generate(
                max_notes=win_max_notes,
                max_bars=win_max_bars,
                min_notes=win_min_notes,
                min_bar=win_min_bar,
                min_pos=win_min_pos,
                max_steps=win_max_steps,
                prompt=win_prompt_seq,
                slot_codes=eff_slots,
                batch_size=1,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                monotonic_time=True,
            )

            win_notes, _ = tok.decode_tokens_to_notes(seq[0], num_chunks=None, slot_codes=eff_slots)
            ctx_steps = ctx_chunks * CHUNK_STEPS
            if len(win_notes) > 0:
                new_mask = (win_notes[:, 1] >= ctx_steps) & (win_notes[:, 1] < win_max_steps)
                if np.any(new_mask):
                    new_notes = win_notes[new_mask].copy()
                    win_start_step = win_start_chunk * CHUNK_STEPS
                    new_notes[:, 1] += win_start_step
                    new_notes[:, 2] = np.minimum(total_chunks * CHUNK_STEPS, new_notes[:, 2] + win_start_step)
                    valid_new = new_notes[:, 2] > new_notes[:, 1]
                    if np.any(valid_new):
                        accum_parts.append(new_notes[valid_new])

            cur_chunk += step_new_chunks

        full_notes = np.concatenate(accum_parts, axis=0) if len(accum_parts) > 1 else accum_parts[0]
        return full_notes, prompt_len, eff_slots

    @staticmethod
    def _resolve_rvq_prompt(
        slot_codes: Sequence[int],
        prompt_chunks_f32: Optional[np.ndarray] = None,
        prompt_midi_path: Optional[str | Path] = None,
        prompt_chunks_limit: Optional[int] = None,
    ) -> tuple[Optional[np.ndarray], List[int]]:
        """Resolve either `prompt_midi_path` or `prompt_chunks_f32` into canonical `[N, 8, 128, 24]` chunks and `slot_codes`."""
        if prompt_midi_path is not None:
            raw_chunks, roll_meta = midi_to_chunks(prompt_midi_path)
            eff_slots = [int(c) for c in slots_to_slot_codes(roll_meta.slots)]
            limit_c = int(prompt_chunks_limit) if prompt_chunks_limit is not None else int(raw_chunks.shape[0])
            sliced = raw_chunks[: max(0, limit_c)]
            if int(sliced.shape[0]) == 0:
                return None, eff_slots
            return _normalize_chunks_f32_to_vq_layout(sliced), eff_slots

        eff_slots = [int(c) for c in slot_codes]
        if prompt_chunks_f32 is not None and int(prompt_chunks_f32.shape[0]) > 0:
            limit_c = (
                int(prompt_chunks_limit)
                if prompt_chunks_limit is not None
                else int(prompt_chunks_f32.shape[0])
            )
            sliced = prompt_chunks_f32[: max(0, limit_c)]
            if int(sliced.shape[0]) == 0:
                return None, eff_slots
            return _normalize_chunks_f32_to_vq_layout(sliced), eff_slots

        return None, eff_slots

    @torch.inference_mode()
    def generate_chunks(
        self,
        slot_codes: Sequence[int],
        num_chunks: int,
        sampling: Optional[Mortm5SamplingConfig] = None,
        prompt_chunks_f32: Optional[np.ndarray] = None,
        decode_batch_size: int = 64,
        prompt_midi_path: Optional[str | Path] = None,
        prompt_chunks_limit: Optional[int] = None,
    ) -> np.ndarray:
        """Generate `num_chunks` new piano-roll chunks [num_chunks, 8, 128, 24] in float32 [0, 1]."""
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")

        if self.is_symbolic:
            full_notes, prompt_len, eff_slots = self._generate_symbolic_notes(
                slot_codes=slot_codes,
                num_chunks=num_chunks,
                sampling=sampling,
                prompt_chunks_f32=prompt_chunks_f32,
                prompt_midi_path=prompt_midi_path,
                prompt_chunks_limit=prompt_chunks_limit,
            )
            tok = self._model.get_symbolic_tokenizer()
            full_f32 = tok.render_notes_to_chunks_float(
                full_notes, num_chunks=prompt_len + int(num_chunks), slot_codes=eff_slots
            )
            full_vq = _normalize_chunks_f32_to_vq_layout(full_f32)
            probs = full_vq[prompt_len : prompt_len + int(num_chunks)]
            for slot_idx, code in enumerate(eff_slots):
                if int(code) == SLOT_CODE_EMPTY:
                    probs[:, slot_idx * 2 : slot_idx * 2 + 2] = 0.0
            return probs

        if self._vq_model is None:
            raise RuntimeError("RVQ generation requires a loaded MidiVision vq_model.")
        cfg = sampling or Mortm5SamplingConfig()

        norm_prompt_vq, eff_slots = self._resolve_rvq_prompt(
            slot_codes=slot_codes,
            prompt_chunks_f32=prompt_chunks_f32,
            prompt_midi_path=prompt_midi_path,
            prompt_chunks_limit=prompt_chunks_limit,
        )
        prompt_len = int(norm_prompt_vq.shape[0]) if norm_prompt_vq is not None else 0
        prompt_seq = (
            Mortm5PromptEncoder.encode_chunks_to_prompt(
                norm_prompt_vq,
                vq_model=self._vq_model,
                model=self._model,
                slot_codes=eff_slots,
                device=self._device,
            )
            if norm_prompt_vq is not None and prompt_len > 0
            else None
        )

        total_target_chunks = prompt_len + num_chunks
        L = int(self._model.vq_layers)
        G = int(self._model.vq_groups)
        tokens_per_chunk = L * G

        if cfg.sliding_window_chunks is None or total_target_chunks <= int(cfg.sliding_window_chunks):
            seq = self._model.generate(
                num_chunks=total_target_chunks,
                prompt=prompt_seq,
                slot_codes=eff_slots,
                batch_size=1,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                repetition_window=cfg.repetition_window,
                fine_temperature=cfg.fine_temperature,
                fine_top_k=cfg.fine_top_k,
                fine_top_p=cfg.fine_top_p,
            )
            rvq_all = self._model.extract_rvq_tokens(seq)
            new_rvq = rvq_all[prompt_len * tokens_per_chunk :].view(num_chunks, L, G)
        else:
            max_win_chunks = max(2, int(cfg.sliding_window_chunks))
            prefix = self._model.build_prefix_tokens(slot_codes=eff_slots, batch_size=1, device=self._device)
            is_grouped = getattr(self._model.args, "model_mode", "flat") == "grouped"
            if prompt_seq is not None and prompt_len > 0:
                p_flat = prompt_seq.reshape(-1)
                body_model_3d = p_flat[p_flat >= self._model.token_offset].reshape(prompt_len, L, G)
            else:
                body_model_3d = torch.empty((0, L, G), dtype=torch.long, device=self._device)

            cur_chunk = prompt_len
            while cur_chunk < total_target_chunks:
                win_start_chunk, ctx_chunks, step_new_chunks = _compute_sliding_window_step(
                    cur_chunk=cur_chunk,
                    total_chunks=total_target_chunks,
                    max_win_chunks=max_win_chunks,
                )
                win_total_chunks = ctx_chunks + step_new_chunks
                if ctx_chunks > 0:
                    ctx_3d = body_model_3d[win_start_chunk : cur_chunk]
                    ctx_body = ctx_3d.reshape(1, ctx_chunks * L, G) if is_grouped else ctx_3d.reshape(1, ctx_chunks * tokens_per_chunk)
                    win_prompt = torch.cat([prefix, ctx_body], dim=1)
                else:
                    win_prompt = None

                win_seq = self._model.generate(
                    num_chunks=win_total_chunks,
                    prompt=win_prompt,
                    slot_codes=eff_slots,
                    batch_size=1,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    repetition_penalty=cfg.repetition_penalty,
                    repetition_window=cfg.repetition_window,
                    fine_temperature=cfg.fine_temperature,
                    fine_top_k=cfg.fine_top_k,
                    fine_top_p=cfg.fine_top_p,
                )
                win_flat = win_seq.reshape(-1)
                win_body_3d = win_flat[win_flat >= self._model.token_offset].reshape(win_total_chunks, L, G)
                body_model_3d = torch.cat([body_model_3d, win_body_3d[ctx_chunks : win_total_chunks]], dim=0)
                cur_chunk += step_new_chunks

            new_body_model = body_model_3d[prompt_len : prompt_len + num_chunks]
            new_rvq = self._model.extract_rvq_tokens(new_body_model.reshape(1, -1)).view(num_chunks, L, G)

        bs = max(1, int(decode_batch_size))
        decoded_parts: List[np.ndarray] = []
        for start in range(0, num_chunks, bs):
            logits = self._vq_model.decode_tokens(new_rvq[start : start + bs].to(self._device))
            decoded_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
        probs = np.concatenate(decoded_parts, axis=0)

        for slot_idx, code in enumerate(eff_slots):
            if int(code) == SLOT_CODE_EMPTY:
                probs[:, slot_idx * 2 : slot_idx * 2 + 2] = 0.0
        return probs

    def generate_to_midi(
        self,
        output_path: str | Path,
        slot_codes: Sequence[int],
        num_chunks: int,
        sampling: Optional[Mortm5SamplingConfig] = None,
        prompt_chunks_f32: Optional[np.ndarray] = None,
        prompt_midi_path: Optional[str | Path] = None,
        prompt_chunks_limit: Optional[int] = None,
    ) -> Path:
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if self.is_symbolic:
            full_notes, prompt_len, eff_slots = self._generate_symbolic_notes(
                slot_codes=slot_codes,
                num_chunks=num_chunks,
                sampling=sampling,
                prompt_chunks_f32=prompt_chunks_f32,
                prompt_midi_path=prompt_midi_path,
                prompt_chunks_limit=prompt_chunks_limit,
            )
            tok = self._model.get_symbolic_tokenizer()
            tok.render_notes_to_midi(
                full_notes,
                output_path=str(out_p),
                slot_codes=eff_slots,
                num_chunks=prompt_len + int(num_chunks),
            )
            return out_p

        norm_prompt_vq, eff_slots = self._resolve_rvq_prompt(
            slot_codes=slot_codes,
            prompt_chunks_f32=prompt_chunks_f32,
            prompt_midi_path=prompt_midi_path,
            prompt_chunks_limit=prompt_chunks_limit,
        )
        new_probs = self.generate_chunks(
            slot_codes=eff_slots,
            num_chunks=num_chunks,
            sampling=sampling,
            prompt_chunks_f32=norm_prompt_vq,
        )
        full_probs = (
            np.concatenate([norm_prompt_vq, new_probs], axis=0)
            if norm_prompt_vq is not None and int(norm_prompt_vq.shape[0]) > 0
            else new_probs
        )
        meta = slot_codes_to_roll_meta(eff_slots)
        chunks_to_midi(full_probs, meta, out_p)
        return out_p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MORTM5 4-Track Conditional MIDI Generator")
    parser.add_argument("--ablation", type=str, default="#6", help="Ablation preset ('#1'..'#8' or JSON path)")
    parser.add_argument("--ckpt", type=str, default=None, help="Downstream MORTM5 checkpoint (.pth / .checkpoint.pt)")
    parser.add_argument("--vq_ckpt", type=str, default=None, help="MidiVision tokenizer checkpoint (.pth)")
    parser.add_argument(
        "--slots",
        nargs="+",
        default=["drums", "piano", "sax", "bass"],
        help="1..4 instrument slots (ignored when --prompt_midi is provided)",
    )
    parser.add_argument("--prompt_midi", type=str, default=None, help="Optional input MIDI file for continuation")
    parser.add_argument("--prompt_bars", type=int, default=8, help="Max opening bars to use from --prompt_midi")
    parser.add_argument("--bars", type=int, default=8, help="Number of bars to generate (1 bar = 8 chunks)")
    parser.add_argument("--chunks", type=int, default=None, help="Explicit number of chunks to generate")
    parser.add_argument("--temperature", type=float, default=0.95, help="Layer-0 coarse sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Layer-0 top-k")
    parser.add_argument("--top_p", type=float, default=0.95, help="Layer-0 nucleus top-p")
    parser.add_argument("--fine_temperature", type=float, default=0.50, help="Residual layers 1..L-1 temperature")
    parser.add_argument("--fine_top_k", type=int, default=8, help="Residual layers 1..L-1 top-k")
    parser.add_argument("--fine_top_p", type=float, default=0.85, help="Residual layers 1..L-1 top-p")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Layer-0 chunk repetition penalty")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / mps / cpu)")
    parser.add_argument("--out", type=str, default="out/generated_mortm5.mid", help="Output MIDI file path")
    args = parser.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    num_chunks = int(args.chunks) if args.chunks is not None else int(args.bars) * CHUNKS_PER_BAR
    sampling = Mortm5SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        fine_temperature=args.fine_temperature,
        fine_top_k=args.fine_top_k,
        fine_top_p=args.fine_top_p,
        repetition_penalty=args.repetition_penalty,
    )

    generator = Mortm5MidiGenerator.from_pretrained(
        ablation=args.ablation,
        ckpt_path=args.ckpt,
        vq_ckpt_path=args.vq_ckpt,
        device=args.device,
    )

    prompt_midi_path: Optional[str] = None
    max_prompt_chunks: Optional[int] = None
    if args.prompt_midi:
        max_prompt_chunks = max(1, int(args.prompt_bars) * CHUNKS_PER_BAR)
        prompt_midi_path = args.prompt_midi
        slot_codes = [SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
    else:
        slot_codes = parse_slot_spec(args.slots)

    out_path = generator.generate_to_midi(
        output_path=args.out,
        slot_codes=slot_codes,
        num_chunks=num_chunks,
        sampling=sampling,
        prompt_midi_path=prompt_midi_path,
        prompt_chunks_limit=max_prompt_chunks,
    )
    print(f"Saved generated MIDI -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
