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
    """Locate the most recently modified checkpoint for an ablation model under standard project directories."""
    sub = resolve_ablation_subdir(model_name=model_name, ablation_id=ablation_id)
    search_patterns = [
        f"out/mortm5_ablation/{sub}/MORTM5.{model_name}*_latest*.pth",
        f"out/mortm5_ablation/{sub}/MORTM5.{model_name}*.checkpoint.pt",
        f"out/mortm5_ablation/{sub}/*.pth",
        f"out/mortm5_ablation/{sub}/*.pt",
        f"out/mortm5_ablation/MORTM5.{model_name}*_latest*.pth",
        f"out/mortm5_ablation/MORTM5.{model_name}*.pth",
        f"checkpoints/mortm5_ablation/{sub}/MORTM5.{model_name}*_latest*.pth",
        f"checkpoints/mortm5_ablation/{sub}/MORTM5.{model_name}*.checkpoint.pt",
        f"checkpoints/mortm5_ablation/{sub}/*.pth",
        f"checkpoints/mortm5_ablation/{sub}/*.pt",
        f"checkpoints/mortm5_ablation/MORTM5.{model_name}*_latest*.pth",
    ]
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

        bs = max(1, int(batch_size))
        parts: List[Tensor] = []
        for start in range(0, n_chunks, bs):
            batch_np = chunks_f32[start : start + bs]
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
    """High-level service coordinating MORTM5 autoregressive generation and MidiVision decoding."""

    def __init__(
        self,
        vq_model: MidiVision,
        model: _BaseMORTM5Transformer,
        device: torch.device,
    ) -> None:
        self._vq_model = vq_model
        self._model = model
        self._device = device

    @property
    def vq_model(self) -> MidiVision:
        return self._vq_model

    @property
    def model(self) -> _BaseMORTM5Transformer:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

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

        tok_key = m_args.tokenizer_key or ("m4_g4l4" if m_args.vq_groups == 4 else "m3_g8l2")
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
    def generate_chunks(
        self,
        slot_codes: Sequence[int],
        num_chunks: int,
        sampling: Optional[Mortm5SamplingConfig] = None,
        prompt_chunks_f32: Optional[np.ndarray] = None,
        decode_batch_size: int = 64,
    ) -> np.ndarray:
        """Generate `num_chunks` new piano-roll chunks [num_chunks, 8, 128, 24] in float32 [0, 1]."""
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        cfg = sampling or Mortm5SamplingConfig()

        prompt_len = int(prompt_chunks_f32.shape[0]) if prompt_chunks_f32 is not None else 0
        prompt_seq = (
            Mortm5PromptEncoder.encode_chunks_to_prompt(
                prompt_chunks_f32,
                vq_model=self._vq_model,
                model=self._model,
                slot_codes=slot_codes,
                device=self._device,
            )
            if prompt_chunks_f32 is not None and prompt_len > 0
            else None
        )

        total_target_chunks = prompt_len + num_chunks
        seq = self._model.generate(
            num_chunks=total_target_chunks,
            prompt=prompt_seq,
            slot_codes=slot_codes,
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
        L = int(self._model.vq_layers)
        G = int(self._model.vq_groups)
        tokens_per_chunk = L * G
        new_rvq = rvq_all[prompt_len * tokens_per_chunk :].view(num_chunks, L, G)

        bs = max(1, int(decode_batch_size))
        decoded_parts: List[np.ndarray] = []
        for start in range(0, num_chunks, bs):
            logits = self._vq_model.decode_tokens(new_rvq[start : start + bs].to(self._device))
            decoded_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
        probs = np.concatenate(decoded_parts, axis=0)

        for slot_idx, code in enumerate(slot_codes):
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
    ) -> Path:
        new_probs = self.generate_chunks(
            slot_codes=slot_codes,
            num_chunks=num_chunks,
            sampling=sampling,
            prompt_chunks_f32=prompt_chunks_f32,
        )
        full_probs = (
            np.concatenate([prompt_chunks_f32, new_probs], axis=0)
            if prompt_chunks_f32 is not None and prompt_chunks_f32.shape[0] > 0
            else new_probs
        )
        meta = slot_codes_to_roll_meta(slot_codes)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
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

    prompt_chunks: Optional[np.ndarray] = None
    if args.prompt_midi:
        raw_chunks, roll_meta = midi_to_chunks(args.prompt_midi)
        max_prompt_chunks = max(1, int(args.prompt_bars) * CHUNKS_PER_BAR)
        prompt_chunks = raw_chunks[:max_prompt_chunks]
        slot_codes = [int(c) for c in slots_to_slot_codes(roll_meta.slots)]
    else:
        slot_codes = parse_slot_spec(args.slots)

    generator = Mortm5MidiGenerator.from_pretrained(
        ablation=args.ablation,
        ckpt_path=args.ckpt,
        vq_ckpt_path=args.vq_ckpt,
        device=args.device,
    )
    out_path = generator.generate_to_midi(
        output_path=args.out,
        slot_codes=slot_codes,
        num_chunks=num_chunks,
        sampling=sampling,
        prompt_chunks_f32=prompt_chunks,
    )
    print(f"Saved generated MIDI -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
