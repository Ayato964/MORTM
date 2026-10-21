"""
ConvertMidiVisionTokens.py

GigaMIDI (GMD) 等の MIDI ファイル群を 1 度だけ CPU でピアノロールチャンクに変換し、
GPU (Colab リモート GPU またはローカル CUDA/MPS) 上で MidiVision (M3: FineGroup G8L2 / M4: DeepRVQ G4L4)
の両モデルで同時にトークン化して、高速・省メモリな単一フラットバイナリデータベース (np.memmap 対応) を構築するスクリプト。

Colab ハイブリッド実行 (--remote_url 指定時):
  - Mac 側のマルチプロセス CPU ワーカーが MIDI を直接 uint8 チャンク [N_chunks, 8, 128, 24] に変換し、
    ワーカー内で即座に zlib 圧縮 (約 1/160 サイズ, 1曲 ~85KB) して親プロセスへ返すため、メモリ爆発・Pipe 詰まりが発生しない。
  - 圧縮チャンクを複数曲まとめて Colab GPU Worker へ送信し、Colab GPU (bfloat16) で M3/M4 同時に高速トークン化する。
  - 出力データセット一式 ({split}_tokens.bin, {split}_offsets.npy, {split}_manifest.jsonl, {split}_state.json) は
    Colab 側の Google Drive (/content/drive/MyDrive/MORTM5_tokens 等) にステージングバッファ経由で直接保存される。

出力構造 (各モデルディレクトリ配下):
  - {split}_tokens.bin     : uint16 フラットバイナリ (RVQ_TOKEN_OFFSET = 3 適用済み)
  - {split}_offsets.npy    : int64 [N_songs, 2] (各曲の start_token_offset, num_tokens)
  - {split}_manifest.jsonl : 各曲のメタデータ (song_idx, midi_idx, midi_path, num_chunks, ...)
  - {split}_state.json     : レジューム用チェックポイント状態

特殊トークン仕様:
  - PAD_TOKEN_ID     = 0 (<PAD>)
  - BOS_TOKEN_ID     = 1 (<BOS>)
  - EOS_TOKEN_ID     = 2 (<EOS>)
  - RVQ_TOKEN_OFFSET = 3 (RVQ トークンは 3 .. 3 + G*K - 1)
"""

import argparse
import gc
import io
import json
import multiprocessing as mp
import os
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

import mido
import numpy as np
import requests
from tqdm import tqdm

from mortm.models.modules.config import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MORTM5Args,
    PAD_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
)
from mortm.utils.convert_pianoroll import (
    BEATS_PER_BAR,
    CHUNK_STEPS,
    MAX_TOTAL_STEPS,
    STEPS_PER_BAR,
    TOTAL_CHANNELS,
    MidiTooLongError,
    _choose_slots,
    _lane_key,
    _quantize_note,
    extract_notes,
)

if TYPE_CHECKING:
    import torch
    from mortm.models.mortm5 import MidiVision


def _resolve_existing_path(candidates: List[str]) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DEFAULT_MODELS = {
    "m3_g8l2": {
        "config": "configs/models/mortm5/m3_finegroup_g8l2_d256_k2048.json",
        "ckpt": _resolve_existing_path([
            "out/models/mortm5/m3_finegroup_g8l2_d256_k2048/MORTM5.M3_FineGroup_G8L2_D256_K2048_latest.pth",
            "out/models/mortm5/M3/MORTM5.M3_FineGroup_G8L2_D256_K2048_latest.pth",
        ]),
        "drive_ckpt_names": [
            "MORTM5.M3_FineGroup_G8L2_D256_K2048_latest.pth",
            "MORTM5.M3_FineGroup_G8L2_D256_K2048.checkpoint.pt",
            "MORTM5.m3_g8l2_tokenizer_cached.pth",
        ],
    },
    "m4_g4l4": {
        "config": "configs/models/mortm5/m4_deeprvq_g4l4_d128_k2048.json",
        "ckpt": _resolve_existing_path([
            "out/models/mortm5/m4_deeprvq_g4l4_d128_k2048/MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth",
            "out/models/mortm5/M4/MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth",
        ]),
        "drive_ckpt_names": [
            "MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth",
            "MORTM5.M4_DeepRVQ_G4L4_D128_K2048.checkpoint.pt",
            "MORTM5.m4_g4l4_tokenizer_cached.pth",
        ],
    },
}

DEFAULT_SPLITS = {
    "val": _resolve_existing_path([
        "data/gmd/eval.json",
        "out/models/mortm5/eval_paths_MIDIVision.json",
        "out/models/mortm5/m2_compact_g4l2_d128_k2048/eval_paths_M2_Compact_G4L2_D128_K2048.json",
    ]),
    "train": _resolve_existing_path([
        "data/gmd/train.json",
        "out/models/mortm5/train_paths_MIDIVision.json",
        "out/models/mortm5/m2_compact_g4l2_d128_k2048/train_paths_M2_Compact_G4L2_D128_K2048.json",
    ]),
}


def extract_quantized_notes_array(path: str) -> Tuple[int, np.ndarray]:
    """
    MIDI ファイルを読み込み、量子化済みのノート配列 int32 [N_notes, 5] (slot, start, end, pitch, velocity) と
    総チャンク数 num_chunks を返す。
    Mac ワーカー側で巨大なピアノロール配列を確保しないため、長尺曲でもメモリ消費は数MB以下に収まり、
    zlib 圧縮後の通信サイズも 1曲あたり平均 ~14 KB (画像チャンク転送比で 1/6 以下) に激減する。
    """
    mid = mido.MidiFile(str(path))
    if mid.ticks_per_beat is None or mid.ticks_per_beat <= 0:
        raise ValueError(f"Invalid ticks_per_beat={mid.ticks_per_beat} in {path}")
    notes = extract_notes(mid)

    ticks_per_step = (mid.ticks_per_beat * BEATS_PER_BAR) / STEPS_PER_BAR
    selected_notes, _slots, lane_to_slot = _choose_slots(notes)

    encoded: List[Tuple[int, int, int, int, int]] = []
    max_end = 0

    for note in selected_notes:
        if note.is_drum:
            slot = 0
        else:
            key = _lane_key(note)
            slot = lane_to_slot.get(key)
            if slot is None:
                continue

        start, end, pitch, velocity = _quantize_note(note, ticks_per_step)
        if end <= 0 or start < 0 or pitch < 0 or pitch >= 128:
            continue
        encoded.append((slot, start, end, pitch, velocity))
        if end > max_end:
            max_end = end

    num_chunks = max(1, (max_end + CHUNK_STEPS - 1) // CHUNK_STEPS)
    total_steps = num_chunks * CHUNK_STEPS
    if total_steps > MAX_TOTAL_STEPS:
        raise MidiTooLongError(
            f"total_steps={total_steps} exceeds MAX_TOTAL_STEPS={MAX_TOTAL_STEPS}; path={path}"
        )

    if len(encoded) == 0:
        notes_arr = np.zeros((0, 5), dtype=np.int32)
    else:
        notes_arr = np.asarray(encoded, dtype=np.int32)
    return num_chunks, notes_arr


def rasterize_notes_to_chunks_u8(num_chunks: int, notes_arr: np.ndarray) -> np.ndarray:
    """
    量子化済みノート配列 int32 [N_notes, 5] から uint8 [num_chunks, 8, 128, 24] (値域 0..254) を構築する。
    """
    total_steps = num_chunks * CHUNK_STEPS
    flat_u8 = np.zeros((TOTAL_CHANNELS, total_steps, 128), dtype=np.uint8)
    for slot, start, end, pitch, velocity in notes_arr:
        onset_ch = int(slot) * 2
        sustain_ch = onset_ch + 1
        onset_val = np.float32(0.5 + 0.5 * (float(velocity) / 127.0))
        onset_u8 = np.uint8(np.rint(np.clip(onset_val, 0.0, 1.0) * np.float32(254.0)))
        if onset_u8 > flat_u8[onset_ch, start, pitch]:
            flat_u8[onset_ch, start, pitch] = onset_u8
        flat_u8[sustain_ch, start:end, pitch] = 254

    chunks_u8 = np.ascontiguousarray(
        flat_u8.reshape(TOTAL_CHANNELS, num_chunks, CHUNK_STEPS, 128).transpose(1, 0, 3, 2)
    )
    del flat_u8
    return chunks_u8


def midi_to_chunks_u8_direct(path: str) -> np.ndarray:
    num_chunks, notes_arr = extract_quantized_notes_array(path)
    return rasterize_notes_to_chunks_u8(num_chunks, notes_arr)


def _midi_worker(task: Tuple[int, str, int]) -> Tuple[int, str, int, Optional[bytes]]:
    """
    CPU ワーカー関数 (spawn プロセスで実行、PyTorch/CUDA 非ロード):
    MIDI ファイルを読み込み、量子化済みスパースノート配列 int32 [N_notes, 5] を zlib 圧縮して返す。
    1曲あたり平均 ~14 KB しか通信・メモリ消費しないため、Pipe 詰まり・OOM・トンネル帯域制限を完全に回避する。
    """
    midi_idx, midi_path, min_chunks = task
    try:
        n_chunks, notes_arr = extract_quantized_notes_array(midi_path)
        if n_chunks < min_chunks:
            return midi_idx, midi_path, 0, None
        comp_bytes = zlib.compress(notes_arr.tobytes(), level=1)
        return midi_idx, midi_path, int(n_chunks), comp_bytes
    except Exception:
        return midi_idx, midi_path, 0, None


class ModelWriterState:
    """
    各トークナイザーモデル (M3 / M4) の出力ファイルとオフセット状態を管理するクラス。
    ステージングバッファにより、細かい write/fsync の連発を避け、チェックポイント単位で一括フラッシュする。
    """

    def __init__(
        self,
        model_key: str,
        model: Any,
        args: MORTM5Args,
        out_dir: str,
        split: str,
        resume: bool = True,
    ):
        self.model_key = model_key
        self.model = model
        self.args = args
        self.split = split
        self.out_dir = Path(out_dir) / model_key
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.tokens_bin_path = self.out_dir / f"{split}_tokens.bin"
        self.offsets_npy_path = self.out_dir / f"{split}_offsets.npy"
        self.manifest_path = self.out_dir / f"{split}_manifest.jsonl"
        self.state_path = self.out_dir / f"{split}_state.json"

        self.next_midi_idx = 0
        self.valid_songs = 0
        self.skipped_songs = 0
        self.total_chunks = 0
        self.total_tokens = 0
        self.offsets_list: List[Tuple[int, int]] = []
        self.pending_bin_buf = bytearray()
        self.pending_manifest_lines: List[str] = []

        if resume and self.state_path.exists() and self.tokens_bin_path.exists() and self.offsets_npy_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.next_midi_idx = int(state["next_midi_idx"])
            self.valid_songs = int(state["valid_songs"])
            self.skipped_songs = int(state["skipped_songs"])
            self.total_chunks = int(state["total_chunks"])
            self.total_tokens = int(state["total_tokens"])

            loaded_offsets = np.load(self.offsets_npy_path)
            if loaded_offsets.ndim != 2 or loaded_offsets.shape[1] != 2 or len(loaded_offsets) < self.valid_songs:
                raise RuntimeError(
                    f"[{self.model_key}/{self.split}] Resume failed: offsets.npy has {len(loaded_offsets)} rows, "
                    f"but state.json requires at least {self.valid_songs} valid_songs. Run with --no_resume to rebuild."
                )
            loaded_offsets = loaded_offsets[: self.valid_songs]
            if self.valid_songs > 0:
                expected_end_tok = int(loaded_offsets[-1, 0] + loaded_offsets[-1, 1])
                if expected_end_tok != self.total_tokens:
                    raise RuntimeError(
                        f"[{self.model_key}/{self.split}] Resume failed: offsets end token ({expected_end_tok}) "
                        f"!= state total_tokens ({self.total_tokens}). Run with --no_resume to rebuild."
                    )
            elif self.total_tokens != 0:
                raise RuntimeError(
                    f"[{self.model_key}/{self.split}] Resume failed: valid_songs=0 but total_tokens={self.total_tokens}."
                )
            self.offsets_list = [(int(r[0]), int(r[1])) for r in loaded_offsets]

            expected_bytes = self.total_tokens * 2
            actual_bytes = os.path.getsize(self.tokens_bin_path)
            if actual_bytes < expected_bytes:
                raise RuntimeError(
                    f"[{self.model_key}/{self.split}] Resume failed: {self.tokens_bin_path} is truncated "
                    f"({actual_bytes} bytes < expected {expected_bytes} bytes). Run with --no_resume to rebuild."
                )
            if actual_bytes > expected_bytes:
                with open(self.tokens_bin_path, "r+b") as bf:
                    bf.truncate(expected_bytes)

            if self.manifest_path.exists():
                lines = []
                with open(self.manifest_path, "r", encoding="utf-8") as mf:
                    for _ in range(self.valid_songs):
                        line = mf.readline()
                        if not line:
                            raise RuntimeError(
                                f"[{self.model_key}/{self.split}] Resume failed: {self.manifest_path} has fewer than "
                                f"{self.valid_songs} lines. Run with --no_resume to rebuild."
                            )
                        lines.append(line)
                with open(self.manifest_path, "w", encoding="utf-8") as mf:
                    mf.writelines(lines)
            elif self.valid_songs > 0:
                raise RuntimeError(
                    f"[{self.model_key}/{self.split}] Resume failed: {self.manifest_path} missing while valid_songs={self.valid_songs}."
                )
        else:
            with open(self.tokens_bin_path, "wb"):
                pass
            with open(self.manifest_path, "w", encoding="utf-8"):
                pass

    def append_song(self, midi_idx: int, midi_path: str, num_chunks: int, song_tokens_u16: np.ndarray):
        if midi_idx < self.next_midi_idx:
            return
        if midi_idx > self.next_midi_idx:
            raise RuntimeError(
                f"[{self.model_key}/{self.split}] Out-of-order midi_idx={midi_idx}, expected={self.next_midi_idx}"
            )

        num_tokens = int(song_tokens_u16.size)
        start_offset = self.total_tokens

        self.pending_bin_buf.extend(song_tokens_u16.tobytes())
        self.offsets_list.append((start_offset, num_tokens))

        record = {
            "song_idx": self.valid_songs,
            "midi_idx": midi_idx,
            "midi_path": midi_path,
            "num_chunks": num_chunks,
            "num_tokens": num_tokens,
            "start_offset": start_offset,
        }
        self.pending_manifest_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

        self.valid_songs += 1
        self.total_chunks += num_chunks
        self.total_tokens += num_tokens
        self.next_midi_idx = midi_idx + 1

    def record_skip(self, midi_idx: int):
        if midi_idx < self.next_midi_idx:
            return
        if midi_idx > self.next_midi_idx:
            raise RuntimeError(
                f"[{self.model_key}/{self.split}] Out-of-order skip midi_idx={midi_idx}, expected={self.next_midi_idx}"
            )
        self.skipped_songs += 1
        self.next_midi_idx = midi_idx + 1

    def flush_checkpoint(self, completed: bool = False):
        if self.pending_bin_buf:
            with open(self.tokens_bin_path, "ab") as bf:
                bf.write(self.pending_bin_buf)
                bf.flush()
                os.fsync(bf.fileno())
            self.pending_bin_buf.clear()

        if self.pending_manifest_lines:
            with open(self.manifest_path, "a", encoding="utf-8") as mf:
                mf.writelines(self.pending_manifest_lines)
                mf.flush()
                os.fsync(mf.fileno())
            self.pending_manifest_lines.clear()

        if len(self.offsets_list) > 0:
            offsets_arr = np.asarray(self.offsets_list, dtype=np.int64)
        else:
            offsets_arr = np.zeros((0, 2), dtype=np.int64)

        tmp_offsets = str(self.offsets_npy_path) + ".tmp.npy"
        np.save(tmp_offsets, offsets_arr)
        os.replace(tmp_offsets, self.offsets_npy_path)

        state = {
            "model_key": self.model_key,
            "split": self.split,
            "completed": completed,
            "next_midi_idx": self.next_midi_idx,
            "valid_songs": self.valid_songs,
            "skipped_songs": self.skipped_songs,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "vq_groups": self.args.vq_groups,
            "vq_layers": self.args.vq_layers,
            "codebook_size": self.args.codebook_size,
            "tokens_per_chunk": self.args.vq_groups * self.args.vq_layers,
            "pad_token_id": self.args.pad_token_id,
            "bos_token_id": self.args.bos_token_id,
            "eos_token_id": self.args.eos_token_id,
            "token_offset": self.args.token_offset,
            "vocab_size": self.args.vocab_size,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        tmp_state = str(self.state_path) + ".tmp"
        with open(tmp_state, "w", encoding="utf-8") as sf:
            json.dump(state, sf, indent=2, ensure_ascii=False)
        os.replace(tmp_state, self.state_path)

    def close_files_only(self):
        pass

    def close(self, completed: bool = True):
        self.flush_checkpoint(completed=completed)


# ==============================================================================
# Colab Remote GPU Worker Hot-Patch & Client Logic (Google Drive Direct Save)
# ==============================================================================
TOKENIZER_HOT_PATCH = """
import io, json, os, time, zlib, threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from starlette.responses import JSONResponse

RVQ_TOKEN_OFFSET = 3
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

class _TokMidiVision(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = ResNetEncoder(args)
        self.vq_groups = getattr(args, "vq_groups", 4)
        self.num_quantizers = getattr(args, "encoder_layer", 2)
        self.codebook_dim = args.encoder_wout // self.vq_groups
        self.codebook_size = getattr(args, "codebook_size", 2048)
        self.token_offset = getattr(args, "token_offset", RVQ_TOKEN_OFFSET)
        self.pre_vq_norm = nn.LayerNorm(self.codebook_dim, elementwise_affine=False)
        self.rvq = GroupedResidualVQ(
            dim=args.encoder_wout,
            groups=self.vq_groups,
            codebook_dim=self.codebook_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            sync_codebook=False,
            threshold_ema_dead_code=0.05,
            rotation_trick=False,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.999,
        )
        self.use_pre_vq_norm = getattr(args, "use_pre_vq_norm", True)
        self.decoder = ResNetDecoder(args)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith("vq.") for k in state_dict.keys()):
            self.use_pre_vq_norm = False
            state_dict = {(k.replace("vq.", "rvq.", 1) if k.startswith("vq.") else k): v for k, v in state_dict.items()}
        elif "use_pre_vq_norm" not in self.__dict__:
            self.use_pre_vq_norm = True
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.inference_mode()
    def encode_tokens(self, x: torch.Tensor, include_offset: bool = True) -> torch.Tensor:
        z = self.encoder(x)
        if getattr(self, "use_pre_vq_norm", True):
            orig_shape = z.shape
            z = self.pre_vq_norm(z.reshape(*orig_shape[:-1], self.vq_groups, self.codebook_dim)).reshape(orig_shape)
        _, tokens, _ = self.rvq(z)
        offset_base = getattr(self, "token_offset", RVQ_TOKEN_OFFSET) if include_offset else 0
        group_offsets = (
            torch.arange(self.vq_groups, device=tokens.device, dtype=tokens.dtype) * self.codebook_size + offset_base
        )
        return (tokens.movedim(0, -1) + group_offsets).flatten(-2)


class _RemoteWriterState:
    def __init__(self, model_key: str, meta_cfg: dict, out_dir: str, split: str, resume: bool = True):
        self.model_key = model_key
        self.meta_cfg = meta_cfg
        self.split = split
        self.out_dir = Path(out_dir) / model_key
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.tokens_bin_path = self.out_dir / f"{split}_tokens.bin"
        self.offsets_npy_path = self.out_dir / f"{split}_offsets.npy"
        self.manifest_path = self.out_dir / f"{split}_manifest.jsonl"
        self.state_path = self.out_dir / f"{split}_state.json"

        self.next_midi_idx = 0
        self.valid_songs = 0
        self.skipped_songs = 0
        self.total_chunks = 0
        self.total_tokens = 0
        self.offsets_list = []
        self.pending_bin_buf = bytearray()
        self.pending_manifest_lines = []

        if resume and self.state_path.exists() and self.tokens_bin_path.exists() and self.offsets_npy_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.next_midi_idx = int(state["next_midi_idx"])
            self.valid_songs = int(state["valid_songs"])
            self.skipped_songs = int(state["skipped_songs"])
            self.total_chunks = int(state["total_chunks"])
            self.total_tokens = int(state["total_tokens"])

            loaded_offsets = np.load(self.offsets_npy_path)
            if loaded_offsets.ndim != 2 or loaded_offsets.shape[1] != 2 or len(loaded_offsets) < self.valid_songs:
                raise RuntimeError(f"[{self.model_key}/{self.split}] Resume offsets mismatch.")
            loaded_offsets = loaded_offsets[: self.valid_songs]
            if self.valid_songs > 0:
                expected_end_tok = int(loaded_offsets[-1, 0] + loaded_offsets[-1, 1])
                if expected_end_tok != self.total_tokens:
                    raise RuntimeError(f"[{self.model_key}/{self.split}] Resume total_tokens mismatch.")
            self.offsets_list = [(int(r[0]), int(r[1])) for r in loaded_offsets]

            expected_bytes = self.total_tokens * 2
            actual_bytes = os.path.getsize(self.tokens_bin_path)
            if actual_bytes < expected_bytes:
                raise RuntimeError(f"[{self.model_key}/{self.split}] tokens.bin truncated ({actual_bytes} < {expected_bytes}).")
            if actual_bytes > expected_bytes:
                with open(self.tokens_bin_path, "r+b") as bf:
                    bf.truncate(expected_bytes)

            if self.manifest_path.exists():
                lines = []
                with open(self.manifest_path, "r", encoding="utf-8") as mf:
                    for _ in range(self.valid_songs):
                        line = mf.readline()
                        if not line:
                            raise RuntimeError(f"[{self.model_key}/{self.split}] manifest.jsonl truncated.")
                        lines.append(line)
                with open(self.manifest_path, "w", encoding="utf-8") as mf:
                    mf.writelines(lines)
        else:
            with open(self.tokens_bin_path, "wb"):
                pass
            with open(self.manifest_path, "w", encoding="utf-8"):
                pass

    def append_song(self, midi_idx: int, midi_path: str, num_chunks: int, song_tokens_u16: np.ndarray):
        if midi_idx < self.next_midi_idx:
            return
        if midi_idx > self.next_midi_idx:
            raise RuntimeError(f"[{self.model_key}/{self.split}] Out-of-order midi_idx={midi_idx}, expected={self.next_midi_idx}")

        num_tokens = int(song_tokens_u16.size)
        start_offset = self.total_tokens
        self.pending_bin_buf.extend(song_tokens_u16.tobytes())
        self.offsets_list.append((start_offset, num_tokens))

        record = {
            "song_idx": self.valid_songs,
            "midi_idx": midi_idx,
            "midi_path": midi_path,
            "num_chunks": num_chunks,
            "num_tokens": num_tokens,
            "start_offset": start_offset,
        }
        self.pending_manifest_lines.append(json.dumps(record, ensure_ascii=False) + "\\n")

        self.valid_songs += 1
        self.total_chunks += num_chunks
        self.total_tokens += num_tokens
        self.next_midi_idx = midi_idx + 1

    def record_skip(self, midi_idx: int):
        if midi_idx < self.next_midi_idx:
            return
        if midi_idx > self.next_midi_idx:
            raise RuntimeError(f"[{self.model_key}/{self.split}] Out-of-order skip midi_idx={midi_idx}, expected={self.next_midi_idx}")
        self.skipped_songs += 1
        self.next_midi_idx = midi_idx + 1

    def get_state_dict(self, completed: bool = False) -> dict:
        return {
            "model_key": self.model_key,
            "split": self.split,
            "completed": completed,
            "next_midi_idx": self.next_midi_idx,
            "valid_songs": self.valid_songs,
            "skipped_songs": self.skipped_songs,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "vq_groups": int(self.meta_cfg["vq_groups"]),
            "vq_layers": int(self.meta_cfg["vq_layers"]),
            "codebook_size": int(self.meta_cfg["codebook_size"]),
            "tokens_per_chunk": int(self.meta_cfg["vq_groups"]) * int(self.meta_cfg["vq_layers"]),
            "pad_token_id": PAD_TOKEN_ID,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "token_offset": RVQ_TOKEN_OFFSET,
            "vocab_size": int(self.meta_cfg["vocab_size"]),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def flush_checkpoint(self, completed: bool = False) -> dict:
        if self.pending_bin_buf:
            with open(self.tokens_bin_path, "ab") as bf:
                bf.write(self.pending_bin_buf)
                bf.flush()
                os.fsync(bf.fileno())
            self.pending_bin_buf.clear()

        if self.pending_manifest_lines:
            with open(self.manifest_path, "a", encoding="utf-8") as mf:
                mf.writelines(self.pending_manifest_lines)
                mf.flush()
                os.fsync(mf.fileno())
            self.pending_manifest_lines.clear()

        if len(self.offsets_list) > 0:
            offsets_arr = np.asarray(self.offsets_list, dtype=np.int64)
        else:
            offsets_arr = np.zeros((0, 2), dtype=np.int64)

        tmp_offsets = str(self.offsets_npy_path) + ".tmp.npy"
        np.save(tmp_offsets, offsets_arr)
        os.replace(tmp_offsets, self.offsets_npy_path)

        state = self.get_state_dict(completed=completed)
        tmp_state = str(self.state_path) + ".tmp"
        with open(tmp_state, "w", encoding="utf-8") as sf:
            json.dump(state, sf, indent=2, ensure_ascii=False)
        os.replace(tmp_state, self.state_path)
        return state


def _clean_sd(sd):
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        cleaned[nk] = v
    return cleaned


async def _tok_probe_models(request):
    body = await request.json()
    models_spec = body.get("models", {})
    tok_models = STATE.setdefault("tok_models", {})
    tok_metas = STATE.setdefault("tok_metas", {})

    # Free training model VRAM if present
    if STATE.get("model") is not None:
        STATE["model"] = None
        STATE["optimizer"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = []
    loaded_from = {}
    search_dirs = ["/content/drive/MyDrive/MORTM5_checkpoints", "/content"]

    for m_key, spec in models_spec.items():
        cfg_dict = spec["config_dict"]
        meta_cfg = spec["meta_cfg"]
        tok_metas[m_key] = meta_cfg
        if m_key in tok_models and tok_models[m_key] is not None:
            loaded_from[m_key] = "already_in_vram"
            continue

        found_path = None
        for d in search_dirs:
            for fname in spec.get("drive_ckpt_names", []):
                cand = os.path.join(d, fname)
                if os.path.exists(cand):
                    found_path = cand
                    break
            if found_path:
                break

        if found_path:
            args = SimpleNamespace(**cfg_dict)
            m = _TokMidiVision(args)
            raw_sd = torch.load(found_path, map_location="cpu", weights_only=False)
            m.load_state_dict(_clean_sd(raw_sd), strict=True)
            m.to(DEVICE).eval()
            tok_models[m_key] = m
            loaded_from[m_key] = found_path
        else:
            missing.append(m_key)

    return JSONResponse({
        "status": "ok",
        "loaded_from": loaded_from,
        "missing": missing,
        "drive_mounted": os.path.exists("/content/drive/MyDrive"),
        "device": str(DEVICE),
    })


def _rasterize_sparse_notes_u8(num_chunks: int, notes_arr: np.ndarray) -> np.ndarray:
    total_steps = num_chunks * 24
    flat_u8 = np.zeros((8, total_steps, 128), dtype=np.uint8)
    for slot, start, end, pitch, velocity in notes_arr:
        onset_ch = int(slot) * 2
        sustain_ch = onset_ch + 1
        onset_val = np.float32(0.5 + 0.5 * (float(velocity) / 127.0))
        onset_u8 = np.uint8(np.rint(np.clip(onset_val, 0.0, 1.0) * np.float32(254.0)))
        if onset_u8 > flat_u8[onset_ch, start, pitch]:
            flat_u8[onset_ch, start, pitch] = onset_u8
        flat_u8[sustain_ch, start:end, pitch] = 254
    chunks_u8 = np.ascontiguousarray(
        flat_u8.reshape(8, num_chunks, 24, 128).transpose(1, 0, 3, 2)
    )
    del flat_u8
    return chunks_u8


async def _tok_upload_model(request):
    m_key = request.query_params["model_key"]
    raw = await request.body()
    if len(raw) == 0 and STATE.get("tok_upload_parts"):
        raw = b"".join(STATE["tok_upload_parts"])
        STATE["tok_upload_parts"] = []
    payload = torch.load(io.BytesIO(zlib.decompress(raw)), map_location="cpu", weights_only=False)
    cfg_dict = payload["config_dict"]
    meta_cfg = payload["meta_cfg"]
    sd = _clean_sd(payload["state_dict"])

    args = SimpleNamespace(**cfg_dict)
    m = _TokMidiVision(args)
    m.load_state_dict(sd, strict=False)
    m.to(DEVICE).eval()

    if os.path.exists("/content"):
        cache_name = f"MORTM5.{m_key}_tokenizer_cached.pth"
        try:
            torch.save(m.state_dict(), os.path.join("/content", cache_name))
        except Exception:
            pass

    STATE.setdefault("tok_models", {})[m_key] = m
    STATE.setdefault("tok_metas", {})[m_key] = meta_cfg
    return JSONResponse({"status": "uploaded_and_loaded", "model_key": m_key, "device": str(DEVICE)})


async def _tok_init_split(request):
    body = await request.json()
    out_dir = body["out_dir"]
    split = body["split"]
    model_keys = body["models"]
    resume = bool(body.get("resume", True))
    require_drive = bool(body.get("require_drive", True))

    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if require_drive and out_dir.startswith("/content/drive") and not drive_mounted:
        return JSONResponse(
            {"status": "error", "error": "Google Drive (/content/drive/MyDrive) is not mounted on Colab!"},
            status_code=400,
        )

    tok_metas = STATE["tok_metas"]
    writers = {}
    states_info = {}
    for m_key in model_keys:
        w = _RemoteWriterState(
            model_key=m_key,
            meta_cfg=tok_metas[m_key],
            out_dir=out_dir,
            split=split,
            resume=resume,
        )
        writers[m_key] = w
        states_info[m_key] = w.get_state_dict(completed=False)

    STATE["tok_writers"] = writers
    STATE["tok_upload_parts"] = []
    start_idx = min(w.next_midi_idx for w in writers.values())
    return JSONResponse({
        "status": "ok",
        "out_dir": out_dir,
        "split": split,
        "start_idx": start_idx,
        "states": states_info,
        "drive_mounted": drive_mounted,
    })


async def _tok_clear_upload(request):
    STATE["tok_upload_parts"] = []
    return JSONResponse({"status": "cleared"})


async def _tok_append_upload(request):
    chunk = await request.body()
    STATE.setdefault("tok_upload_parts", []).append(chunk)
    return JSONResponse({"parts": len(STATE["tok_upload_parts"]), "bytes": len(chunk)})


async def _tok_process_batch(request):
    raw = await request.body()
    if len(raw) == 0 and STATE.get("tok_upload_parts"):
        raw = b"".join(STATE["tok_upload_parts"])
        STATE["tok_upload_parts"] = []

    gpu_batch_chunks = int(request.query_params.get("gpu_batch_chunks", "1024"))
    flush_ckpt = request.query_params.get("flush", "0") == "1"

    header_len = int.from_bytes(raw[:4], byteorder="big")
    header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
    blob = raw[4 + header_len :]

    events = header["events"]  # list of ["skip", midi_idx] or ["song", midi_idx, midi_path, n_chunks, comp_len]
    tok_models = STATE["tok_models"]
    writers = STATE["tok_writers"]
    model_keys = list(writers.keys())

    # Rasterize sparse quantized notes into uint8 chunks for a single coalesced GPU pass
    song_arrays = []
    cursor_b = 0
    for ev in events:
        if ev[0] == "song":
            _tag, _midx, _mpath, n_chunks, comp_len = ev
            comp_slice = blob[cursor_b : cursor_b + comp_len]
            cursor_b += comp_len
            decomp = zlib.decompress(comp_slice)
            notes_arr = np.frombuffer(decomp, dtype=np.int32).reshape(-1, 5)
            arr_u8 = _rasterize_sparse_notes_u8(int(n_chunks), notes_arr)
            song_arrays.append(arr_u8)

    model_token_buffers = {k: [] for k in model_keys}
    if song_arrays:
        all_chunks_u8 = np.concatenate(song_arrays, axis=0)
        total_chunks = len(all_chunks_u8)
        del song_arrays

        with torch.inference_mode():
            for start in range(0, total_chunks, gpu_batch_chunks):
                end = min(start + gpu_batch_chunks, total_chunks)
                sub_u8 = torch.from_numpy(all_chunks_u8[start:end]).to(device=DEVICE, non_blocking=True)
                x = sub_u8.to(dtype=torch.float32).div_(254.0)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda")):
                    for m_key in model_keys:
                        toks = tok_models[m_key].encode_tokens(x, include_offset=True)
                        toks_u16 = toks.to(dtype=torch.int32).cpu().numpy().astype(np.uint16)
                        model_token_buffers[m_key].append(toks_u16)
        del all_chunks_u8

    full_tokens_map = {
        m_key: (np.concatenate(model_token_buffers[m_key], axis=0) if model_token_buffers[m_key] else None)
        for m_key in model_keys
    }

    chunk_cursor = 0
    for ev in events:
        if ev[0] == "skip":
            midx = int(ev[1])
            for w in writers.values():
                w.record_skip(midx)
        else:
            _tag, midx, mpath, n_chunks, _comp_len = ev
            midx = int(midx)
            n_chunks = int(n_chunks)
            for m_key, w in writers.items():
                song_toks = full_tokens_map[m_key][chunk_cursor : chunk_cursor + n_chunks].reshape(-1)
                w.append_song(midx, mpath, n_chunks, song_toks)
            chunk_cursor += n_chunks

    # Auto-flush if staging buffer exceeds 32 MB or explicit flush requested
    any_large_buf = any(len(w.pending_bin_buf) >= 32 * 1024 * 1024 for w in writers.values())
    flushed = False
    states_out = {}
    if flush_ckpt or any_large_buf:
        for m_key, w in writers.items():
            states_out[m_key] = w.flush_checkpoint(completed=False)
        flushed = True
    else:
        for m_key, w in writers.items():
            states_out[m_key] = w.get_state_dict(completed=False)

    return JSONResponse({"status": "ok", "flushed": flushed, "states": states_out})


async def _tok_finish_split(request):
    writers = STATE.get("tok_writers", {})
    states_out = {}
    sizes_mb = {}
    for m_key, w in writers.items():
        states_out[m_key] = w.flush_checkpoint(completed=True)
        sizes_mb[m_key] = round(os.path.getsize(w.tokens_bin_path) / (1024 * 1024), 2)
    return JSONResponse({"status": "completed", "states": states_out, "sizes_mb": sizes_mb})


app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") not in {
    "/tok_probe_models", "/tok_upload_model", "/tok_init_split",
    "/tok_clear_upload", "/tok_append_upload", "/tok_process_batch", "/tok_finish_split"
}]
app.add_route("/tok_probe_models", _tok_probe_models, methods=["POST"])
app.add_route("/tok_upload_model", _tok_upload_model, methods=["POST"])
app.add_route("/tok_init_split", _tok_init_split, methods=["POST"])
app.add_route("/tok_clear_upload", _tok_clear_upload, methods=["POST"])
app.add_route("/tok_append_upload", _tok_append_upload, methods=["POST"])
app.add_route("/tok_process_batch", _tok_process_batch, methods=["POST"])
app.add_route("/tok_finish_split", _tok_finish_split, methods=["POST"])
"""


class _TokenizerHotPatch:
    def __reduce__(self):
        return (eval, (f"exec({TOKENIZER_HOT_PATCH!r}, __import__('colab_gpu_worker').__dict__)",))


def init_remote_colab_worker(
    session: requests.Session,
    remote_url: str,
    model_keys: List[str],
) -> Dict[str, MORTM5Args]:
    print(f"[RemoteColab] Connecting to {remote_url} ...", flush=True)
    r = session.get(f"{remote_url}/health", timeout=30)
    r.raise_for_status()
    health = r.json()
    print(
        f"[RemoteColab] Connected! GPU: {health.get('gpu_name')} | VRAM: {health.get('vram_gb')} GB | Device: {health.get('device')}",
        flush=True,
    )

    # 1. Inject Tokenizer Hot-Patch via /init
    first_cfg_path = DEFAULT_MODELS[model_keys[0]]["config"]
    with open(first_cfg_path, "r", encoding="utf-8") as f:
        dummy_cfg = json.load(f)

    init_buf = io.BytesIO()
    import torch

    torch.save(
        {
            "patch": _TokenizerHotPatch(),
            "model_config": dummy_cfg,
            "lr": 1e-4,
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "optimizer_step": 0,
        },
        init_buf,
    )
    init_comp = zlib.compress(init_buf.getvalue(), level=1)
    r_patch = session.post(
        f"{remote_url}/init",
        data=init_comp,
        headers={"Content-Type": "application/octet-stream"},
        timeout=60,
    )
    r_patch.raise_for_status()

    # 2. Probe which models are already on Colab Drive / SSD / VRAM
    models_spec = {}
    args_map: Dict[str, MORTM5Args] = {}
    for m_key in model_keys:
        cfg_path = DEFAULT_MODELS[m_key]["config"]
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        m_args = MORTM5Args(cfg_path)
        args_map[m_key] = m_args
        models_spec[m_key] = {
            "config_dict": cfg_dict,
            "drive_ckpt_names": DEFAULT_MODELS[m_key].get("drive_ckpt_names", []),
            "meta_cfg": {
                "vq_groups": m_args.vq_groups,
                "vq_layers": m_args.vq_layers,
                "codebook_size": m_args.codebook_size,
                "vocab_size": m_args.vocab_size,
            },
        }

    r_probe = session.post(f"{remote_url}/tok_probe_models", json={"models": models_spec}, timeout=90)
    r_probe.raise_for_status()
    probe_info = r_probe.json()
    print(
        f"[RemoteColab] Probe result: loaded={probe_info.get('loaded_from')}, "
        f"missing={probe_info.get('missing')}, drive_mounted={probe_info.get('drive_mounted')}",
        flush=True,
    )

    # 3. Upload any missing model weights from Mac (encoder+rvq only, 4MB chunked to bypass Cloudflare limits)
    for m_key in probe_info.get("missing", []):
        ckpt_path = DEFAULT_MODELS[m_key]["ckpt"]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint for {m_key} not found on Mac: {ckpt_path}")
        raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
            raw_sd = raw_sd["model_state_dict"]
        elif isinstance(raw_sd, dict) and "model" in raw_sd:
            raw_sd = raw_sd["model"]

        # Strip decoder.* weights since encode_tokens() only uses encoder, pre_vq_norm, and rvq
        enc_sd = {k: v for k, v in raw_sd.items() if not k.startswith("decoder.") and not k.startswith("module.decoder.")}

        up_buf = io.BytesIO()
        torch.save(
            {
                "config_dict": models_spec[m_key]["config_dict"],
                "meta_cfg": models_spec[m_key]["meta_cfg"],
                "state_dict": enc_sd,
            },
            up_buf,
        )
        up_comp = zlib.compress(up_buf.getvalue(), level=1)
        print(f"[RemoteColab] Uploading {m_key} encoder+rvq weights ({len(up_comp) / 1e6:.2f} MB in 4MB chunks)...", flush=True)
        session.post(f"{remote_url}/tok_clear_upload", timeout=30).raise_for_status()
        chunk_sz = 4 * 1024 * 1024
        for st in range(0, len(up_comp), chunk_sz):
            part = up_comp[st : st + chunk_sz]
            session.post(
                f"{remote_url}/tok_append_upload",
                data=part,
                headers={"Content-Type": "application/octet-stream"},
                timeout=60,
            ).raise_for_status()
        r_up = session.post(
            f"{remote_url}/tok_upload_model?model_key={m_key}",
            data=b"",
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        )
        r_up.raise_for_status()
        print(f"[RemoteColab] {m_key} ready on Colab GPU!", flush=True)

    return args_map


def _pack_remote_batch(events_with_bytes: List[Tuple[str, int, str, int, Optional[bytes]]]) -> bytes:
    events_meta = []
    blobs = []
    for tag, midx, mpath, n_chunks, comp_b in events_with_bytes:
        if tag == "skip":
            events_meta.append(["skip", midx])
        else:
            assert comp_b is not None
            events_meta.append(["song", midx, mpath, n_chunks, len(comp_b)])
            blobs.append(comp_b)
    header_bytes = json.dumps({"events": events_meta}, ensure_ascii=False).encode("utf-8")
    header_len = len(header_bytes).to_bytes(4, byteorder="big")
    return header_len + header_bytes + b"".join(blobs)


def _send_remote_batch(
    session: requests.Session,
    remote_url: str,
    payload: bytes,
    gpu_batch_chunks: int,
    flush_ckpt: bool,
) -> Dict[str, Any]:
    flush_flag = "1" if flush_ckpt else "0"
    max_single_post = 4 * 1024 * 1024  # 4 MB safe limit for Cloudflare Tunnel

    for attempt in range(6):
        try:
            if len(payload) <= max_single_post:
                r = session.post(
                    f"{remote_url}/tok_process_batch?gpu_batch_chunks={gpu_batch_chunks}&flush={flush_flag}",
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=180,
                )
                r.raise_for_status()
                return r.json()
            else:
                r_clr = session.post(f"{remote_url}/tok_clear_upload", timeout=30)
                r_clr.raise_for_status()
                chunk_sz = 4 * 1024 * 1024
                for st in range(0, len(payload), chunk_sz):
                    part = payload[st : st + chunk_sz]
                    r_ap = session.post(
                        f"{remote_url}/tok_append_upload",
                        data=part,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=90,
                    )
                    r_ap.raise_for_status()
                r = session.post(
                    f"{remote_url}/tok_process_batch?gpu_batch_chunks={gpu_batch_chunks}&flush={flush_flag}",
                    data=b"",
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=240,
                )
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if attempt == 5:
                raise
            print(f"\n  [RemoteColab Retry {attempt + 1}/5] {e}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def _bounded_midi_iter(tasks: List[Tuple[int, str, int]], num_workers: int, pool: Optional[Any]):
    """
    spawn ワーカープールから順序保証付きで (midi_idx, midi_path, n_chunks, comp_bytes) を取得するジェネレータ。
    万一ワーカーがタイムアウトまたはクラッシュしても永久ハングせず安全にスキップする。
    """
    from collections import deque

    if pool is None:
        for t in tasks:
            yield _midi_worker(t)
        return

    max_in_flight = max(8, num_workers * 3)
    in_flight = deque()
    next_pos = 0
    n_tasks = len(tasks)

    while next_pos < min(n_tasks, max_in_flight):
        in_flight.append((tasks[next_pos], pool.apply_async(_midi_worker, (tasks[next_pos],))))
        next_pos += 1

    while in_flight:
        task_info, res_future = in_flight.popleft()
        if next_pos < n_tasks:
            in_flight.append((tasks[next_pos], pool.apply_async(_midi_worker, (tasks[next_pos],))))
            next_pos += 1
        try:
            yield res_future.get(timeout=90)
        except Exception as e:
            midx, mpath, _ = task_info
            print(f"\n  [Worker Warning] Skipping midi_idx={midx} ({mpath}) due to worker error/timeout: {e}", flush=True)
            yield midx, mpath, 0, None


def process_split_remote(
    session: requests.Session,
    remote_url: str,
    remote_out_dir: str,
    local_state_dir: Optional[str],
    split: str,
    json_path: str,
    model_keys: List[str],
    resume: bool = True,
    require_drive: bool = True,
    num_workers: int = 8,
    min_chunks: int = 8,
    gpu_batch_chunks: int = 1024,
    remote_batch_chunks: int = 4096,
    checkpoint_every: int = 1000,
    limit: Optional[int] = None,
):
    with open(json_path, "r", encoding="utf-8") as f:
        midi_paths: List[str] = json.load(f)

    if limit is not None and limit > 0:
        midi_paths = midi_paths[:limit]

    r_init = session.post(
        f"{remote_url}/tok_init_split",
        json={
            "out_dir": remote_out_dir,
            "split": split,
            "models": model_keys,
            "resume": resume,
            "require_drive": require_drive,
        },
        timeout=60,
    )
    if r_init.status_code != 200:
        raise RuntimeError(f"Failed to initialize split '{split}' on Colab: {r_init.text}")
    init_info = r_init.json()
    start_idx = int(init_info["start_idx"])
    total_files = len(midi_paths)

    if start_idx >= total_files:
        print(f"[{split}] Already completed on Google Drive ({start_idx}/{total_files} files). Skipping.", flush=True)
        return

    print(
        f"[{split}] Processing {total_files - start_idx:,} / {total_files:,} MIDI files "
        f"(starting from index {start_idx}) -> Colab GPU -> {remote_out_dir}",
        flush=True,
    )

    tasks = [(idx, midi_paths[idx], min_chunks) for idx in range(start_idx, total_files)]
    pending_events: List[Tuple[str, int, str, int, Optional[bytes]]] = []
    pending_chunks = 0
    processed_since_ckpt = 0

    pbar = tqdm(total=total_files, initial=start_idx, desc=f"Colab Tokenizing [{split}]", unit="song", dynamic_ncols=True)

    if num_workers > 0:
        mp_ctx = mp.get_context("spawn")
        pool = mp_ctx.Pool(processes=num_workers, maxtasksperchild=250)
    else:
        pool = None

    sender_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="colab_sender")
    in_flight_send = None

    def _mirror_local_states(states_dict: Dict[str, Any]):
        if not local_state_dir:
            return
        for m_key, st in states_dict.items():
            mdir = Path(local_state_dir) / m_key
            mdir.mkdir(parents=True, exist_ok=True)
            spath = mdir / f"{split}_state.json"
            tmp_s = str(spath) + ".tmp"
            with open(tmp_s, "w", encoding="utf-8") as sf:
                json.dump(st, sf, indent=2, ensure_ascii=False)
            os.replace(tmp_s, spath)

    def _wait_previous_send():
        nonlocal in_flight_send
        if in_flight_send is None:
            return
        resp = in_flight_send.result()
        in_flight_send = None
        states = resp.get("states", {})
        if resp.get("flushed"):
            _mirror_local_states(states)
        if states:
            st0 = states[model_keys[0]]
            pbar.set_postfix(
                valid=st0["valid_songs"],
                skip=st0["skipped_songs"],
                M_tok=f"{st0['total_tokens'] / 1e6:.2f}M",
                drive="ok" if resp.get("flushed") else "buf",
            )

    try:
        for midi_idx, midi_path, n_chunks, comp_bytes in _bounded_midi_iter(tasks, num_workers, pool):
            if comp_bytes is None:
                pending_events.append(("skip", midi_idx, midi_path, 0, None))
            else:
                pending_events.append(("song", midi_idx, midi_path, n_chunks, comp_bytes))
                pending_chunks += n_chunks

            processed_since_ckpt += 1
            pbar.update(1)

            need_ckpt = processed_since_ckpt >= checkpoint_every
            if pending_chunks >= remote_batch_chunks or (need_ckpt and len(pending_events) >= 64):
                _wait_previous_send()
                payload = _pack_remote_batch(pending_events)
                flush_now = need_ckpt
                if flush_now:
                    processed_since_ckpt = 0
                in_flight_send = sender_exec.submit(
                    _send_remote_batch,
                    session,
                    remote_url,
                    payload,
                    gpu_batch_chunks,
                    flush_now,
                )
                pending_events.clear()
                pending_chunks = 0

        if pending_events:
            _wait_previous_send()
            payload = _pack_remote_batch(pending_events)
            in_flight_send = sender_exec.submit(
                _send_remote_batch,
                session,
                remote_url,
                payload,
                gpu_batch_chunks,
                True,
            )
            pending_events.clear()
            pending_chunks = 0

        _wait_previous_send()

        r_fin = session.post(f"{remote_url}/tok_finish_split", timeout=120)
        r_fin.raise_for_status()
        fin_info = r_fin.json()
        _mirror_local_states(fin_info.get("states", {}))

    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise

    finally:
        pbar.close()
        sender_exec.shutdown(wait=False, cancel_futures=True)
        if pool is not None:
            pool.close()
            pool.join()

    for m_key in model_keys:
        st = fin_info["states"][m_key]
        sz_mb = fin_info["sizes_mb"][m_key]
        print(
            f"  -> [GoogleDrive: {remote_out_dir}/{m_key} / {split}] "
            f"valid_songs={st['valid_songs']:,}, skipped={st['skipped_songs']:,}, "
            f"total_chunks={st['total_chunks']:,}, total_tokens={st['total_tokens']:,} ({sz_mb:.2f} MB)",
            flush=True,
        )


def load_midivision_model(config_path: str, ckpt_path: str, device: Any) -> Tuple[Any, MORTM5Args]:
    import torch
    from mortm.models.mortm5 import MidiVision

    args = MORTM5Args(config_path)
    model = MidiVision(args)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model, args


def flush_gpu_batch(
    pending_songs: List[Tuple[int, str, int, bytes]],
    writers: List[ModelWriterState],
    device: Any,
    gpu_batch_chunks: int,
    use_bf16: bool = True,
):
    if not pending_songs:
        return
    import torch

    song_arrays = [
        rasterize_notes_to_chunks_u8(
            item[2],
            np.frombuffer(zlib.decompress(item[3]), dtype=np.int32).reshape(-1, 5),
        )
        for item in pending_songs
    ]
    all_chunks_u8 = np.concatenate(song_arrays, axis=0)
    del song_arrays
    total_chunks = len(all_chunks_u8)
    song_lengths = [item[2] for item in pending_songs]

    model_token_buffers: Dict[str, List[np.ndarray]] = {w.model_key: [] for w in writers}

    with torch.inference_mode():
        for start in range(0, total_chunks, gpu_batch_chunks):
            end = min(start + gpu_batch_chunks, total_chunks)
            sub_u8 = torch.from_numpy(all_chunks_u8[start:end]).to(device=device, non_blocking=True)
            x = sub_u8.to(dtype=torch.float32) / 254.0

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(use_bf16 and device.type == "cuda")):
                for w in writers:
                    tokens = w.model.encode_tokens(x, include_offset=True)
                    tokens_u16 = tokens.to(dtype=torch.int32).cpu().numpy().astype(np.uint16)
                    model_token_buffers[w.model_key].append(tokens_u16)

    for w in writers:
        full_tokens_u16 = np.concatenate(model_token_buffers[w.model_key], axis=0)
        cursor = 0
        for (midi_idx, midi_path, n_chunks, _comp_b), _ in zip(pending_songs, song_lengths):
            song_tokens = full_tokens_u16[cursor : cursor + n_chunks].reshape(-1)
            w.append_song(midi_idx, midi_path, n_chunks, song_tokens)
            cursor += n_chunks


def process_split(
    split: str,
    json_path: str,
    writers: List[ModelWriterState],
    device: Any,
    num_workers: int = 8,
    min_chunks: int = 1,
    gpu_batch_chunks: int = 512,
    checkpoint_every: int = 1000,
    limit: Optional[int] = None,
):
    with open(json_path, "r", encoding="utf-8") as f:
        midi_paths: List[str] = json.load(f)

    if limit is not None and limit > 0:
        midi_paths = midi_paths[:limit]

    total_files = len(midi_paths)
    start_idx = min(w.next_midi_idx for w in writers)
    if start_idx >= total_files:
        for w in writers:
            w.close_files_only()
        print(f"[{split}] Already completed ({start_idx}/{total_files} files). Skipping.")
        return

    print(
        f"[{split}] Processing {total_files - start_idx:,} / {total_files:,} MIDI files "
        f"(starting from index {start_idx}) with {num_workers} CPU workers (spawn)..."
    )

    tasks = [(idx, midi_paths[idx], min_chunks) for idx in range(start_idx, total_files)]
    pending_songs: List[Tuple[int, str, int, bytes]] = []
    pending_chunks = 0
    processed_since_ckpt = 0

    pbar = tqdm(total=total_files, initial=start_idx, desc=f"Tokenizing [{split}]", unit="song", dynamic_ncols=True)

    if num_workers > 0:
        mp_ctx = mp.get_context("spawn")
        pool = mp_ctx.Pool(processes=num_workers, maxtasksperchild=250)
    else:
        pool = None

    try:
        for midi_idx, midi_path, n_chunks, comp_bytes in _bounded_midi_iter(tasks, num_workers, pool):
            if comp_bytes is None:
                if pending_songs:
                    flush_gpu_batch(pending_songs, writers, device, gpu_batch_chunks)
                    pending_songs.clear()
                    pending_chunks = 0
                for w in writers:
                    w.record_skip(midi_idx)
            else:
                pending_songs.append((midi_idx, midi_path, n_chunks, comp_bytes))
                pending_chunks += n_chunks

                if pending_chunks >= gpu_batch_chunks:
                    flush_gpu_batch(pending_songs, writers, device, gpu_batch_chunks)
                    pending_songs.clear()
                    pending_chunks = 0

            processed_since_ckpt += 1
            pbar.update(1)

            if processed_since_ckpt >= checkpoint_every and not pending_songs:
                for w in writers:
                    w.flush_checkpoint(completed=False)
                processed_since_ckpt = 0
                w0 = writers[0]
                pbar.set_postfix(
                    valid=w0.valid_songs,
                    skip=w0.skipped_songs,
                    M_tok=f"{w0.total_tokens / 1e6:.2f}M",
                )

        if pending_songs:
            flush_gpu_batch(pending_songs, writers, device, gpu_batch_chunks)
            pending_songs.clear()
            pending_chunks = 0

        for w in writers:
            w.close(completed=True)

    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise

    finally:
        pbar.close()
        if pool is not None:
            pool.close()
            pool.join()
        for w in writers:
            w.close_files_only()

    for w in writers:
        size_mb = os.path.getsize(w.tokens_bin_path) / (1024 * 1024)
        print(
            f"  -> [{w.model_key} / {split}] valid_songs={w.valid_songs:,}, "
            f"skipped={w.skipped_songs:,}, total_chunks={w.total_chunks:,}, "
            f"total_tokens={w.total_tokens:,} ({size_mb:.2f} MB)"
        )


def run_selftest(device: Any):
    """
    M3 / M4 両モデルの encode_tokens / decode_tokens と、
    MORTM5MemmapDataset (<PAD>=0, <BOS>=1, <EOS>=2)、および
    AblationTransformer1 / AblationTransformer2 の学習フォワード・ロス計算・KVキャッシュ制御を一気通貫で検証する。
    """
    import torch
    from mortm.models.mortm5 import AblationTransformer1, AblationTransformer2, MidiVision
    from mortm.train.datasets import MORTM5MemmapDataset, mortm5_collate_fn

    print(f"=== Running MORTM5 Tokenizer + Memmap Dataset + AblationTransformer Self-Test on {device} ===")
    eval_json = DEFAULT_SPLITS["val"]
    with open(eval_json, "r", encoding="utf-8") as f:
        sample_paths = json.load(f)[:12]

    with tempfile.TemporaryDirectory(prefix="mortm5_selftest_") as tmpdir:
        sample_json = os.path.join(tmpdir, "sample_paths.json")
        with open(sample_json, "w", encoding="utf-8") as f:
            json.dump(sample_paths, f)

        models_loaded: Dict[str, Tuple[MidiVision, MORTM5Args]] = {}
        writers: List[ModelWriterState] = []
        for model_key, cfg in DEFAULT_MODELS.items():
            model, m_args = load_midivision_model(cfg["config"], cfg["ckpt"], device)
            models_loaded[model_key] = (model, m_args)
            writers.append(
                ModelWriterState(
                    model_key=model_key,
                    model=model,
                    args=m_args,
                    out_dir=tmpdir,
                    split="val",
                    resume=False,
                )
            )

        # 前半6曲で一度停止し、後半6曲をレジュームで処理してレジューム整合性も検証する
        process_split(
            split="val",
            json_path=sample_json,
            writers=writers,
            device=device,
            num_workers=2,
            min_chunks=4,
            gpu_batch_chunks=128,
            checkpoint_every=5,
            limit=6,
        )
        resumed_writers = [
            ModelWriterState(
                model_key=model_key,
                model=models_loaded[model_key][0],
                args=models_loaded[model_key][1],
                out_dir=tmpdir,
                split="val",
                resume=True,
            )
            for model_key in DEFAULT_MODELS.keys()
        ]
        process_split(
            split="val",
            json_path=sample_json,
            writers=resumed_writers,
            device=device,
            num_workers=2,
            min_chunks=4,
            gpu_batch_chunks=128,
            checkpoint_every=5,
            limit=12,
        )

        for model_key, (mv_model, m_args) in models_loaded.items():
            print(f"\n--- Verifying {model_key} (G={m_args.vq_groups}, L={m_args.vq_layers}, vocab={m_args.vocab_size}) ---")
            bin_path = os.path.join(tmpdir, model_key, "val_tokens.bin")
            off_path = os.path.join(tmpdir, model_key, "val_offsets.npy")

            # 1. Flat mode (AblationTransformer1)
            ds_flat = MORTM5MemmapDataset(
                tokens_bin_path=bin_path,
                offsets_npy_path=off_path,
                vq_groups=m_args.vq_groups,
                vq_layers=m_args.vq_layers,
                max_chunks=16,
                min_chunks=4,
                mode="flat",
            )
            assert len(ds_flat) > 0, "Flat dataset should not be empty"
            seq0 = ds_flat[0]
            assert int(seq0[0]) == BOS_TOKEN_ID, f"Expected BOS={BOS_TOKEN_ID}, got {seq0[0]}"
            assert int(seq0[-1]) == EOS_TOKEN_ID, f"Expected EOS={EOS_TOKEN_ID}, got {seq0[-1]}"
            body = seq0[1:-1]
            assert int(body.min()) >= RVQ_TOKEN_OFFSET, f"Body tokens must be >= {RVQ_TOKEN_OFFSET}"
            assert int(body.max()) < m_args.vocab_size, f"Body tokens must be < {m_args.vocab_size}"

            # Decode body tokens back via MidiVision.decode_tokens (1D, 2D flat, 2D grouped, 3D grouped)
            tpc = m_args.vq_groups * m_args.vq_layers
            four_chunks_flat = body[: 4 * tpc].to(device)
            recon_1d = mv_model.decode_tokens(four_chunks_flat)
            recon_2d_flat = mv_model.decode_tokens(four_chunks_flat.reshape(4, tpc))
            recon_2d_grp = mv_model.decode_tokens(four_chunks_flat.reshape(4 * m_args.vq_layers, m_args.vq_groups))
            recon_3d_grp = mv_model.decode_tokens(four_chunks_flat.reshape(2, 2 * m_args.vq_layers, m_args.vq_groups))
            assert recon_1d.shape == (4, 8, 128, 24), f"Unexpected recon shape {recon_1d.shape}"
            assert torch.allclose(recon_1d, recon_2d_flat, atol=1e-5) and torch.allclose(recon_1d, recon_2d_grp, atol=1e-5) and torch.allclose(recon_1d, recon_3d_grp, atol=1e-5)

            # Test AblationTransformer1 forward, causality, KV-cache reset, & loss
            test_args = MORTM5Args(DEFAULT_MODELS[model_key]["config"])
            test_args.d_model = 128
            test_args.d_layer = 2
            test_args.num_heads = 4
            test_args.dim_feedforward = 256

            batch_flat = [ds_flat[0], ds_flat[min(1, len(ds_flat) - 1)][:-16]]
            in_flat, tgt_flat, mask_flat = mortm5_collate_fn(
                batch_flat, mode="flat", vq_groups=m_args.vq_groups
            )
            in_flat, tgt_flat, mask_flat = in_flat.to(device), tgt_flat.to(device), mask_flat.to(device)

            tf1 = AblationTransformer1(test_args).to(device).eval()
            logits1 = tf1(in_flat, padding_mask=mask_flat, is_causal=True)
            loss1 = torch.nn.functional.cross_entropy(
                logits1.reshape(-1, test_args.vocab_size),
                tgt_flat.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
            tf1.reset_kv_cache()
            bos_1 = torch.tensor([[BOS_TOKEN_ID]], device=device, dtype=torch.long)
            _ = tf1(bos_1, is_save_cache=True)
            assert int(tf1.decoder.layers[0].self_attention.cache_seqlens[0]) == 1
            _ = tf1(bos_1, is_save_cache=True)
            assert int(tf1.decoder.layers[0].self_attention.cache_seqlens[0]) == 2
            tf1.reset_kv_cache()
            assert not tf1._has_active_kv_cache()
            print(
                f"  [AblationTransformer1] batch={tuple(in_flat.shape)}, "
                f"logits={tuple(logits1.shape)}, loss={loss1.item():.4f}"
            )

            # 2. Grouped mode (AblationTransformer2)
            ds_grouped = MORTM5MemmapDataset(
                tokens_bin_path=bin_path,
                offsets_npy_path=off_path,
                vq_groups=m_args.vq_groups,
                vq_layers=m_args.vq_layers,
                max_chunks=16,
                min_chunks=4,
                mode="grouped",
            )
            gseq0 = ds_grouped[0]
            G = m_args.vq_groups
            assert gseq0.numel() % G == 0, "Grouped sequence length must be divisible by G"
            assert torch.all(gseq0[:G] == BOS_TOKEN_ID), "First soft-token must be all BOS"
            assert torch.all(gseq0[-G:] == EOS_TOKEN_ID), "Last soft-token must be all EOS"

            batch_grp = [ds_grouped[0], ds_grouped[min(1, len(ds_grouped) - 1)][:-16]]
            in_grp, tgt_grp, mask_grp = mortm5_collate_fn(
                batch_grp, mode="grouped", vq_groups=G
            )
            in_grp, tgt_grp, mask_grp = in_grp.to(device), tgt_grp.to(device), mask_grp.to(device)

            tf2 = AblationTransformer2(test_args).to(device).eval()
            logits2 = tf2(in_grp, padding_mask=mask_grp, is_causal=True)
            local_tgts = tf2.to_local_targets(tgt_grp, ignore_index=PAD_TOKEN_ID)

            loss2 = torch.nn.functional.cross_entropy(
                logits2.reshape(-1, tf2.group_vocab_size),
                local_tgts.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
            tf2.reset_kv_cache()
            bos_g = torch.full((1, 1, G), BOS_TOKEN_ID, device=device, dtype=torch.long)
            _ = tf2(bos_g, is_save_cache=True)
            assert int(tf2.decoder.layers[0].self_attention.cache_seqlens[0]) == 1
            _ = tf2(bos_g, is_save_cache=True)
            assert int(tf2.decoder.layers[0].self_attention.cache_seqlens[0]) == 2
            tf2.reset_kv_cache()
            assert not tf2._has_active_kv_cache()
            print(
                f"  [AblationTransformer2] batch={tuple(in_grp.shape)}, "
                f"logits={tuple(logits2.shape)}, group_vocab={tf2.group_vocab_size}, loss={loss2.item():.4f}"
            )

    print("\n=== Self-Test PASSED Successfully! ===")


def _resolve_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="Convert MIDI files into MORTM5 MidiVision flat binary token datasets.")
    parser.add_argument(
        "--remote_url",
        type=str,
        default=None,
        help="Colab Remote GPU Worker tunnel URL (when set, tokenizes on Colab GPU and saves directly to Google Drive).",
    )
    parser.add_argument(
        "--remote_out_dir",
        type=str,
        default="/content/drive/MyDrive/MORTM5_tokens",
        help="Output directory on Colab / Google Drive when --remote_url is used.",
    )
    parser.add_argument(
        "--allow_unmounted_drive",
        action="store_true",
        help="Allow saving to Colab local disk even if Google Drive is not mounted.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/mortm5_tokens",
        help="Local output directory (or local state mirror directory when --remote_url is used).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["m3_g8l2", "m4_g4l4"],
        choices=list(DEFAULT_MODELS.keys()),
        help="Tokenizer models to run simultaneously.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "train"],
        choices=list(DEFAULT_SPLITS.keys()),
        help="Dataset splits to process (default: val then train).",
    )
    parser.add_argument("--train_json", type=str, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val_json", type=str, default=DEFAULT_SPLITS["val"])
    parser.add_argument("--device", type=str, default="auto", help="Local device (auto, cuda:0, mps, cpu).")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU worker processes for MIDI parsing.")
    parser.add_argument("--min_chunks", type=int, default=8, help="Minimum chunks (8 chunks = 1 bar) required to keep a song.")
    parser.add_argument("--gpu_batch_chunks", type=int, default=1024, help="Number of chunks per GPU forward batch.")
    parser.add_argument("--remote_batch_chunks", type=int, default=4096, help="Number of chunks per HTTP batch sent to Colab.")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Flush state & offsets to disk/Drive every N songs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of MIDI files per split.")
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch instead of resuming.")
    parser.add_argument("--selftest", action="store_true", help="Run end-to-end verification self-test and exit.")
    args = parser.parse_args()

    split_json_map = {
        "val": args.val_json,
        "train": args.train_json,
    }

    if args.selftest:
        device = _resolve_device(args.device)
        run_selftest(device)
        return

    if args.remote_url:
        remote_url = args.remote_url.rstrip("/")
        session = requests.Session()
        init_remote_colab_worker(session, remote_url, args.models)
        for split in args.splits:
            process_split_remote(
                session=session,
                remote_url=remote_url,
                remote_out_dir=args.remote_out_dir,
                local_state_dir=args.out_dir,
                split=split,
                json_path=split_json_map[split],
                model_keys=args.models,
                resume=(not args.no_resume),
                require_drive=(not args.allow_unmounted_drive),
                num_workers=args.num_workers,
                min_chunks=args.min_chunks,
                gpu_batch_chunks=args.gpu_batch_chunks,
                remote_batch_chunks=args.remote_batch_chunks,
                checkpoint_every=args.checkpoint_every,
                limit=args.limit,
            )
        return

    device = _resolve_device(args.device)
    print(f"Loading MidiVision models {args.models} on {device}...")
    loaded_models: Dict[str, Tuple[Any, MORTM5Args]] = {}
    for m_key in args.models:
        cfg = DEFAULT_MODELS[m_key]
        loaded_models[m_key] = load_midivision_model(cfg["config"], cfg["ckpt"], device)
        m_args = loaded_models[m_key][1]
        print(
            f"  Loaded {m_key}: G={m_args.vq_groups}, L={m_args.vq_layers}, "
            f"K={m_args.codebook_size}, vocab_size={m_args.vocab_size} "
            f"(<PAD>={m_args.pad_token_id}, <BOS>={m_args.bos_token_id}, <EOS>={m_args.eos_token_id}, offset={m_args.token_offset})"
        )

    for split in args.splits:
        writers = [
            ModelWriterState(
                model_key=m_key,
                model=loaded_models[m_key][0],
                args=loaded_models[m_key][1],
                out_dir=args.out_dir,
                split=split,
                resume=(not args.no_resume),
            )
            for m_key in args.models
        ]
        process_split(
            split=split,
            json_path=split_json_map[split],
            writers=writers,
            device=device,
            num_workers=args.num_workers,
            min_chunks=args.min_chunks,
            gpu_batch_chunks=args.gpu_batch_chunks,
            checkpoint_every=args.checkpoint_every,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
