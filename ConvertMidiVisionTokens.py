"""
ConvertMidiVisionTokens.py

GigaMIDI (GMD) 等の MIDI ファイル群を 1 度だけ CPU でピアノロールチャンクに変換し、
GPU 上で MidiVision (M3: FineGroup G8L2 / M4: DeepRVQ G4L4) の両モデルで同時にトークン化して、
高速・省メモリな単一フラットバイナリデータベース (np.memmap 対応) を構築するスクリプト。

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
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm

from mortm.models.modules.config import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MORTM5Args,
    PAD_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
)
from mortm.utils.convert_pianoroll import midi_to_chunks

if TYPE_CHECKING:
    import torch
    from mortm.models.mortm5 import MidiVision


DEFAULT_MODELS = {
    "m3_g8l2": {
        "config": "configs/models/mortm5/m3_finegroup_g8l2_d256_k2048.json",
        "ckpt": "out/models/mortm5/M3/MORTM5.M3_FineGroup_G8L2_D256_K2048_latest.pth",
    },
    "m4_g4l4": {
        "config": "configs/models/mortm5/m4_deeprvq_g4l4_d128_k2048.json",
        "ckpt": "out/models/mortm5/M4/MORTM5.M4_DeepRVQ_G4L4_D128_K2048_latest.pth",
    },
}

DEFAULT_SPLITS = {
    "val": "out/models/mortm5/eval_paths_MIDIVision.json",
    "train": "out/models/mortm5/train_paths_MIDIVision.json",
}


def _midi_worker(task: Tuple[int, str, int]) -> Tuple[int, str, Optional[np.ndarray]]:
    """
    CPU ワーカー関数 (spawn プロセスで実行、PyTorch/CUDA 非ロード):
    MIDI ファイルを読み込み、[N_chunks, 8, 128, 24] の uint8 (0..254) 配列に変換して返す。
    ※ 3GB 級の長尺曲で float32 の一時コピーが複数発生しないよう、インプレース演算で uint8 化してから転置する。
    """
    midi_idx, midi_path, min_chunks = task
    try:
        chunks, _ = midi_to_chunks(midi_path)
        if chunks is None or len(chunks) < min_chunks:
            return midi_idx, midi_path, None
        n_chunks = len(chunks)
        # インプレースで [0, 1] -> [0, 254] に丸めてから uint8 に変換（余分な float32 配列確保を防止）
        np.clip(chunks, 0.0, 1.0, out=chunks)
        chunks *= 254.0
        np.rint(chunks, out=chunks)
        chunks_u8 = chunks.astype(np.uint8)
        del chunks
        # uint8 (1/4 サイズ) にしてから [N, 8, 24, 128] -> [N, 8, 128, 24] に転置・連続化
        chunks_u8 = np.ascontiguousarray(np.transpose(chunks_u8, (0, 1, 3, 2)))
        if n_chunks >= 512:
            gc.collect()
        return midi_idx, midi_path, chunks_u8
    except Exception:
        gc.collect()
        return midi_idx, midi_path, None


class ModelWriterState:
    """各トークナイザーモデル (M3 / M4) の出力ファイルとオフセット状態を管理するクラス。"""

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

            # トークンバイナリを確定済みトークン数で検証・切り詰める (uint16 = 2 bytes)
            expected_bytes = self.total_tokens * 2
            actual_bytes = os.path.getsize(self.tokens_bin_path)
            if actual_bytes < expected_bytes:
                raise RuntimeError(
                    f"[{self.model_key}/{self.split}] Resume failed: {self.tokens_bin_path} is truncated "
                    f"({actual_bytes} bytes < expected {expected_bytes} bytes). Run with --no_resume to rebuild."
                )
            with open(self.tokens_bin_path, "r+b") as bf:
                bf.truncate(expected_bytes)

            # マニフェストも確定済み曲数に検証・切り詰める
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

            self.bin_file = open(self.tokens_bin_path, "ab")
            self.manifest_file = open(self.manifest_path, "a", encoding="utf-8")
        else:
            self.bin_file = open(self.tokens_bin_path, "wb")
            self.manifest_file = open(self.manifest_path, "w", encoding="utf-8")

    def append_song(self, midi_idx: int, midi_path: str, num_chunks: int, song_tokens_u16: np.ndarray):
        if midi_idx < self.next_midi_idx:
            # 既にコミット済みの曲はスキップして重複追記を防ぐ
            return
        if midi_idx > self.next_midi_idx:
            raise RuntimeError(
                f"[{self.model_key}/{self.split}] Out-of-order midi_idx={midi_idx}, expected={self.next_midi_idx}"
            )

        num_tokens = int(song_tokens_u16.size)
        start_offset = self.total_tokens

        self.bin_file.write(song_tokens_u16.tobytes())
        self.offsets_list.append((start_offset, num_tokens))

        record = {
            "song_idx": self.valid_songs,
            "midi_idx": midi_idx,
            "midi_path": midi_path,
            "num_chunks": num_chunks,
            "num_tokens": num_tokens,
            "start_offset": start_offset,
        }
        self.manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        self.bin_file.flush()
        os.fsync(self.bin_file.fileno())
        self.manifest_file.flush()
        os.fsync(self.manifest_file.fileno())

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
        if hasattr(self, "bin_file") and not self.bin_file.closed:
            self.bin_file.close()
        if hasattr(self, "manifest_file") and not self.manifest_file.closed:
            self.manifest_file.close()

    def close(self, completed: bool = True):
        self.flush_checkpoint(completed=completed)
        self.close_files_only()


def load_midivision_model(config_path: str, ckpt_path: str, device: Any) -> Tuple[Any, MORTM5Args]:
    import torch
    from mortm.models.mortm5 import MidiVision

    args = MORTM5Args(config_path)
    model = MidiVision(args)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    # Strip '_orig_mod.' or 'module.' prefixes if present
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
    pending_songs: List[Tuple[int, str, np.ndarray]],
    writers: List[ModelWriterState],
    device: Any,
    gpu_batch_chunks: int,
    use_bf16: bool = True,
):
    if not pending_songs:
        return
    import torch

    all_chunks_u8 = np.concatenate([item[2] for item in pending_songs], axis=0)
    total_chunks = len(all_chunks_u8)
    song_lengths = [len(item[2]) for item in pending_songs]

    model_token_buffers: Dict[str, List[np.ndarray]] = {w.model_key: [] for w in writers}

    with torch.inference_mode():
        for start in range(0, total_chunks, gpu_batch_chunks):
            end = min(start + gpu_batch_chunks, total_chunks)
            sub_u8 = torch.from_numpy(all_chunks_u8[start:end]).to(device=device, non_blocking=True)
            x = sub_u8.to(dtype=torch.float32) / 254.0

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(use_bf16 and device.type == "cuda")):
                for w in writers:
                    tokens = w.model.encode_tokens(x, include_offset=True)
                    tokens_u16 = tokens.to(dtype=torch.uint16).cpu().numpy()
                    model_token_buffers[w.model_key].append(tokens_u16)

    for w in writers:
        full_tokens_u16 = np.concatenate(model_token_buffers[w.model_key], axis=0)
        cursor = 0
        for (midi_idx, midi_path, _), n_chunks in zip(pending_songs, song_lengths):
            song_tokens = full_tokens_u16[cursor : cursor + n_chunks].reshape(-1)
            w.append_song(midi_idx, midi_path, n_chunks, song_tokens)
            cursor += n_chunks


def process_split(
    split: str,
    json_path: str,
    writers: List[ModelWriterState],
    device: Any,
    num_workers: int = 12,
    min_chunks: int = 1,
    gpu_batch_chunks: int = 512,
    checkpoint_every: int = 2000,
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

    from collections import deque

    tasks = [(idx, midi_paths[idx], min_chunks) for idx in range(start_idx, total_files)]
    pending_songs: List[Tuple[int, str, np.ndarray]] = []
    pending_chunks = 0
    processed_since_ckpt = 0

    pbar = tqdm(total=total_files, initial=start_idx, desc=f"Tokenizing [{split}]", unit="song", dynamic_ncols=True)

    if num_workers > 0:
        mp_ctx = mp.get_context("spawn")
        pool = mp_ctx.Pool(processes=num_workers)
        max_in_flight = max(8, num_workers * 2)
    else:
        pool = None
        max_in_flight = 1

    def _bounded_task_iter():
        if pool is None:
            for t in tasks:
                yield _midi_worker(t)
            return

        in_flight = deque()
        next_pos = 0
        n_tasks = len(tasks)
        while next_pos < min(n_tasks, max_in_flight):
            in_flight.append(pool.apply_async(_midi_worker, (tasks[next_pos],)))
            next_pos += 1

        while in_flight:
            res_future = in_flight.popleft()
            if next_pos < n_tasks:
                in_flight.append(pool.apply_async(_midi_worker, (tasks[next_pos],)))
                next_pos += 1
            yield res_future.get()

    try:
        for midi_idx, midi_path, chunks_u8 in _bounded_task_iter():
            if chunks_u8 is None:
                # もし pending_songs が溜まっていれば先にフラッシュして midi_idx の順序を保つ
                if pending_songs:
                    flush_gpu_batch(pending_songs, writers, device, gpu_batch_chunks)
                    pending_songs.clear()
                    pending_chunks = 0
                for w in writers:
                    w.record_skip(midi_idx)
            else:
                pending_songs.append((midi_idx, midi_path, chunks_u8))
                pending_chunks += len(chunks_u8)

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

    print("=== Running MORTM5 Tokenizer + Memmap Dataset + AblationTransformer Self-Test ===")
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
            assert torch.allclose(recon_1d, recon_2d_flat) and torch.allclose(recon_1d, recon_2d_grp) and torch.allclose(recon_1d, recon_3d_grp)

            # Test AblationTransformer1 forward, causality, KV-cache reset, & loss
            test_args = MORTM5Args(DEFAULT_MODELS[model_key]["config"])
            test_args.d_model = 128
            test_args.d_layer = 2
            test_args.num_heads = 4
            test_args.dim_feedforward = 256

            batch_flat = [ds_flat[0], ds_flat[min(1, len(ds_flat) - 1)][:-16]]  # 異なる長さで PAD を検証
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
            # Test reset_kv_cache on 1-token BOS prefill + 1-step decode
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
            logits2 = tf2(in_grp, padding_mask=mask_grp, is_causal=True)  # [B, S_soft, G, group_vocab_size]
            local_tgts = tf2.to_local_targets(tgt_grp, ignore_index=PAD_TOKEN_ID)  # [B, S_soft, G]

            loss2 = torch.nn.functional.cross_entropy(
                logits2.reshape(-1, tf2.group_vocab_size),
                local_tgts.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
            # Test reset_kv_cache on 1-soft-token BOS prefill + 1-step decode
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


def main():
    import torch

    parser = argparse.ArgumentParser(description="Convert MIDI files into MORTM5 MidiVision flat binary token datasets.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/nvme0n1p3/MORTM/mortm5_tokens",
        help="Output root directory for tokenized memmap datasets.",
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of CPU worker processes for MIDI parsing.")
    parser.add_argument("--min_chunks", type=int, default=8, help="Minimum chunks (8 chunks = 1 bar) required to keep a song.")
    parser.add_argument("--gpu_batch_chunks", type=int, default=512, help="Number of chunks per GPU forward batch.")
    parser.add_argument("--checkpoint_every", type=int, default=500, help="Flush state & offsets every N songs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of MIDI files per split.")
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch instead of resuming.")
    parser.add_argument("--selftest", action="store_true", help="Run end-to-end verification self-test and exit.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.selftest:
        run_selftest(device)
        return

    split_json_map = {
        "val": args.val_json,
        "train": args.train_json,
    }

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
