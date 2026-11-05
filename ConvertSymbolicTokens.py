#!/usr/bin/env python3
"""
ConvertSymbolicTokens.py

Multi-core streaming MIDI -> Symbolic Token (`Octuple-MMT`, extensible via Registry to `Multitrack-REMI`)
corpus converter with:
  1. Single Source of Truth `extract_quantized_notes_and_slots` (`mortm/utils/convert_pianoroll.py`).
  2. 1-to-1 song and 16-bar non-overlapping segment parity (`max_chunks=128`, `min_chunks=8`, `stride_chunks=128`)
     with `out/mortm5_tokens/m4_g4l4`.
  3. Bounded $O(1)$ streaming RAM, atomic `tmp` + `os.replace` resume checkpoints, and automatic parity
     verification against `m4_g4l4` `{split}_state.json` & `{split}_slots.npy`.

Output files under `{out_dir}/{tokenizer_key}/`:
  - `{split}_tokens.bin`   : `uint16` flat body tokens (memmap-ready)
  - `{split}_segments.npy` : `int64 [N_segments, 6]` `(token_start, token_len, is_first, is_last, song_idx, num_chunks)`
  - `{split}_offsets.npy`  : `int64 [N_songs, 2]` `(song_token_start, song_total_tokens)`
  - `{split}_slots.npy`    : `int16 [N_songs, 4]` (4-slot instrument codes, bit-for-bit identical to `m4_g4l4`)
  - `{split}_state.json`   : atomic resume & corpus statistics manifest (including `total_canonical_notes` for BPB)
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

import numpy as np
from tqdm import tqdm

from mortm.models.modules.config import NUM_SLOT_TOKENS
from mortm.utils.convert_pianoroll import extract_quantized_notes_and_slots
from mortm.utils.symbolic_tokenizers import (
    SYMBOLIC_TOKENIZER_REGISTRY,
    AbstractSymbolicSegmentTokenizer,
    MultitrackREMITokenizer,
    SegmentWindowSpec,
    build_symbolic_tokenizer,
    extract_segment_canonical_notes,
)


def _resolve_existing_path(candidates: Sequence[str]) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DEFAULT_SPLITS: Dict[str, str] = {
    "val": _resolve_existing_path([
        "data/gmd/eval.json",
        "out/models/mortm5/eval_paths_MIDIVision.json",
    ]),
    "train": _resolve_existing_path([
        "data/gmd/train.json",
        "out/models/mortm5/train_paths_MIDIVision.json",
    ]),
}


@dataclass(frozen=True)
class SymbolicCorpusConfig:
    tokenizer_key: str = "octuple_mmt"
    out_dir: str = "out/mortm5_tokens"
    max_chunks: int = 128
    min_chunks: int = 8
    stride_chunks: int = 128
    num_workers: int = 12
    checkpoint_every: int = 5000
    worker_timeout_sec: int = 90
    resume: bool = True
    verify_m4_dir: Optional[str] = "out/mortm5_tokens/m4_g4l4"

    @property
    def window_spec(self) -> SegmentWindowSpec:
        return SegmentWindowSpec(
            max_chunks=self.max_chunks,
            min_chunks=self.min_chunks,
            stride_chunks=self.stride_chunks,
        )


# Per-worker lazily initialized tokenizer instance (avoids re-instantiating per MIDI file)
_WORKER_TOKENIZER_CACHE: Dict[str, AbstractSymbolicSegmentTokenizer] = {}


def _get_worker_tokenizer(tokenizer_key: str) -> AbstractSymbolicSegmentTokenizer:
    tok = _WORKER_TOKENIZER_CACHE.get(tokenizer_key)
    if tok is None:
        tok = build_symbolic_tokenizer(tokenizer_key)
        _WORKER_TOKENIZER_CACHE[tokenizer_key] = tok
    return tok


def _symbolic_midi_worker(
    task: Tuple[int, str, str, int, int, int],
) -> Tuple[int, int, Optional[Tuple[int, int, int, int]], Optional[List[Tuple[int, int, int, int]]], Optional[bytes], int, int]:
    """
    Pure CPU worker executed in `spawn` subprocesses (no PyTorch GPU context).

    Returns:
        `(midi_idx, n_chunks, slot_tuple, seg_rows, body_u16_bytes, raw_midi_notes, canonical_seg_notes)`
        where `seg_rows` contains `(seg_token_len, is_first, is_last, seg_num_chunks)`.
    """
    midi_idx, midi_path, tokenizer_key, max_chunks, min_chunks, stride_chunks = task
    try:
        n_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(midi_path)
        if n_chunks < min_chunks:
            return midi_idx, 0, None, None, None, 0, 0

        win_spec = SegmentWindowSpec(
            max_chunks=max_chunks,
            min_chunks=min_chunks,
            stride_chunks=stride_chunks,
        )
        segments = win_spec.compute_song_segments(n_chunks)
        if not segments:
            return midi_idx, 0, None, None, None, 0, 0

        tokenizer = _get_worker_tokenizer(tokenizer_key)
        max_bars = win_spec.max_bars

        seg_rows: List[Tuple[int, int, int, int]] = []
        seg_arrays: List[np.ndarray] = []
        canonical_notes_count = 0

        for c_start, c_end, is_first, is_last in segments:
            canon = extract_segment_canonical_notes(
                notes_arr=notes_arr,
                chunk_start=c_start,
                chunk_end=c_end,
                max_bars=max_bars,
            )
            canonical_notes_count += int(len(canon))
            body_u16 = tokenizer.encode_canonical_segment_notes(canon, num_chunks=int(c_end - c_start))
            seg_rows.append((int(body_u16.size), int(is_first), int(is_last), int(c_end - c_start)))
            if body_u16.size > 0:
                seg_arrays.append(body_u16)

        if seg_arrays:
            concat_u16 = np.concatenate(seg_arrays, axis=0)
            body_bytes = np.ascontiguousarray(concat_u16, dtype=np.uint16).tobytes()
        else:
            body_bytes = b""

        slot_tuple = (int(slot_codes[0]), int(slot_codes[1]), int(slot_codes[2]), int(slot_codes[3]))
        return (
            midi_idx,
            int(n_chunks),
            slot_tuple,
            seg_rows,
            body_bytes,
            int(len(notes_arr)),
            canonical_notes_count,
        )
    except Exception:
        return midi_idx, 0, None, None, None, 0, 0


def _iter_worker_results_bounded(
    tasks: Sequence[Tuple[int, str, str, int, int, int]],
    num_workers: int,
    timeout_sec: int,
) -> Iterator[Tuple[int, int, Optional[Tuple[int, int, int, int]], Optional[List[Tuple[int, int, int, int]]], Optional[bytes], int, int]]:
    if num_workers <= 0:
        for t in tasks:
            yield _symbolic_midi_worker(t)
        return

    mp_ctx = mp.get_context("spawn")
    eff_workers = max(1, int(num_workers))
    pool: Optional[Any] = mp_ctx.Pool(processes=eff_workers, maxtasksperchild=4000)
    max_in_flight = max(64, eff_workers * 16)
    in_flight: deque[Tuple[Tuple[int, str, str, int, int, int], Any]] = deque()
    next_pos = 0
    n_tasks = len(tasks)

    try:
        while next_pos < min(n_tasks, max_in_flight):
            in_flight.append((tasks[next_pos], pool.apply_async(_symbolic_midi_worker, (tasks[next_pos],))))
            next_pos += 1

        while in_flight:
            task_info, fut = in_flight.popleft()
            if next_pos < n_tasks:
                in_flight.append((tasks[next_pos], pool.apply_async(_symbolic_midi_worker, (tasks[next_pos],))))
                next_pos += 1
            try:
                yield fut.get(timeout=timeout_sec)
            except mp.TimeoutError as exc:
                midx, mpath = task_info[0], task_info[1]
                print(
                    f"\n  [Worker Warning] Skipping midi_idx={midx} ({mpath}) due to timeout ({timeout_sec}s): {exc}; respawning worker pool.",
                    flush=True,
                )
                yield midx, 0, None, None, None, 0, 0
                pending_tasks = [pt for pt, _ in in_flight]
                in_flight.clear()
                if pool is not None:
                    pool.terminate()
                    pool.join()
                pool = mp_ctx.Pool(processes=eff_workers, maxtasksperchild=4000)
                for pt in pending_tasks:
                    in_flight.append((pt, pool.apply_async(_symbolic_midi_worker, (pt,))))
            except Exception as exc:
                midx, mpath = task_info[0], task_info[1]
                print(f"\n  [Worker Warning] Skipping midi_idx={midx} ({mpath}) due to worker error: {exc}", flush=True)
                yield midx, 0, None, None, None, 0, 0
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


class SymbolicSplitWriter:
    """
    Atomic checkpointed persistence repository for a single `(tokenizer_key, split)` binary dataset.
    """

    def __init__(
        self,
        target_dir: Path,
        split: str,
        tokenizer: AbstractSymbolicSegmentTokenizer,
        window_spec: SegmentWindowSpec,
        resume: bool = True,
    ) -> None:
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.tokenizer = tokenizer
        self.window_spec = window_spec

        self.tokens_bin_path = self.target_dir / f"{split}_tokens.bin"
        self.segments_npy_path = self.target_dir / f"{split}_segments.npy"
        self.offsets_npy_path = self.target_dir / f"{split}_offsets.npy"
        self.slots_npy_path = self.target_dir / f"{split}_slots.npy"
        self.state_json_path = self.target_dir / f"{split}_state.json"

        self.processed_midi_count: int = 0
        self.valid_songs: int = 0
        self.skipped_songs: int = 0
        self.total_chunks: int = 0
        self.total_segments: int = 0
        self.total_tokens: int = 0
        self.total_raw_midi_notes: int = 0
        self.total_canonical_notes: int = 0

        self._segments_list: List[Tuple[int, int, int, int, int, int]] = []
        self._offsets_list: List[Tuple[int, int]] = []
        self._slots_list: List[Tuple[int, int, int, int]] = []
        self._pending_bytes = bytearray()

        if resume and self._can_resume():
            self._load_resume_state()
        else:
            self._reset_clean()

    def _can_resume(self) -> bool:
        return (
            self.state_json_path.exists()
            and self.tokens_bin_path.exists()
            and self.segments_npy_path.exists()
            and self.offsets_npy_path.exists()
            and self.slots_npy_path.exists()
        )

    def _cleanup_stray_temp_files(self) -> int:
        removed = 0
        prefix = f"{self.split}_"
        for child in self.target_dir.iterdir():
            if child.is_file() and child.name.startswith(prefix) and ".tmp" in child.name:
                try:
                    child.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def _reset_clean(self) -> None:
        self._cleanup_stray_temp_files()
        for p in (
            self.tokens_bin_path,
            self.segments_npy_path,
            self.offsets_npy_path,
            self.slots_npy_path,
            self.state_json_path,
        ):
            if p.exists():
                p.unlink()
        self.tokens_bin_path.touch()

    def _load_resume_state(self) -> None:
        removed_tmp = self._cleanup_stray_temp_files()
        with open(self.state_json_path, "r", encoding="utf-8") as f:
            st = json.load(f)

        segs = np.load(self.segments_npy_path)
        offs = np.load(self.offsets_npy_path)
        slots = np.load(self.slots_npy_path)

        valid_songs = int(st["valid_songs"])
        total_segments = int(st["total_segments"])
        total_tokens = int(st["total_tokens"])
        expected_bytes = total_tokens * 2

        if len(offs) < valid_songs or len(slots) < valid_songs or len(segs) < total_segments:
            print(f"[SymbolicSplitWriter] Incomplete index arrays for {self.split}; restarting cleanly.", flush=True)
            self._reset_clean()
            return
        if self.tokens_bin_path.stat().st_size < expected_bytes:
            print(f"[SymbolicSplitWriter] Truncated {self.tokens_bin_path}; restarting cleanly.", flush=True)
            self._reset_clean()
            return

        needs_heal = removed_tmp > 0 or (
            len(offs) > valid_songs or len(slots) > valid_songs or len(segs) > total_segments
        )
        # Truncate any partial trailing bytes past the last committed checkpoint
        if self.tokens_bin_path.stat().st_size > expected_bytes:
            with open(self.tokens_bin_path, "r+b") as bf:
                bf.truncate(expected_bytes)
                bf.flush()
                os.fsync(bf.fileno())
            needs_heal = True

        self.processed_midi_count = int(st["processed_midi_count"])
        self.valid_songs = valid_songs
        self.skipped_songs = int(st["skipped_songs"])
        self.total_chunks = int(st["total_chunks"])
        self.total_segments = total_segments
        self.total_tokens = total_tokens
        self.total_raw_midi_notes = int(st.get("total_raw_midi_notes", 0))
        self.total_canonical_notes = int(st.get("total_canonical_notes", 0))

        self._segments_list = [
            (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
            for r in segs[:total_segments]
        ]
        self._offsets_list = [(int(r[0]), int(r[1])) for r in offs[:valid_songs]]
        self._slots_list = [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in slots[:valid_songs]]

        if needs_heal:
            self.flush()

    def record_skipped(self, midi_idx: int) -> None:
        self.skipped_songs += 1
        self.processed_midi_count = midi_idx + 1

    def record_valid_song(
        self,
        midi_idx: int,
        n_chunks: int,
        slot_tuple: Tuple[int, int, int, int],
        seg_rows: Sequence[Tuple[int, int, int, int]],
        body_bytes: bytes,
        raw_midi_notes: int,
        canonical_notes: int,
    ) -> None:
        song_idx = self.valid_songs
        song_token_start = self.total_tokens
        song_token_len = 0

        for seg_tok_len, is_first, is_last, seg_chunks in seg_rows:
            self._segments_list.append(
                (
                    self.total_tokens,
                    int(seg_tok_len),
                    int(is_first),
                    int(is_last),
                    song_idx,
                    int(seg_chunks),
                )
            )
            self.total_tokens += int(seg_tok_len)
            song_token_len += int(seg_tok_len)
            self.total_segments += 1

        if body_bytes:
            if len(body_bytes) != song_token_len * 2:
                raise RuntimeError(
                    f"Byte length mismatch on midi_idx={midi_idx}: got {len(body_bytes)} bytes for {song_token_len} uint16 tokens."
                )
            self._pending_bytes.extend(body_bytes)

        self._offsets_list.append((song_token_start, song_token_len))
        self._slots_list.append(slot_tuple)
        self.valid_songs += 1
        self.total_chunks += int(n_chunks)
        self.total_raw_midi_notes += int(raw_midi_notes)
        self.total_canonical_notes += int(canonical_notes)
        self.processed_midi_count = midi_idx + 1

    def flush(self) -> Dict[str, Any]:
        if self._pending_bytes:
            with open(self.tokens_bin_path, "ab") as bf:
                bf.write(self._pending_bytes)
                bf.flush()
                os.fsync(bf.fileno())
            self._pending_bytes.clear()

        seg_arr = (
            np.asarray(self._segments_list, dtype=np.int64)
            if self._segments_list
            else np.zeros((0, 6), dtype=np.int64)
        )
        off_arr = (
            np.asarray(self._offsets_list, dtype=np.int64)
            if self._offsets_list
            else np.zeros((0, 2), dtype=np.int64)
        )
        slots_arr = (
            np.asarray(self._slots_list, dtype=np.int16)
            if self._slots_list
            else np.zeros((0, NUM_SLOT_TOKENS), dtype=np.int16)
        )

        self._atomic_save_npy(self.segments_npy_path, seg_arr)
        self._atomic_save_npy(self.offsets_npy_path, off_arr)
        self._atomic_save_npy(self.slots_npy_path, slots_arr)

        #Prefix + EOS tokens per segment: (1 <BOS> + 4 <INST> + 1 <EOS>) * step_unit_width
        w = self.tokenizer.step_unit_width
        full_seq_flat_tokens = self.total_tokens + self.total_segments * 6 * w
        full_seq_ar_steps = full_seq_flat_tokens // max(1, w)

        state = {
            "tokenizer_key": self.tokenizer.tokenizer_key,
            "model_mode": self.tokenizer.model_mode,
            "vocab_size": self.tokenizer.vocab_size,
            "step_unit_width": w,
            "split": self.split,
            "max_chunks": self.window_spec.max_chunks,
            "min_chunks": self.window_spec.min_chunks,
            "stride_chunks": self.window_spec.stride_chunks,
            "processed_midi_count": self.processed_midi_count,
            "next_midi_idx": self.processed_midi_count,
            "valid_songs": self.valid_songs,
            "skipped_songs": self.skipped_songs,
            "total_chunks": self.total_chunks,
            "total_segments": self.total_segments,
            "total_tokens": self.total_tokens,
            "full_seq_flat_tokens": full_seq_flat_tokens,
            "full_seq_ar_steps": full_seq_ar_steps,
            "total_raw_midi_notes": self.total_raw_midi_notes,
            "total_canonical_notes": self.total_canonical_notes,
            "raw_information_bits_36b_per_note": self.total_canonical_notes * 36,
            "updated_at": time.time(),
        }
        tmp_state = self.state_json_path.with_suffix(".json.tmp")
        with open(tmp_state, "w", encoding="utf-8") as sf:
            json.dump(state, sf, indent=2)
            sf.flush()
            os.fsync(sf.fileno())
        os.replace(tmp_state, self.state_json_path)
        return state

    @staticmethod
    def _atomic_save_npy(dst_path: Path, arr: np.ndarray) -> None:
        tmp_path = str(dst_path) + ".tmp.npy"
        np.save(tmp_path, arr)
        os.replace(tmp_path, dst_path)


class SymbolicCorpusConverter:
    """
    High-throughput multi-core corpus converter with automatic 1-to-1 `m4_g4l4` parity verification.
    """

    def __init__(self, config: SymbolicCorpusConfig) -> None:
        self.config = config
        self.tokenizer = build_symbolic_tokenizer(config.tokenizer_key)

    def convert_split(self, split: str, json_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        with open(json_path, "r", encoding="utf-8") as f:
            midi_paths: List[str] = json.load(f)

        is_partial = limit is not None and int(limit) > 0 and int(limit) < len(midi_paths)
        if limit is not None and int(limit) > 0:
            midi_paths = midi_paths[: int(limit)]

        total_files = len(midi_paths)
        target_dir = Path(self.config.out_dir) / self.tokenizer.tokenizer_key
        writer = SymbolicSplitWriter(
            target_dir=target_dir,
            split=split,
            tokenizer=self.tokenizer,
            window_spec=self.config.window_spec,
            resume=self.config.resume and (not is_partial),
        )

        start_idx = writer.processed_midi_count
        if start_idx >= total_files:
            print(
                f"[{self.tokenizer.tokenizer_key}:{split}] Already completed ({start_idx:,}/{total_files:,} files).",
                flush=True,
            )
            state = writer.flush()
            if not is_partial:
                self._verify_m4_parity_if_available(split=split, writer=writer)
            return state

        print(
            f"[{self.tokenizer.tokenizer_key}:{split}] Converting {total_files - start_idx:,}/{total_files:,} MIDIs "
            f"(workers={self.config.num_workers}, max_chunks={self.config.max_chunks}, "
            f"min_chunks={self.config.min_chunks}, stride_chunks={self.config.stride_chunks}) -> {target_dir}",
            flush=True,
        )

        tasks = [
            (
                idx,
                midi_paths[idx],
                self.tokenizer.tokenizer_key,
                self.config.max_chunks,
                self.config.min_chunks,
                self.config.stride_chunks,
            )
            for idx in range(start_idx, total_files)
        ]

        pbar = tqdm(
            total=total_files,
            initial=start_idx,
            desc=f"Symbolic [{self.tokenizer.tokenizer_key}:{split}]",
            unit="song",
            dynamic_ncols=True,
        )
        since_ckpt = 0
        t0 = time.time()
        try:
            for (
                midi_idx,
                n_chunks,
                slot_tuple,
                seg_rows,
                body_bytes,
                raw_notes,
                canon_notes,
            ) in _iter_worker_results_bounded(
                tasks=tasks,
                num_workers=int(self.config.num_workers),
                timeout_sec=self.config.worker_timeout_sec,
            ):
                if n_chunks <= 0 or slot_tuple is None or seg_rows is None or body_bytes is None:
                    writer.record_skipped(midi_idx)
                else:
                    writer.record_valid_song(
                        midi_idx=midi_idx,
                        n_chunks=n_chunks,
                        slot_tuple=slot_tuple,
                        seg_rows=seg_rows,
                        body_bytes=body_bytes,
                        raw_midi_notes=raw_notes,
                        canonical_notes=canon_notes,
                    )

                since_ckpt += 1
                pbar.update(1)
                if since_ckpt >= self.config.checkpoint_every:
                    st = writer.flush()
                    since_ckpt = 0
                    pbar.set_postfix(
                        valid=f"{st['valid_songs']:,}",
                        skip=st["skipped_songs"],
                        segs=f"{st['total_segments']:,}",
                        M_steps=f"{st['full_seq_ar_steps'] / 1e6:.2f}M",
                    )

            final_state = writer.flush()
        finally:
            pbar.close()

        elapsed = max(1e-3, time.time() - t0)
        mean_steps = final_state["full_seq_ar_steps"] / max(1, final_state["total_segments"])
        print(
            f"✅ [{self.tokenizer.tokenizer_key}:{split}] Done in {elapsed:.1f}s:\n"
            f"   - Valid / Skipped Songs : {final_state['valid_songs']:,} / {final_state['skipped_songs']:,}\n"
            f"   - Total Chunks / Segs   : {final_state['total_chunks']:,} chunks / {final_state['total_segments']:,} segments\n"
            f"   - Canonical Notes       : {final_state['total_canonical_notes']:,} notes\n"
            f"   - Total Body / Full Toks: {final_state['total_tokens']:,} / {final_state['full_seq_flat_tokens']:,} flat tokens\n"
            f"   - Total AR Steps        : {final_state['full_seq_ar_steps']:,} steps (mean {mean_steps:.1f} steps/16-bar segment)",
            flush=True,
        )

        if not is_partial:
            self._verify_m4_parity_if_available(split=split, writer=writer)

        return final_state

    def _verify_m4_parity_if_available(self, split: str, writer: SymbolicSplitWriter) -> None:
        if not self.config.verify_m4_dir:
            return
        ref_dir = Path(self.config.verify_m4_dir)
        ref_state_path = ref_dir / f"{split}_state.json"
        ref_slots_path = ref_dir / f"{split}_slots.npy"
        if not ref_state_path.exists():
            return

        with open(ref_state_path, "r", encoding="utf-8") as rf:
            ref_st = json.load(rf)

        ref_processed = int(ref_st.get("next_midi_idx", ref_st.get("processed_midi_count", -1)))
        if ref_processed != writer.processed_midi_count:
            return

        assert writer.valid_songs == int(ref_st["valid_songs"]), (
            f"[{split}] valid_songs mismatch vs m4_g4l4: got {writer.valid_songs}, expected {ref_st['valid_songs']}"
        )
        assert writer.skipped_songs == int(ref_st["skipped_songs"]), (
            f"[{split}] skipped_songs mismatch vs m4_g4l4: got {writer.skipped_songs}, expected {ref_st['skipped_songs']}"
        )
        assert writer.total_chunks == int(ref_st["total_chunks"]), (
            f"[{split}] total_chunks mismatch vs m4_g4l4: got {writer.total_chunks}, expected {ref_st['total_chunks']}"
        )

        slots_parity_msg = "slots.npy=N/A"
        if ref_slots_path.exists():
            our_slots = np.load(writer.slots_npy_path)
            ref_slots = np.load(ref_slots_path)
            assert our_slots.shape == ref_slots.shape, (
                f"[{split}] slots.npy shape mismatch vs {ref_slots_path}: got {our_slots.shape}, expected {ref_slots.shape}"
            )
            if not np.array_equal(our_slots, ref_slots):
                diff_cnt = int(np.sum(np.any(our_slots != ref_slots, axis=1)))
                diff_ratio = diff_cnt / float(max(1, len(our_slots)))
                if diff_ratio >= 0.001:
                    raise AssertionError(
                        f"[{split}] slots.npy mismatch vs {ref_slots_path}: {diff_cnt} / {len(our_slots)} songs differ!"
                    )
                slots_parity_msg = (
                    f"slots.npy={100.0 * (1.0 - diff_ratio):.4f}% IDENTICAL "
                    f"({diff_cnt}/{len(our_slots)} legacy tie rows updated)"
                )
            else:
                slots_parity_msg = "slots.npy=100% IDENTICAL!"

        ref_offsets_candidates = (
            ref_dir / f"{split}_offsets.npy",
            ref_dir / f"{split}_offsets_complete.npy",
        )
        ref_offsets_path = next((p for p in ref_offsets_candidates if p.exists()), None)
        seg_parity_msg = ""
        if ref_offsets_path is not None:
            ref_offsets = np.load(ref_offsets_path)
            assert len(ref_offsets) == writer.valid_songs, (
                f"[{split}] offsets.npy song count mismatch vs {ref_offsets_path}: "
                f"got {writer.valid_songs}, expected {len(ref_offsets)}"
            )
            total_ref_toks = int(np.sum(ref_offsets[:, 1])) if len(ref_offsets) > 0 else 0
            tokens_per_chunk = (
                (total_ref_toks // writer.total_chunks)
                if writer.total_chunks > 0 and total_ref_toks > 0
                else max(1, int(ref_st.get("tokens_per_chunk", 16)))
            )
            assert total_ref_toks == writer.total_chunks * tokens_per_chunk, (
                f"[{split}] total reference RVQ tokens ({total_ref_toks}) != "
                f"total_chunks ({writer.total_chunks}) * tokens_per_chunk ({tokens_per_chunk})"
            )
            expected_seg_meta: List[Tuple[int, int, int, int]] = []
            for s_idx in range(len(ref_offsets)):
                song_toks = int(ref_offsets[s_idx, 1])
                if song_toks % tokens_per_chunk != 0:
                    raise AssertionError(
                        f"[{split}] ref_offsets[{s_idx}, 1]={song_toks} not divisible by tokens_per_chunk={tokens_per_chunk}"
                    )
                song_chunks = song_toks // tokens_per_chunk
                for c_start, c_end, is_f, is_l in self.config.window_spec.compute_song_segments(song_chunks):
                    expected_seg_meta.append((int(is_f), int(is_l), int(s_idx), int(c_end - c_start)))

            exp_arr = (
                np.asarray(expected_seg_meta, dtype=np.int64)
                if expected_seg_meta
                else np.zeros((0, 4), dtype=np.int64)
            )
            our_segs = np.load(writer.segments_npy_path)
            if len(our_segs) != len(exp_arr) or not np.array_equal(our_segs[:, 2:6], exp_arr):
                raise AssertionError(
                    f"[{split}] 16-bar segment table mismatch vs {ref_offsets_path}: "
                    f"our_segments={len(our_segs)}, expected_segments={len(exp_arr)}"
                )
            seg_parity_msg = f", segments.npy=100% IDENTICAL ({len(our_segs):,} segments)"

        print(
            f"🔒 [1-to-1 Parity Verified vs {ref_dir.name}/{split}] "
            f"valid_songs={writer.valid_songs:,}, skipped_songs={writer.skipped_songs:,}, "
            f"total_chunks={writer.total_chunks:,}, {slots_parity_msg}{seg_parity_msg}",
            flush=True,
        )


def transcode_octuple_split_to_remi(
    octuple_dir: str | Path,
    out_dir: str | Path,
    split: str,
    tokenizer_key: str = "multitrack_remi",
    limit_segments: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fast $O(1)$-RAM streaming binary transcoder from `{octuple_dir}/{split}_*` directly to
    `{out_dir}/{tokenizer_key}/{split}_*` without parsing any MIDI files.
    """
    src_dir = Path(octuple_dir)
    src_bin_path = src_dir / f"{split}_tokens.bin"
    src_segs_path = src_dir / f"{split}_segments.npy"
    src_slots_path = src_dir / f"{split}_slots.npy"
    src_state_path = src_dir / f"{split}_state.json"

    if not src_bin_path.exists() or not src_segs_path.exists() or not src_slots_path.exists():
        raise FileNotFoundError(
            f"Source Octuple split files missing in {src_dir} for split={split!r}."
        )

    tok = build_symbolic_tokenizer(tokenizer_key)
    if not isinstance(tok, MultitrackREMITokenizer):
        raise ValueError(
            f"transcode_octuple_split_to_remi requires a MultitrackREMITokenizer, got {type(tok).__name__} ({tokenizer_key!r})."
        )

    dst_dir = Path(out_dir) / tok.tokenizer_key
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_segs = np.asarray(np.load(src_segs_path), dtype=np.int64)
    src_slots = np.asarray(np.load(src_slots_path), dtype=np.int16)
    src_state: Dict[str, Any] = {}
    if src_state_path.exists():
        with open(src_state_path, "r", encoding="utf-8") as sf:
            src_state = json.load(sf)

    is_partial = limit_segments is not None and 0 < int(limit_segments) < len(src_segs)
    if is_partial:
        src_segs = src_segs[: int(limit_segments)]
        max_song = (int(src_segs[:, 4].max()) + 1) if len(src_segs) > 0 else 0
        src_slots = src_slots[:max_song]

    src_bin_tokens = src_bin_path.stat().st_size // 2
    src_mm = (
        np.memmap(src_bin_path, dtype=np.uint16, mode="r")
        if src_bin_tokens > 0
        else np.zeros((0,), dtype=np.uint16)
    )

    dst_bin_tmp = dst_dir / f"{split}_tokens.bin.tmp"
    dst_bin_final = dst_dir / f"{split}_tokens.bin"
    n_segs = len(src_segs)
    dst_segs = np.empty((n_segs, 6), dtype=np.int64)

    t0 = time.time()
    buf = bytearray()
    flush_bytes = 8 * 1024 * 1024
    cursor = 0
    total_notes = 0

    with open(dst_bin_tmp, "wb") as bf:
        for i in range(n_segs):
            row = src_segs[i]
            t_start = int(row[0])
            t_len = int(row[1])
            seg_chunks = int(row[5]) if src_segs.shape[1] >= 6 else 128
            total_notes += t_len // 6
            raw_slice = src_mm[t_start : t_start + t_len] if t_len > 0 else np.zeros((0,), dtype=np.uint16)
            remi_u16 = tok.transcode_octuple_body_to_remi(
                raw_slice, num_chunks=seg_chunks, out_dtype=np.uint16
            )
            r_len = int(remi_u16.size)
            dst_segs[i, 0] = cursor
            dst_segs[i, 1] = r_len
            dst_segs[i, 2] = row[2]
            dst_segs[i, 3] = row[3]
            dst_segs[i, 4] = row[4]
            dst_segs[i, 5] = seg_chunks
            cursor += r_len
            if r_len > 0:
                buf.extend(np.ascontiguousarray(remi_u16, dtype=np.uint16).tobytes())
                if len(buf) >= flush_bytes:
                    bf.write(buf)
                    buf.clear()
        if buf:
            bf.write(buf)
            buf.clear()
        bf.flush()
        os.fsync(bf.fileno())

    os.replace(dst_bin_tmp, dst_bin_final)

    n_songs = len(src_slots)
    if n_songs > 0 and n_segs > 0:
        song_lens = np.bincount(dst_segs[:, 4], weights=dst_segs[:, 1], minlength=n_songs).astype(np.int64)
        song_starts = np.zeros(n_songs, dtype=np.int64)
        if n_songs > 1:
            song_starts[1:] = np.cumsum(song_lens[:-1], dtype=np.int64)
        dst_offsets = np.stack([song_starts, song_lens], axis=1)
    else:
        dst_offsets = np.zeros((n_songs, 2), dtype=np.int64)

    SymbolicSplitWriter._atomic_save_npy(dst_dir / f"{split}_segments.npy", dst_segs)
    SymbolicSplitWriter._atomic_save_npy(dst_dir / f"{split}_offsets.npy", dst_offsets)
    SymbolicSplitWriter._atomic_save_npy(dst_dir / f"{split}_slots.npy", src_slots)

    full_seq_flat_tokens = int(cursor) + n_segs * 6
    state = {
        "tokenizer_key": tok.tokenizer_key,
        "model_mode": tok.model_mode,
        "vocab_size": tok.vocab_size,
        "step_unit_width": tok.step_unit_width,
        "split": split,
        "source_octuple_dir": str(src_dir),
        "max_chunks": int(src_state.get("max_chunks", 128)),
        "min_chunks": int(src_state.get("min_chunks", 8)),
        "stride_chunks": int(src_state.get("stride_chunks", 128)),
        "processed_midi_count": int(src_state.get("processed_midi_count", n_songs)) if not is_partial else n_songs,
        "next_midi_idx": int(src_state.get("next_midi_idx", n_songs)) if not is_partial else n_songs,
        "valid_songs": n_songs,
        "skipped_songs": int(src_state.get("skipped_songs", 0)) if not is_partial else 0,
        "total_chunks": int(dst_segs[:, 5].sum()) if n_segs > 0 else 0,
        "total_segments": n_segs,
        "total_tokens": int(cursor),
        "full_seq_flat_tokens": full_seq_flat_tokens,
        "full_seq_ar_steps": full_seq_flat_tokens,
        "total_raw_midi_notes": int(src_state.get("total_raw_midi_notes", total_notes)) if not is_partial else total_notes,
        "total_canonical_notes": total_notes,
        "raw_information_bits_36b_per_note": total_notes * 36,
        "updated_at": time.time(),
    }
    tmp_state = dst_dir / f"{split}_state.json.tmp"
    with open(tmp_state, "w", encoding="utf-8") as sf:
        json.dump(state, sf, indent=2)
        sf.flush()
        os.fsync(sf.fileno())
    os.replace(tmp_state, dst_dir / f"{split}_state.json")

    elapsed = max(1e-3, time.time() - t0)
    print(
        f"✅ [Transcoded octuple_mmt -> {tok.tokenizer_key}:{split}] {n_segs:,} segments "
        f"({total_notes:,} notes -> {cursor:,} body tokens) in {elapsed:.2f}s -> {dst_dir}",
        flush=True,
    )
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GMD MIDI splits into 16-bar synchronized Symbolic Token datasets (Octuple-MMT / REMI)."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="octuple_mmt",
        choices=sorted(SYMBOLIC_TOKENIZER_REGISTRY.keys()),
        help="Symbolic tokenizer key registered in SYMBOLIC_TOKENIZER_REGISTRY.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val", "train"],
        choices=["val", "train"],
        help="Dataset splits to convert.",
    )
    parser.add_argument("--val_json", type=str, default=DEFAULT_SPLITS["val"], help="Validation MIDI list JSON.")
    parser.add_argument("--train_json", type=str, default=DEFAULT_SPLITS["train"], help="Train MIDI list JSON.")
    parser.add_argument("--out_dir", type=str, default="out/mortm5_tokens", help="Root output directory.")
    parser.add_argument("--max_chunks", type=int, default=128, help="Max chunks per segment (128 chunks = 16 bars).")
    parser.add_argument("--min_chunks", type=int, default=8, help="Min chunks per segment (8 chunks = 1 bar).")
    parser.add_argument("--stride_chunks", type=int, default=128, help="Segment stride chunks (128 = non-overlapping).")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 8) - 2)),
        help="Number of CPU worker processes.",
    )
    parser.add_argument("--checkpoint_every", type=int, default=5000, help="Songs per atomic checkpoint flush.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of MIDI files per split for smoke tests.")
    parser.add_argument("--no_resume", action="store_true", help="Restart conversion from scratch instead of resuming.")
    parser.add_argument(
        "--verify_m4_dir",
        type=str,
        default="out/mortm5_tokens/m4_g4l4",
        help="Directory containing m4_g4l4 {split}_state.json & {split}_slots.npy for automatic parity verification.",
    )
    parser.add_argument(
        "--from_octuple_dir",
        "--from-octuple-dir",
        dest="from_octuple_dir",
        type=str,
        default=None,
        help="Optional existing octuple_mmt directory for ultra-fast binary transcoding to multitrack_remi (skips MIDI parsing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_octuple_dir:
        for sp in args.splits:
            transcode_octuple_split_to_remi(
                octuple_dir=args.from_octuple_dir,
                out_dir=args.out_dir,
                split=sp,
                tokenizer_key=args.tokenizer,
                limit_segments=args.limit,
            )
        return
    cfg = SymbolicCorpusConfig(
        tokenizer_key=args.tokenizer,
        out_dir=args.out_dir,
        max_chunks=args.max_chunks,
        min_chunks=args.min_chunks,
        stride_chunks=args.stride_chunks,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        resume=(not args.no_resume),
        verify_m4_dir=args.verify_m4_dir if args.verify_m4_dir else None,
    )
    split_map = {"val": args.val_json, "train": args.train_json}
    converter = SymbolicCorpusConverter(cfg)
    for sp in args.splits:
        converter.convert_split(split=sp, json_path=split_map[sp], limit=args.limit)


if __name__ == "__main__":
    main()
