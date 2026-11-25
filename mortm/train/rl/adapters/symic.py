"""SyMIc-G4L4 + Depth (`AblationTransformer2` + `MidiVision`) Adapters for Unified CARL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mortm.models.modules.attention import pad_input, unpad_input
from mortm.models.modules.config import (
    EOS_TOKEN_ID,
    INST_EMPTY_TOKEN_ID,
    MORTM5Args,
    NUM_SLOT_TOKENS,
    PAD_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
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
        ana_songs_list = [int(s) for s in shuffled_songs[:cut].tolist()]
        gem_songs_list = [int(s) for s in shuffled_songs[cut:].tolist()]
        ana_song_set = set(ana_songs_list)

        ana_by_song: Dict[int, List[int]] = {}
        gem_by_song: Dict[int, List[int]] = {}
        for w_idx in valid_win_indices.tolist():
            s_id = int(song_ids[w_idx])
            if s_id in ana_song_set:
                ana_by_song.setdefault(s_id, []).append(int(w_idx))
            else:
                gem_by_song.setdefault(s_id, []).append(int(w_idx))

        half_lim = max(1, int(limit) // 2) if (limit is not None and int(limit) > 0) else None
        ana_active, ana_win_songs = self._collect_song_balanced_active_windows(
            ana_by_song, ana_songs_list, rng=rng, max_count=half_lim
        )
        gem_active, gem_win_songs = self._collect_song_balanced_active_windows(
            gem_by_song, gem_songs_list, rng=rng, max_count=half_lim
        )
        if not ana_active or not gem_active:
            raise RuntimeError(
                f"After filtering silent prompt/continuation windows, got ana={len(ana_active)}, gem={len(gem_active)}."
            )

        return (
            LazyIndexedSamplePool(ana_active, self._load_window_example, song_ids=ana_win_songs),
            LazyIndexedSamplePool(gem_active, self._load_window_example, song_ids=gem_win_songs),
        )

    def _is_active_window(self, mmap: np.ndarray, w_idx: int) -> bool:
        tpc = self._vq_groups * self._vq_layers
        p_chunks = self._prompt_chunks
        tok_start = int(self._dataset.windows[w_idx, 0])
        tok_len = int(self._dataset.windows[w_idx, 1])
        raw = np.asarray(mmap[tok_start : tok_start + tok_len]).reshape(-1, tpc)
        prompt_c = raw[:p_chunks]
        cont_c = raw[p_chunks : self._total_chunks]
        if prompt_c.shape[0] > 1 and bool(np.all(prompt_c == prompt_c[0])):
            return False
        if cont_c.shape[0] > 1:
            same_first = float(np.all(cont_c == cont_c[0], axis=1).mean())
            same_last = float(np.all(cont_c == cont_c[-1], axis=1).mean())
            if max(same_first, same_last) >= 0.85:
                return False
        return True

    def _collect_song_balanced_active_windows(
        self,
        windows_by_song: Dict[int, List[int]],
        ordered_songs: Sequence[int],
        rng: np.random.Generator,
        max_count: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """Round-robin across unique songs (1 active window/song first) so 10k+ pools maximize song diversity."""
        mmap = self._dataset._get_memmap()
        shuffled_per_song: List[Tuple[int, List[int]]] = []
        for s_id in ordered_songs:
            w_list = windows_by_song.get(int(s_id))
            if not w_list:
                continue
            if len(w_list) > 1:
                perm_w = [w_list[int(i)] for i in rng.permutation(len(w_list))]
            else:
                perm_w = list(w_list)
            shuffled_per_song.append((int(s_id), perm_w))

        cursors = [0] * len(shuffled_per_song)
        kept_wins: List[int] = []
        kept_songs: List[int] = []
        made_progress = True

        while made_progress:
            made_progress = False
            for idx, (s_id, w_list) in enumerate(shuffled_per_song):
                if max_count is not None and len(kept_wins) >= max_count:
                    return kept_wins, kept_songs
                cur = cursors[idx]
                while cur < len(w_list):
                    w_cand = w_list[cur]
                    cur += 1
                    if self._is_active_window(mmap, w_cand):
                        kept_wins.append(int(w_cand))
                        kept_songs.append(s_id)
                        made_progress = True
                        break
                cursors[idx] = cur
        return kept_wins, kept_songs


def _scan_midi_entry_worker(
    args: Tuple[int, str, int, int, int, int, int, int, float, int],
) -> List[Tuple[int, int, Tuple[int, ...], np.ndarray]]:
    """Top-level multiprocess worker extracting valid non-blank 16-bar note windows from one MIDI."""
    (
        raw_idx,
        path_str,
        prompt_bars,
        total_bars,
        total_chunks,
        min_active_tracks,
        min_prompt_notes,
        min_cont_notes,
        max_blank_ratio,
        per_song_cap,
    ) = args
    if not path_str or not os.path.isfile(path_str):
        return []
    try:
        num_chunks, notes_arr, slot_codes = extract_quantized_notes_and_slots(path_str)
    except Exception:
        return []
    if num_chunks < total_chunks or len(notes_arr) == 0:
        return []
    slot_tuple = tuple(int(c) for c in slot_codes.tolist())
    if sum(1 for c in slot_tuple if c != SLOT_CODE_EMPTY) < min_active_tracks:
        return []

    cont_bars = max(1, total_bars - prompt_bars)
    found: List[Tuple[int, int, Tuple[int, ...], np.ndarray]] = []
    for c0 in range(0, num_chunks - total_chunks + 1, total_chunks):
        if len(found) >= per_song_cap:
            break
        seg_notes = extract_segment_canonical_notes(
            notes_arr,
            chunk_start=c0,
            chunk_end=c0 + total_chunks,
            max_bars=total_bars,
        )
        if len(seg_notes) == 0:
            continue
        bars = seg_notes[:, 0]
        p_mask = bars < prompt_bars
        c_mask = bars >= prompt_bars
        if int(np.sum(p_mask)) < min_prompt_notes or int(np.sum(c_mask)) < min_cont_notes:
            continue
        active_cont_bars = len(np.unique(bars[c_mask]))
        blank_ratio = 1.0 - (float(active_cont_bars) / float(cont_bars))
        if blank_ratio > max_blank_ratio:
            continue

        starts = seg_notes[:, 0] * 192 + seg_notes[:, 1]
        ends = np.minimum(total_chunks * 24, starts + seg_notes[:, 4])
        notes_5col = np.stack(
            [seg_notes[:, 2], starts, ends, seg_notes[:, 3], seg_notes[:, 5]], axis=1
        ).astype(np.int32)
        found.append((int(raw_idx), int(c0), slot_tuple, notes_5col))
    return found


class BatchedMidiSamplePool(CARLDataSamplePool):
    """Sample pool for `SyMIcMidiRepository` that batch-encodes uncached windows on GPU in a single pass."""

    def __init__(
        self,
        indices: Sequence[int] | np.ndarray,
        single_loader: Any,
        batch_loader: Any,
        song_ids: Optional[Sequence[int] | np.ndarray] = None,
    ) -> None:
        self._indices = np.asarray(indices, dtype=np.int64).copy()
        self._single_loader = single_loader
        self._batch_loader = batch_loader
        self._unique_songs: Optional[int] = (
            int(np.unique(np.asarray(song_ids, dtype=np.int64)).size)
            if song_ids is not None
            else None
        )

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, index: int) -> CARLExample:
        return self._single_loader(int(self._indices[int(index)]))

    def sample(self, count: int, rng: np.random.Generator) -> List[CARLExample]:
        total = len(self)
        if total == 0 or count <= 0:
            return []
        picked_pos = np.atleast_1d(rng.choice(total, size=count, replace=(total < count)))
        spec_indices = [int(self._indices[int(p)]) for p in picked_pos]
        return self._batch_loader(spec_indices)

    def unique_song_count(self) -> Optional[int]:
        return self._unique_songs


class SyMIcMidiRepository(CARLDataRepository):
    """Lazy on-the-fly MIDI -> `MidiVision` repository with non-blank prompt & continuation filtering."""

    def __init__(
        self,
        midi_manifest_path: str | Path,
        model: AblationTransformer2,
        vq_model: MidiVision,
        device: torch.device,
        prompt_bars: int = 4,
        total_bars: int = 16,
        min_active_tracks: int = 2,
        min_prompt_notes: int = 8,
        min_cont_notes: int = 8,
        max_cont_blank_bar_ratio: float = 0.5,
        max_windows_per_song: int = 4,
        max_cache_items: int = 4096,
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
        self._prompt_bars = int(prompt_bars)
        self._total_bars = int(total_bars)
        self._prompt_chunks = self._prompt_bars * CHUNKS_PER_BAR
        self._total_chunks = self._total_bars * CHUNKS_PER_BAR
        self._min_active_tracks = max(1, int(min_active_tracks))
        self._min_prompt_notes = max(1, int(min_prompt_notes))
        self._min_cont_notes = max(1, int(min_cont_notes))
        self._max_blank_ratio = float(max_cont_blank_bar_ratio)
        self._max_windows_per_song = max(1, int(max_windows_per_song))
        self._max_cache_items = max(16, int(max_cache_items))
        self._vq_layers = int(model.vq_layers)
        self._prefix_len = 1 + (NUM_SLOT_TOKENS if bool(model.use_instrument_tokens) else 0)
        self._prompt_soft_len = self._prefix_len + self._prompt_chunks * self._vq_layers
        self._specs: List[Tuple[int, int, Tuple[int, ...], np.ndarray]] = []
        self._cache: OrderedDict[int, CARLExample] = OrderedDict()

    def _put_cache(self, spec_idx: int, ex: CARLExample) -> None:
        if spec_idx in self._cache:
            self._cache.move_to_end(spec_idx)
            return
        if len(self._cache) >= self._max_cache_items:
            self._cache.popitem(last=False)
        self._cache[spec_idx] = ex

    @torch.no_grad()
    def _encode_specs_batch(self, spec_indices: Sequence[int]) -> List[CARLExample]:
        if not spec_indices:
            return []
        missing = sorted({int(i) for i in spec_indices if int(i) not in self._cache})
        if missing:
            rolls = [
                rasterize_notes_to_chunks_float(self._total_chunks, self._specs[m_i][3])
                for m_i in missing
            ]
            stacked = np.concatenate(rolls, axis=0)
            vq_dev = next(self._vq.parameters()).device
            tok_parts: List[Tensor] = []
            for start in range(0, stacked.shape[0], 512):
                x_t = torch.from_numpy(stacked[start : start + 512]).to(
                    device=vq_dev, dtype=torch.float32
                )
                tok_parts.append(self._vq.encode_tokens(x_t, include_offset=True).clone())
            raw_rvq = torch.cat(tok_parts, dim=0)

            l_cnt = int(self._model.vq_layers)
            g_cnt = int(self._model.vq_groups)
            k_sz = int(self._model.codebook_size)
            tok_off = int(self._model.token_offset)
            vq_off = int(getattr(self._vq, "token_offset", RVQ_TOKEN_OFFSET))

            rvq_4d = raw_rvq.view(len(missing), self._total_chunks, l_cnt, g_cnt).to(dtype=torch.long) + (
                tok_off - vq_off
            )
            if bool(getattr(self._model, "separate_rvq_layers", False)):
                layer_offsets = (
                    torch.arange(l_cnt, device=vq_dev, dtype=torch.long).view(1, 1, l_cnt, 1)
                    * (g_cnt * k_sz)
                )
                rvq_4d = rvq_4d + layer_offsets
            body_np = rvq_4d.reshape(len(missing), self._total_chunks * l_cnt, g_cnt).cpu().numpy().astype(np.int64)

            for pos, m_i in enumerate(missing):
                song_id, c0, slot_tuple, _ = self._specs[m_i]
                prefix = self._model.build_prefix_tokens(slot_codes=slot_tuple, batch_size=1, device="cpu")
                prefix_2d = prefix[0].numpy().astype(np.int64)
                full_np = np.concatenate([prefix_2d, body_np[pos]], axis=0)
                prompt_np = np.ascontiguousarray(full_np[: self._prompt_soft_len], dtype=np.int64)
                self._put_cache(
                    m_i,
                    CARLExample(
                        example_id=f"midi_s{song_id:07d}_c{c0:04d}",
                        song_id=song_id,
                        prompt_tokens=prompt_np,
                        human_full_tokens=full_np,
                        slot_codes=slot_tuple,
                    ),
                )

        out: List[CARLExample] = []
        for idx in spec_indices:
            i_int = int(idx)
            self._cache.move_to_end(i_int)
            out.append(self._cache[i_int])
        return out

    @torch.no_grad()
    def _encode_spec(self, spec_idx: int) -> CARLExample:
        return self._encode_specs_batch([spec_idx])[0]

    def load_disjoint_pools(
        self,
        limit: Optional[int] = 10000,
        seed: int = 42,
    ) -> Tuple[CARLDataSamplePool, CARLDataSamplePool]:
        import json

        with self._manifest_path.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(entries)).tolist()
        target_total = (
            max(2, int(limit))
            if limit is not None and int(limit) > 0
            else max(2, min(len(entries), 10000))
        )
        # Prioritize 1 window per unique song when manifest is large (>= target_total * 2),
        # while allowing up to `max_windows_per_song` on small test manifests.
        if len(entries) >= target_total * 2:
            per_song_cap = 1
        else:
            per_song_cap = max(1, min(self._max_windows_per_song, target_total // 2))

        self._specs = []
        self._cache = OrderedDict()
        song_to_spec_indices: Dict[int, List[int]] = {}

        def _ingest_song_windows(
            found_wins: List[Tuple[int, int, Tuple[int, ...], np.ndarray]],
        ) -> bool:
            for raw_id, c0, s_tup, n_5 in found_wins:
                if len(self._specs) >= target_total and len(song_to_spec_indices) >= 2:
                    return True
                s_idx = len(self._specs)
                self._specs.append((raw_id, c0, s_tup, n_5))
                song_to_spec_indices.setdefault(raw_id, []).append(s_idx)
            return len(self._specs) >= target_total and len(song_to_spec_indices) >= 2

        use_mp = len(perm) > 64 and target_total > 32
        if use_mp:
            workers = min(12, os.cpu_count() or 4)
            scan_chunk = max(target_total * 2, 512)
            cursor = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                while cursor < len(perm):
                    sub_perm = perm[cursor : cursor + scan_chunk]
                    cursor += len(sub_perm)
                    tasks = [
                        (
                            int(r_i),
                            entries[r_i] if isinstance(entries[r_i], str) else str(entries[r_i].get("path", "")),
                            self._prompt_bars,
                            self._total_bars,
                            self._total_chunks,
                            self._min_active_tracks,
                            self._min_prompt_notes,
                            self._min_cont_notes,
                            self._max_blank_ratio,
                            per_song_cap,
                        )
                        for r_i in sub_perm
                    ]
                    done_scanning = False
                    for res in pool.map(_scan_midi_entry_worker, tasks, chunksize=64):
                        if res and _ingest_song_windows(res):
                            done_scanning = True
                            break
                    if done_scanning:
                        break
        else:
            for r_i in perm:
                p_str = entries[r_i] if isinstance(entries[r_i], str) else str(entries[r_i].get("path", ""))
                res = _scan_midi_entry_worker(
                    (
                        int(r_i),
                        p_str,
                        self._prompt_bars,
                        self._total_bars,
                        self._total_chunks,
                        self._min_active_tracks,
                        self._min_prompt_notes,
                        self._min_cont_notes,
                        self._max_blank_ratio,
                        per_song_cap,
                    )
                )
                if res and _ingest_song_windows(res):
                    break

        unique_songs = list(song_to_spec_indices.keys())
        if len(unique_songs) < 2:
            raise RuntimeError(
                f"SyMIcMidiRepository found only {len(unique_songs)} valid non-blank songs (need >= 2)."
            )
        shuffled_songs = rng.permutation(np.asarray(unique_songs, dtype=np.int64))
        cut = max(1, len(shuffled_songs) // 2)
        ana_songs = set(int(s) for s in shuffled_songs[:cut].tolist())

        ana_idx: List[int] = []
        ana_song_ids: List[int] = []
        gem_idx: List[int] = []
        gem_song_ids: List[int] = []
        for s_id, s_specs in song_to_spec_indices.items():
            if s_id in ana_songs:
                ana_idx.extend(s_specs)
                ana_song_ids.extend([s_id] * len(s_specs))
            else:
                gem_idx.extend(s_specs)
                gem_song_ids.extend([s_id] * len(s_specs))

        return (
            BatchedMidiSamplePool(ana_idx, self._encode_spec, self._encode_specs_batch, song_ids=ana_song_ids),
            BatchedMidiSamplePool(gem_idx, self._encode_spec, self._encode_specs_batch, song_ids=gem_song_ids),
        )


# ======================================================================
# 2. Critic (`AnalysisSyMIc` + `SyMIcCriticPoolHead` Strategies + `SyMIcCriticAdapter`)
# ======================================================================


class MultiHeadAttentiveStatsPool(nn.Module):
    """Length-invariant multi-head attention pooling computing 1st & 2nd moments (`[mean, std]`).

    Summarizing a variable-length segment (`[B, S, D]`, e.g., 4-bar prompt RVQ tokens vs
    12-bar continuation RVQ tokens) via softmax-normalized weights `sum_t alpha_{t,h} = 1`
    maps both segments into the exact same `[B, 2 * proj_dim]` feature space regardless of
    bar count, capturing both macro density/register (`mean`) and rhythmic/harmonic dispersion (`std`).
    """

    def __init__(
        self,
        d_model: int,
        max_proj_dim: int = 256,
        num_heads: int = 8,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        proj_dim = min(int(d_model), int(max_proj_dim))
        eff_heads = math.gcd(proj_dim, max(1, int(num_heads))) or 1

        self.d_model = int(d_model)
        self.proj_dim = proj_dim
        self.num_heads = eff_heads
        self.head_dim = proj_dim // eff_heads
        self.out_dim = 2 * proj_dim
        self.eps = float(eps)

        self.norm = nn.LayerNorm(self.d_model)
        self.attn = nn.Sequential(
            nn.Linear(self.d_model, self.proj_dim),
            nn.Tanh(),
            nn.Linear(self.proj_dim, self.num_heads),
        )
        self.val_proj = nn.Linear(self.d_model, self.proj_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(1) == 0:
            raise ValueError(
                f"MultiHeadAttentiveStatsPool expects non-empty [B, S, D], got {tuple(x.shape)}."
            )
        bsz, seq_len, _ = x.shape
        h = self.norm(x.float())
        weights = torch.softmax(self.attn(h), dim=1).unsqueeze(-1)  # [B, S, H, 1]
        vals = F.gelu(self.val_proj(h)).view(bsz, seq_len, self.num_heads, self.head_dim)

        mean = (weights * vals).sum(dim=1)  # [B, H, head_dim]
        var = (weights * (vals - mean.unsqueeze(1)).square()).sum(dim=1)
        std = torch.sqrt(var.clamp_min(self.eps))
        return torch.cat([mean.flatten(1), std.flatten(1)], dim=-1)  # [B, 2 * proj_dim]


class SyMIcCriticPoolHead(nn.Module, ABC):
    """Strategy interface (OCP) aggregating prompt & continuation states into 1D AI/Human logits."""

    @property
    @abstractmethod
    def pool_mode(self) -> str:
        """Canonical registry identifier (`'attn'`, `'mil'`, ...)."""

    @abstractmethod
    def forward(self, prompt_hidden: Tensor, cont_hidden: Tensor) -> Tensor:
        """Return 1D float32 discriminator logits `[B]` (`> 0` = AI, `< 0` = Human)."""


class PromptContinuationAttentionPoolHead(SyMIcCriticPoolHead):
    """Shared Prompt-vs-Continuation `AttentionPool` relational discriminator head (`'attn'`).

    Pools the 4-bar prompt RVQ states into `p` and the 12-bar continuation RVQ states into `c`
    through the **same** `MultiHeadAttentiveStatsPool`, then scores `[c, c - p, |c - p|, p_hat * c_hat]`
    so the Critic directly detects macro prompt-continuation density/style mismatches (e.g., a sparse
    6.5-note/bar prompt inflating to 39.9 notes/bar) without brittle rule-based penalties.
    """

    def __init__(
        self,
        d_model: int,
        max_proj_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.shared_pool = MultiHeadAttentiveStatsPool(
            d_model=d_model,
            max_proj_dim=max_proj_dim,
            num_heads=num_heads,
        )
        s_dim = self.shared_pool.out_dim
        rel_dim = 4 * s_dim
        hidden_dim = self.shared_pool.proj_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(rel_dim),
            nn.Linear(rel_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def pool_mode(self) -> str:
        return "attn"

    def forward(self, prompt_hidden: Tensor, cont_hidden: Tensor) -> Tensor:
        with torch.autocast(device_type=cont_hidden.device.type, enabled=False):
            p_vec = self.shared_pool(prompt_hidden)
            c_vec = self.shared_pool(cont_hidden)
            diff = c_vec - p_vec
            p_unit = F.normalize(p_vec, p=2.0, dim=-1, eps=1e-5)
            c_unit = F.normalize(c_vec, p=2.0, dim=-1, eps=1e-5)
            rel_feat = torch.cat([c_vec, diff, diff.abs(), p_unit * c_unit], dim=-1)
            return self.classifier(rel_feat).squeeze(-1).float()


class ContinuationMILPoolHead(SyMIcCriticPoolHead):
    """Continuation-only LogSumExp Multiple-Instance Learning head (`'mil'`)."""

    def __init__(self, d_model: int, r_init: float = 2.0) -> None:
        super().__init__()
        self.mil = MILPool(int(d_model), r_init=r_init)

    @property
    def pool_mode(self) -> str:
        return "mil"

    def forward(self, prompt_hidden: Tensor, cont_hidden: Tensor) -> Tensor:
        del prompt_hidden
        with torch.autocast(device_type=cont_hidden.device.type, enabled=False):
            cont_fp = cont_hidden.to(dtype=self.mil.scorer[0].weight.dtype)
            tok_scores = self.mil.scorer(cont_fp).squeeze(-1).float()
            r_val = self.mil.log_r.exp().clamp(0.05, 20.0)
            n_cont = float(tok_scores.size(1))
            return (torch.logsumexp(r_val * tok_scores, dim=1) - math.log(n_cont)) / r_val


PoolHeadFactory = Callable[[int, float], SyMIcCriticPoolHead]


class SyMIcCriticPoolRegistry:
    """Open/Closed registry mapping pooling strategy keys (`'attn'`, `'mil'`) to factories."""

    def __init__(self) -> None:
        self._factories: Dict[str, PoolHeadFactory] = {}

    def register(self, key: str, factory: PoolHeadFactory) -> None:
        norm = str(key).strip().lower()
        if not norm:
            raise ValueError("Critic pool key must be non-empty.")
        self._factories[norm] = factory

    def build(self, key: str, d_model: int, mil_r_init: float = 2.0) -> SyMIcCriticPoolHead:
        norm = str(key).strip().lower()
        factory = self._factories.get(norm)
        if factory is None:
            raise ValueError(
                f"Unsupported SyMIc Critic pool_mode '{key}'. Available: {self.available_keys()}"
            )
        return factory(int(d_model), float(mil_r_init))

    def available_keys(self) -> Tuple[str, ...]:
        return tuple(self._factories.keys())


DEFAULT_SYMIC_CRITIC_POOL_REGISTRY = SyMIcCriticPoolRegistry()
DEFAULT_SYMIC_CRITIC_POOL_REGISTRY.register(
    "attn",
    lambda d_model, _r_init: PromptContinuationAttentionPoolHead(d_model=d_model),
)
DEFAULT_SYMIC_CRITIC_POOL_REGISTRY.register(
    "mil",
    lambda d_model, r_init: ContinuationMILPoolHead(d_model=d_model, r_init=r_init),
)
SYMIC_CRITIC_POOLS: Tuple[str, ...] = DEFAULT_SYMIC_CRITIC_POOL_REGISTRY.available_keys()


class AnalysisSyMIc(nn.Module):
    """SyMIc causal discriminator with native Attention LoRA + pluggable `SyMIcCriticPoolHead`.

    Passes `[Prefix | Prompt (4 bars) | Continuation (12 bars)]` through the causal SyMIc
    decoder, slices musical prompt states (`prefix_soft_len .. prompt_soft_len - 1`) and
    continuation states (`prompt_soft_len .. full_soft_len - 1`), and delegates logit aggregation
    to `self.pool_head` (default `'attn'`: shared Prompt-vs-Continuation `AttentionPool`).
    """

    def __init__(
        self,
        backbone: AblationTransformer2,
        prompt_soft_len: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        mil_r_init: float = 2.0,
        pool_mode: str = "attn",
        prefix_soft_len: Optional[int] = None,
        pool_head: Optional[SyMIcCriticPoolHead] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prompt_soft_len = int(prompt_soft_len)
        if prefix_soft_len is None:
            default_prefix = 1 + (
                NUM_SLOT_TOKENS if bool(getattr(backbone, "use_instrument_tokens", True)) else 0
            )
            self.prefix_soft_len = default_prefix if self.prompt_soft_len > default_prefix else 0
        else:
            self.prefix_soft_len = int(prefix_soft_len)
        if not (0 <= self.prefix_soft_len < self.prompt_soft_len):
            raise ValueError(
                f"Expected 0 <= prefix_soft_len ({self.prefix_soft_len}) < prompt_soft_len ({self.prompt_soft_len})."
            )

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        inject_attention_lora(
            self.backbone.decoder.layers,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        dev = self.backbone.embedding.weight.device
        resolved_head = pool_head or DEFAULT_SYMIC_CRITIC_POOL_REGISTRY.build(
            pool_mode, d_model=int(self.backbone.args.d_model), mil_r_init=mil_r_init
        )
        self.pool_head: SyMIcCriticPoolHead = resolved_head.to(dev)

    @property
    def pool_mode(self) -> str:
        return self.pool_head.pool_mode

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
        prompt_hidden = dec_out[:, self.prefix_soft_len : self.prompt_soft_len, :]
        cont_hidden = dec_out[:, self.prompt_soft_len :, :]
        return self.pool_head(prompt_hidden, cont_hidden)


class SyMIcPianoRollCanonicalizer:
    """Experimental diagnostic filter (`Decode -> Pixelwise Discretize -> Re-Encode`) for `SyMIcCriticAdapter`.

    Note: `MidiVision` is a lossy neural RVQ autoencoder, so `Decode -> Discretize -> Re-Encode`
    is NOT strictly idempotent (each pass incurs slight onset/sustain/velocity reconstruction drift,
    especially in fine residual layers `L1..L3`) and adds CNN decode+encode overhead. Keep disabled
    (`canonicalize_pianoroll=False`) in standard CARL runs unless explicitly benchmarking round-trip behavior.
    """

    def __init__(
        self,
        vq_model: MidiVision,
        prompt_soft_len: int,
        vq_layers: int = 4,
        vq_groups: int = 4,
        codebook_size: int = 2048,
        token_offset: int = 133,
        separate_rvq_layers: bool = True,
        use_instrument_tokens: bool = True,
        decode_chunk_size: int = 512,
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
        self._codebook_size = int(codebook_size)
        self._token_offset = int(token_offset)
        self._vq_offset = int(getattr(vq_model, "token_offset", RVQ_TOKEN_OFFSET))
        self._separate_rvq_layers = bool(separate_rvq_layers)
        self._use_instrument_tokens = bool(use_instrument_tokens)
        self._decode_chunk_size = max(16, int(decode_chunk_size))

    @torch.no_grad()
    def canonicalize_sequences(self, seq_3d: Tensor) -> Tensor:
        if seq_3d.dim() != 3:
            raise ValueError(f"Expected 3D [B, S, G] tensor, got {tuple(seq_3d.shape)}.")
        bsz, seq_len, g_cnt = seq_3d.shape
        if seq_len <= self._prompt_soft_len:
            return seq_3d

        cont = seq_3d[:, self._prompt_soft_len :, :]
        cont_steps = cont.size(1)
        if cont_steps % self._vq_layers != 0:
            raise ValueError(
                f"Continuation soft-token count ({cont_steps}) must be divisible by vq_layers ({self._vq_layers})."
            )
        n_chunks = cont_steps // self._vq_layers
        vq_dev = next(self._vq.parameters()).device

        if (
            self._use_instrument_tokens
            and self._token_offset > INST_EMPTY_TOKEN_ID
            and seq_len >= 1 + NUM_SLOT_TOKENS
        ):
            slot_active = (seq_3d[:, 1 : 1 + NUM_SLOT_TOKENS, 0] != INST_EMPTY_TOKEN_ID).to(device=vq_dev)
        else:
            slot_active = torch.ones((bsz, 4), dtype=torch.bool, device=vq_dev)

        chan_active = torch.stack([slot_active, slot_active], dim=-1).reshape(bsz, 1, 8, 1, 1)
        flat_chan_active = chan_active.expand(bsz, n_chunks, 8, 1, 1).reshape(bsz * n_chunks, 8, 1, 1)

        self._vq.eval()
        flat_in = cont.to(device=vq_dev, dtype=torch.long).reshape(bsz * n_chunks, self._vq_layers, g_cnt)
        re_parts: List[Tensor] = []
        step = self._decode_chunk_size
        for start in range(0, flat_in.size(0), step):
            stop = min(start + step, flat_in.size(0))
            logits = self._vq.decode_tokens(flat_in[start:stop], token_offset=self._token_offset)
            probs = torch.sigmoid(logits.float()) * flat_chan_active[start:stop].float()

            on = probs[:, 0::2, :, :]
            su = probs[:, 1::2, :, :]
            on_mask = on >= 0.5
            su_mask = (su >= 0.5) | on_mask

            vel_127 = torch.clamp(torch.round((on - 0.5) * 2.0 * 127.0), 1.0, 127.0)
            on_u8 = torch.round((0.5 + 0.5 * (vel_127 / 127.0)) * 254.0) / 254.0
            disc_on = torch.where(on_mask, on_u8, torch.zeros_like(on))
            disc_su = torch.where(su_mask, torch.ones_like(su), torch.zeros_like(su))
            disc_roll = torch.stack([disc_on, disc_su], dim=2).reshape(stop - start, 8, 128, 24)

            # Clone outside `encode_tokens`'s `@torch.inference_mode()` so autograd can save indices in OddRound.
            re_parts.append(self._vq.encode_tokens(disc_roll, include_offset=True).clone())

        re_rvq = torch.cat(re_parts, dim=0).view(bsz, n_chunks, self._vq_layers, g_cnt).to(dtype=torch.long)
        re_rvq = re_rvq + (self._token_offset - self._vq_offset)
        if self._separate_rvq_layers:
            layer_offsets = (
                torch.arange(self._vq_layers, device=vq_dev, dtype=torch.long).view(1, 1, self._vq_layers, 1)
                * (g_cnt * self._codebook_size)
            )
            re_rvq = re_rvq + layer_offsets
        re_cont = re_rvq.reshape(bsz, cont_steps, g_cnt).to(device=seq_3d.device)
        return torch.cat([seq_3d[:, : self._prompt_soft_len, :], re_cont], dim=1).clone()


class SyMIcCriticAdapter(CARLCriticAdapter):
    """`CARLCriticAdapter` wrapping `AnalysisSyMIc` (with optional discrete piano-roll canonicalization)."""

    def __init__(
        self,
        ana_model: AnalysisSyMIc,
        device: torch.device,
        canonicalizer: Optional[SyMIcPianoRollCanonicalizer] = None,
    ) -> None:
        self._ana = ana_model
        self._device = device
        self._canonicalizer = canonicalizer

    @property
    def pool_mode(self) -> str:
        return self._ana.pool_mode

    @property
    def uses_pianoroll_canonicalizer(self) -> bool:
        return self._canonicalizer is not None

    @classmethod
    def from_pretrained(
        cls,
        config_source: str | dict | MORTM5Args,
        ckpt_path: Optional[str | Path],
        device: torch.device,
        prompt_bars: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        pool_mode: str = "attn",
        vq_model: Optional[MidiVision] = None,
        canonicalize_pianoroll: bool = False,
    ) -> "SyMIcCriticAdapter":
        backbone = load_symic_model(config_source, ckpt_path, device)
        prefix_len = 1 + (NUM_SLOT_TOKENS if bool(backbone.use_instrument_tokens) else 0)
        prompt_soft_len = prefix_len + int(prompt_bars) * CHUNKS_PER_BAR * int(backbone.vq_layers)
        ana = AnalysisSyMIc(
            backbone=backbone,
            prompt_soft_len=prompt_soft_len,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            pool_mode=pool_mode,
            prefix_soft_len=prefix_len,
        )
        ana.eval()
        canonicalizer: Optional[SyMIcPianoRollCanonicalizer] = None
        if canonicalize_pianoroll:
            if vq_model is None:
                raise ValueError("canonicalize_pianoroll=True requires a frozen MidiVision `vq_model`.")
            canonicalizer = SyMIcPianoRollCanonicalizer(
                vq_model=vq_model,
                prompt_soft_len=prompt_soft_len,
                vq_layers=int(backbone.vq_layers),
                vq_groups=int(backbone.vq_groups),
                codebook_size=int(backbone.codebook_size),
                token_offset=int(backbone.token_offset),
                separate_rvq_layers=bool(backbone.separate_rvq_layers),
                use_instrument_tokens=bool(backbone.use_instrument_tokens),
            )
        return cls(ana_model=ana, device=device, canonicalizer=canonicalizer)

    @property
    def critic_module(self) -> nn.Module:
        return self._ana

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _is_compact_critic_key(key: str) -> bool:
        return (".lora_A" in key) or (".lora_B" in key) or key.startswith(("pool_head.", "mil."))

    def mark_trainable(self) -> Dict[str, int]:
        groups = {"lora": 0, "pool_head": 0}
        for name, param in self._ana.named_parameters():
            is_lora = ".lora_A" in name or ".lora_B" in name
            is_head = name.startswith(("pool_head.", "mil."))
            param.requires_grad_(bool(is_lora or is_head))
            if is_lora:
                groups["lora"] += param.numel()
            elif is_head:
                groups["pool_head"] += param.numel()
        if groups["lora"] == 0 or groups["pool_head"] == 0:
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
        for start in range(0, len(critic_sequences), step):
            arr = np.stack(critic_sequences[start : start + step], axis=0)
            batch_t = torch.as_tensor(arr, dtype=torch.long, device=self._device)
            if self._canonicalizer is not None:
                batch_t = self._canonicalizer.canonicalize_sequences(batch_t)
            with ctx:
                parts.append(self._ana.forward_disc(batch_t))
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def export_compact_state(self) -> Dict[str, Any]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self._ana.state_dict().items()
            if self._is_compact_critic_key(k)
        }

    def load_compact_state(self, state: Dict[str, Any]) -> None:
        ana_sd = self._ana.state_dict()
        cleaned: Dict[str, Any] = {}
        for k, v in state.items():
            if k == "_carl_unmerged":
                continue
            if k.startswith("mil.") and f"pool_head.{k}" in ana_sd:
                cleaned[f"pool_head.{k}"] = v
            else:
                cleaned[k] = v

        expected_keys = {k for k in ana_sd.keys() if self._is_compact_critic_key(k)}
        missing_compact = sorted(expected_keys - set(cleaned.keys()))
        if missing_compact:
            raise RuntimeError(
                f"Missing expected SyMIc Critic adapter keys for pool_mode='{self.pool_mode}': "
                f"{missing_compact[:5]}"
            )
        _, unexpected = self._ana.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in SyMIc Critic state_dict: {unexpected[:5]}")


# ======================================================================
# 3. Actor (`SyMIcGroupedActorAdapter`)
# ======================================================================


SYMIC_UNFREEZE_MODES: Tuple[str, ...] = ("full_decoder", "attn_and_depth", "depth_only")


class SyMIcGroupedActorAdapter(CARLActorAdapter):
    """`CARLActorAdapter` for `SyMIc-G4L4 + Depth` (`AblationTransformer2`).

    Even-round parameter policies (`unfreeze_mode`):
      - `'full_decoder'`: Unfreezes all 12 layers' `self_attention`, `ffn`, AND `Wout`
        (`201.24M / 234.96M` on 160M), keeping `embedding` and `RMSNorm` frozen.
      - `'attn_and_depth'`: Unfreezes `self_attention` + `Wout`, keeping `embedding`, `ffn`,
        and `RMSNorm` frozen.
      - `'depth_only'`: Unfreezes `Wout` (`DecodeCodebookHead`) only.
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
        if unfreeze_mode not in SYMIC_UNFREEZE_MODES:
            raise ValueError(
                f"unfreeze_mode must be one of {SYMIC_UNFREEZE_MODES}, got '{unfreeze_mode}'."
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
            is_attn = ".self_attention." in name and self._unfreeze_mode in (
                "full_decoder",
                "attn_and_depth",
            )
            is_ffn = ".ffn." in name and self._unfreeze_mode == "full_decoder"
            trainable = bool(is_wout or is_attn or is_ffn)
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
        lp_chunks: List[np.ndarray] = []
        cont_soft_steps = self._full_soft_len - self._prompt_soft_len

        self._model.eval()
        for start in range(0, len(expanded_examples), step):
            p_batch = torch.as_tensor(
                all_prompts[start : start + step], dtype=torch.long, device=self._device
            )
            out_seq, gen_lp = self._model.generate(
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
                return_logprobs=True,
            )  # out_seq: [chunk_B, full_soft_len, G], gen_lp: [chunk_B, cont_soft_steps, G]
            gen_chunks.append(out_seq[:, : self._full_soft_len, :].detach().cpu().numpy().astype(np.int64))
            cont_lp = gen_lp[:, :cont_soft_steps, :].reshape(p_batch.size(0), -1)
            lp_chunks.append(cont_lp.detach().cpu().numpy().astype(np.float32))

        stacked_out = np.concatenate(gen_chunks, axis=0)
        stacked_lp = np.concatenate(lp_chunks, axis=0)
        gen_tuples = tuple(stacked_out[i] for i in range(stacked_out.shape[0]))
        lp_tuples = tuple(stacked_lp[i] for i in range(stacked_lp.shape[0]))
        return RolloutBatch(
            examples=tuple(expanded_examples),
            generated_sequences=gen_tuples,
            critic_sequences=gen_tuples,
            group_size=group_size,
            old_action_logprobs=lp_tuples,
        )

    def _build_layer_temperature_tensor(
        self,
        cont_soft_steps: int,
        sampling: Optional[UnifiedSamplingConfig],
        device: torch.device,
    ) -> Optional[Tensor]:
        if sampling is None:
            return None
        coarse_t = max(1e-4, float(sampling.temperature))
        fine_t = max(
            1e-4,
            float(sampling.fine_temperature if sampling.fine_temperature is not None else sampling.temperature),
        )
        if abs(coarse_t - 1.0) < 1e-7 and abs(fine_t - 1.0) < 1e-7:
            return None
        l_cnt = int(self._model.vq_layers)
        idx = torch.arange(cont_soft_steps, device=device) % l_cnt
        temps = torch.where(
            idx == 0,
            torch.full_like(idx, coarse_t, dtype=torch.float32),
            torch.full_like(idx, fine_t, dtype=torch.float32),
        )
        return temps.view(1, cont_soft_steps, 1, 1)

    def _forward_continuation_codebook_logprobs(
        self,
        model: AblationTransformer2,
        seq_3d: Tensor,
        sampling: Optional[UnifiedSamplingConfig] = None,
    ) -> Tensor:
        """Compute `[B, T_cont, G, K]` temperature-aligned log-softmax over valid RVQ codebook entries."""
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
        temp_4d = self._build_layer_temperature_tensor(codebook_logits.size(1), sampling, seq_3d.device)
        if temp_4d is not None:
            codebook_logits = codebook_logits / temp_4d
        return F.log_softmax(codebook_logits, dim=-1)

    def forward_logprobs_and_exact_kl(
        self,
        base_actor: CARLActorAdapter,
        examples: Sequence[CARLExample],
        generated_sequences: Sequence[np.ndarray],
        sampling: Optional[UnifiedSamplingConfig] = None,
    ) -> StepLogProbsKL:
        if not isinstance(base_actor.actor_module, AblationTransformer2):
            raise TypeError("SyMIcGroupedActorAdapter requires an AblationTransformer2 base_actor.")

        seq_np = np.stack(generated_sequences, axis=0)
        seq_3d = torch.as_tensor(seq_np, dtype=torch.long, device=self._device)
        bsz = seq_3d.size(0)

        with torch.no_grad():
            logp_base = self._forward_continuation_codebook_logprobs(
                base_actor.actor_module, seq_3d, sampling=sampling
            )
        logp_new = self._forward_continuation_codebook_logprobs(
            self._model, seq_3d, sampling=sampling
        )

        # Extract local codebook targets `[0, K)` and flatten `[B, T_cont, G]` -> `[B, T_cont * G]`
        # (`384 * 4 = 1,536` tokens for 12 bars) so per-token ratio clipping and KL normalization are exact.
        cont_tokens = seq_3d[:, self._prompt_soft_len :, :]  # [B, T_cont, G]
        local_k = (
            (cont_tokens - int(self._model.token_offset)).clamp(min=0) % int(self._model.codebook_size)
        )  # [B, T_cont, G]

        tok_logp_per_group = logp_new.gather(-1, local_k.unsqueeze(-1)).squeeze(-1)  # [B, T_cont, G]
        flat_action_logprobs = tok_logp_per_group.reshape(bsz, -1)  # [B, T_cont * G]

        p_new = logp_new.exp()
        kl_per_group = (p_new * (logp_new - logp_base)).sum(dim=-1)  # [B, T_cont, G]
        flat_exact_kl = kl_per_group.reshape(bsz, -1)  # [B, T_cont * G]

        mask = torch.ones_like(flat_action_logprobs, dtype=torch.bool)
        return StepLogProbsKL(
            action_logprobs=flat_action_logprobs,
            exact_kl=flat_exact_kl,
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
    """Penalizes severe note-density collapse (`< 0.5x`) or note-spam explosion (`> 2.0x`) vs GT.

    Uses an asymmetric safe upper reference floor (`min_ref_onsets`) so even if a ground-truth
    continuation is unusually sparse, normal musical generation is never falsely penalized while
    preserving `penalty == 0.0` on all human ground-truth continuations.
    """

    def __init__(
        self,
        weight: float = 0.25,
        low_ratio: float = 0.50,
        high_ratio: float = 2.00,
        min_ref_onsets: float = 16.0,
        span_octaves: float = 1.5,
    ) -> None:
        self._weight = float(weight)
        self._low_ratio = float(low_ratio)
        self._high_ratio = float(high_ratio)
        self._min_ref = float(min_ref_onsets)
        self._span = max(0.1, float(span_octaves))

    @property
    def name(self) -> str:
        return "density"

    def evaluate(self, context: SyMIcDecodedContext) -> Tuple[Tensor, Dict[str, float]]:
        gen = context.gen_onset_counts
        gt = context.gt_onset_counts
        low_bound = self._low_ratio * gt
        high_bound = self._high_ratio * torch.clamp(gt, min=self._min_ref)

        oct_low = torch.clamp(torch.log2((low_bound + 1.0) / (gen + 1.0)), min=0.0)
        oct_high = torch.clamp(torch.log2((gen + 1.0) / (high_bound + 1.0)), min=0.0)
        oct_violation = torch.where(
            gen < low_bound,
            oct_low,
            torch.where(gen > high_bound, oct_high, torch.zeros_like(gen)),
        )
        violation = torch.clamp(oct_violation / self._span, min=0.0, max=1.0)
        penalties = -self._weight * violation
        ratio = (gen + 1.0) / (gt + 1.0)
        return (
            penalties,
            {
                "onset_count_mean": float(gen.mean().item()),
                "density_ratio_mean": float(ratio.mean().item()),
            },
        )


class SyMIcScaleOutlierRule(GuardrailRule[SyMIcDecodedContext]):
    """Suppresses Fine-layer (`L1..L3`) chromatic scatter exceeding `max(gt_oos, tau_floor) + margin`."""

    def __init__(
        self,
        weight: float = 0.25,
        tau_floor: float = 0.15,
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


SYMIC_GUARDRAIL_MODES: Tuple[str, ...] = ("empty_only", "full")


@dataclass(frozen=True)
class SyMIcEmptyContext:
    """Minimal GPU-reduced context when using `R_disc + empty` only (`guardrail_mode='empty_only'`)."""

    is_empty: Tensor


class SyMIcEmptyOnlyContextBuilder(GuardrailContextBuilder[SyMIcEmptyContext]):
    """Checks non-empty continuation activity directly on GPU without GT decode or CPU roll copies."""

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
    def build_context(self, batch: RolloutBatch) -> SyMIcEmptyContext:
        m_cnt = len(batch)
        dev = next(self._vq.parameters()).device
        stacked = np.stack(
            [seq[self._prompt_soft_len :, :] for seq in batch.generated_sequences], axis=0
        )
        cont_steps = stacked.shape[1]
        if cont_steps % self._vq_layers != 0:
            raise ValueError(
                f"Continuation soft-token count ({cont_steps}) must be divisible by vq_layers ({self._vq_layers})."
            )
        c_cont = cont_steps // self._vq_layers
        flat_chunks = stacked.reshape(m_cnt * c_cont, self._vq_layers, self._vq_groups)

        chan_rows: List[List[bool]] = []
        for ex in batch.examples:
            row = [False] * 8
            for s_i, code in enumerate(ex.slot_codes[:4]):
                if int(code) != SLOT_CODE_EMPTY:
                    row[2 * s_i] = True
                    row[2 * s_i + 1] = True
            chan_rows.append(row)
        flat_chan_mask = (
            torch.tensor(chan_rows, dtype=torch.bool, device=dev)
            .unsqueeze(1)
            .expand(m_cnt, c_cont, 8)
            .reshape(m_cnt * c_cont, 8, 1, 1)
        )

        chunk_active_parts: List[Tensor] = []
        step = self._decode_chunk_size
        for start in range(0, flat_chunks.shape[0], step):
            stop = min(start + step, flat_chunks.shape[0])
            tok_t = torch.as_tensor(flat_chunks[start:stop], dtype=torch.long, device=dev)
            logits = self._vq.decode_tokens(tok_t, token_offset=self._token_offset)
            has_note = ((logits >= 0.0) & flat_chan_mask[start:stop]).any(dim=(1, 2, 3))
            chunk_active_parts.append(has_note)

        seq_has_note = torch.cat(chunk_active_parts, dim=0).view(m_cnt, c_cont).any(dim=1)
        return SyMIcEmptyContext(is_empty=(~seq_has_note).cpu())


def build_symic_guardrail_suite(
    vq_model: MidiVision,
    actor: SyMIcGroupedActorAdapter,
    enable_gt_copy_penalty: bool = True,
    mode: str = "full",
) -> GuardrailSuite[Any]:
    if mode not in SYMIC_GUARDRAIL_MODES:
        raise ValueError(
            f"SyMIc guardrail mode must be one of {SYMIC_GUARDRAIL_MODES}, got '{mode}'."
        )
    raw_model = actor.actor_module
    if not isinstance(raw_model, AblationTransformer2):
        raise TypeError(f"Expected AblationTransformer2, got {type(raw_model).__name__}.")
    if mode == "empty_only":
        empty_builder = SyMIcEmptyOnlyContextBuilder(
            vq_model=vq_model,
            prompt_soft_len=actor.prompt_soft_len,
            vq_layers=int(raw_model.vq_layers),
            vq_groups=int(raw_model.vq_groups),
            token_offset=int(raw_model.token_offset),
        )
        return GuardrailSuite(
            context_builder=empty_builder,
            empty_detector=lambda ctx: ctx.is_empty,
            rules=(),
        )
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
