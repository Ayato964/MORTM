'''
Tokenizerで変換したシーケンスを全て保管します。
'''
import json
import os.path
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from mortm.models.modules.progress import LearningProgress
from einops import rearrange

def get_tag(sequence: np.ndarray, begin_tag_token_id: int, end_tag_token_id: int) -> Tuple[List[int], int]:
    """
    シーケンスから <SYSTEM> から <TAG_END> までのタグ部分を抜き出し、
    タグのリストと、<TAG_END> のインデックスを返す。
    """

    # <SYSTEM> を探す
    ind_begin = np.where(sequence == begin_tag_token_id)[0]
    if len(ind_begin) == 0:
        # <SYSTEM> が見つからなければエラー
        raise IndexError("Begin tag token not found in sequence.")
    ind_begin = ind_begin[0]

    # <SYSTEM> より後ろの <TAG_END> を探す
    ind_end = np.where(sequence == end_tag_token_id)[0]
    ind_end = ind_end[ind_end > ind_begin]
    if len(ind_end) == 0:
        # <TAG_END> が見つからなければエラー
        raise IndexError("End tag token not found after begin tag.")
    ind_end = ind_end[0]

    # 1. タグ部分のリスト と 2. 終了インデックス を両方返す
    return sequence[ind_begin : ind_end + 1].tolist(), ind_end

class MORTM_SEQDataset(Dataset):
    def __init__(self, progress: LearningProgress, positional_length, min_length, is_random_delete_key = False,
                 mask_sample_task: Optional[List[int]]=None, program_token_id: Optional[List[int]]=None, system_tag: Optional[Tuple[int, int]]=None, sampling_inst_max=1):
        self.seq: list = list()
        self.seq_dataset_ids: List[int] = list()
        self.progress = progress
        self.positional_length = positional_length
        self.min_length = min_length
        self.is_random_delete_key = is_random_delete_key
        self.sampling_inst_max = sampling_inst_max
        self.program_token_id = program_token_id if program_token_id is not None else []
        self.mask_sample_task = set(mask_sample_task) if mask_sample_task is not None else []
        self.system_tag = system_tag

    def __len__(self):
        return len(self.seq)

    def add_data(self, music_seq: np.ndarray, dataset_id: int = 0, *args):
        suc_count = 0

        i = 1
        while True:
            key = f"array{i}"
            if key not in music_seq.files:
                break

            arr = music_seq[key]
            seq = np.asarray(arr)

            if seq.ndim == 0:
                i += 1
                continue

            seq = seq.astype(np.int64, copy=False)

            if self.min_length < len(seq) < self.positional_length:
                self.seq.append(np.ascontiguousarray(seq.copy()))
                self.seq_dataset_ids.append(int(dataset_id))
                suc_count += 1

            i += 1

        return suc_count

    def __getitem__(self, item):
        sequence_array = self.seq[item]

        has_mask_task = any(
            int(token) in self.mask_sample_task for token in sequence_array) if self.mask_sample_task else False

        apply_augmentation = (
                self.is_random_delete_key
                and self.system_tag is not None
                and self.program_token_id
                and self.sampling_inst_max > 0
                and has_mask_task
        )

        if not apply_augmentation:
            return torch.from_numpy(sequence_array.astype(np.int64, copy=False))

        try:
            begin_tag_id, end_tag_id = self.system_tag
            footer_tokens = {626, 627}

            ind_begin_tag_search = np.where(sequence_array == begin_tag_id)[0]
            if len(ind_begin_tag_search) == 0:
                raise IndexError("Begin tag token not found in sequence.")
            ind_begin_tag = ind_begin_tag_search[0]

            prefix_part = sequence_array[:ind_begin_tag].tolist()

            system_seq_list, ind_end_tag = get_tag(sequence_array, begin_tag_id, end_tag_id)

            present_inst_tokens = [
                token for token in system_seq_list if token in self.program_token_id
            ]
            inst_count = len(present_inst_tokens)

            if inst_count <= 1:
                return torch.from_numpy(sequence_array.astype(np.int64, copy=False))

            max_k = min(self.sampling_inst_max, inst_count)
            k = random.randint(1, max_k)
            kept_inst_tokens = set(random.sample(present_inst_tokens, k))

            if len(kept_inst_tokens) == inst_count:
                return torch.from_numpy(sequence_array.astype(np.int64, copy=False))

            new_system_seq = [
                token for token in system_seq_list
                if (token not in self.program_token_id) or (token in kept_inst_tokens)
            ]

            music_events_part = sequence_array[ind_end_tag + 1:]
            inst_token_indices = []

            for i, token in enumerate(music_events_part):
                if token in self.program_token_id:
                    inst_token_indices.append((i, token))

            new_music_events = []

            if not inst_token_indices:
                new_music_events.extend(music_events_part.tolist())
            else:
                first_inst_idx = inst_token_indices[0][0]
                new_music_events.extend(music_events_part[:first_inst_idx].tolist())

                for i, (chunk_start_idx, inst_token_id) in enumerate(inst_token_indices):
                    if inst_token_id not in kept_inst_tokens:
                        continue

                    if i + 1 < len(inst_token_indices):
                        next_chunk_start_idx = inst_token_indices[i + 1][0]
                        chunk_end_idx = next_chunk_start_idx
                    else:
                        footer_start_idx = -1
                        for j, tok in enumerate(music_events_part[chunk_start_idx:]):
                            if tok in footer_tokens:
                                footer_start_idx = chunk_start_idx + j
                                break

                        if footer_start_idx != -1:
                            chunk_end_idx = footer_start_idx
                        else:
                            chunk_end_idx = len(music_events_part)

                    chunk = music_events_part[chunk_start_idx:chunk_end_idx]
                    new_music_events.extend(chunk.tolist())

                last_inst_start_idx = inst_token_indices[-1][0]
                footer_start_idx = -1
                for j, tok in enumerate(music_events_part[last_inst_start_idx:]):
                    if tok in footer_tokens:
                        footer_start_idx = last_inst_start_idx + j
                        break

                if footer_start_idx != -1:
                    new_music_events.extend(music_events_part[footer_start_idx:].tolist())

            final_sequence_list = prefix_part + new_system_seq + new_music_events
            return torch.tensor(final_sequence_list, dtype=torch.long)

        except Exception as e:
            print(f"Warning: Augmentation failed for item {item}, returning original. Error: {e} {sequence_array}")
            return torch.from_numpy(sequence_array.astype(np.int64, copy=False))


class ClassDataSets(Dataset):
    def __init__(self, progress: LearningProgress, positional_length):
        self.key: list = list()
        self.value: list = list()
        self.progress = progress
        self.positional_length = positional_length

    def __len__(self):
        return len(self.key)

    def __getitem__(self, item):
        #print(self.value[item], max(self.key[item]))
        if self.value[item] == 0:
            ind = [i for i, v in enumerate(self.key[item]) if v == 8]
            r = 4 + random.randint(0, 8)
            if r != 12 and r < len(ind):
                v = self.key[item][:ind[r]]
            else:
                v = self.key[item]
        else:
            v = self.key[item]
        return (torch.tensor(v, dtype=torch.long),
                torch.tensor(self.value[item], dtype=torch.long))

    def add_data(self, music_seq: np.ndarray, value):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            seq = music_seq[f'array{i + 1}'].tolist()
            if 90 < len(seq) < self.positional_length and seq.count(4) < 3:
                self.key.append(seq)
                self.value.append(value)
                suc_count += 1

        return suc_count


class PianoRollDataset(Dataset):
    def __init__(self, progress: LearningProgress):
        self.progress = progress
        self.src_list: List[np.ndarray] = list()

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, item: int) :
        return torch.tensor(self.src_list[item])

    def add_data(self, piano_roll: np.ndarray, *args):
        self.src_list.append(piano_roll)

    def set_tokenizer_dataset(self, chunk_size=16):
        new_src_list = []
        for piano_roll in self.src_list:
            if len(piano_roll) > 0:
                num_chunks = len(piano_roll) // chunk_size
                for i in range(num_chunks):
                    chunk = piano_roll[i * chunk_size : (i + 1) * chunk_size]
                    new_src_list.append(chunk)
        self.src_list = new_src_list


class PreLoadingDatasets(Dataset):
    def __init__(self, progress: LearningProgress):
        self.progress = progress
        self.src_list: List[str] = list()
        self.dataset_ids: List[int] = list()
        self.dataset_names: List[str] = list()

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, item: int) :
        return self.src_list[item], self.dataset_ids[item]


    def add_data(self, directory: List[str], filename: List[str], dataset_id: int = 0, dataset_name: Optional[str] = None):
        for i in range(len(directory)):
            self.src_list.append(os.path.join(directory[i], filename[i]))
            self.dataset_ids.append(dataset_id)
            self.dataset_names.append(dataset_name or f"dataset_{dataset_id}")

    def add_paths(self, paths: List[str], dataset_id: int = 0, dataset_name: Optional[str] = None):
        for path in paths:
            self.src_list.append(path)
            self.dataset_ids.append(dataset_id)
            self.dataset_names.append(dataset_name or f"dataset_{dataset_id}")

    def add_data_json(self, json_data: str, dataset_id: int = 0, dataset_name: Optional[str] = None):
        with open(json_data, 'r') as f:
            data = json.load(f)
            for item in data:
                self.src_list.append(item)
                self.dataset_ids.append(dataset_id)
                self.dataset_names.append(dataset_name or f"dataset_{dataset_id}")



class TensorDataset(Dataset):
    def __init__(self, progress: LearningProgress):
        self.seq: list = list()
        self.progress = progress

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item].to(self.progress.get_device())

    def add_data(self, patch_list: list, *args):
        self.seq = patch_list


from mortm.models.modules.config import (
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
    INST_RVQ_TOKEN_OFFSET,
    NUM_SLOT_TOKENS,
    MORTM5Args,
    slot_code_to_token_id,
)


class PPODataset(Dataset):
    def __init__(self,
                 sequences: Tensor,
                 log_probs: Tensor,
                 values: Tensor,
                 advantages: Tensor,
                 returns: Tensor):

        # 各テンソルをそのままインスタンス変数として保持する
        self.sequences = sequences
        self.log_probs = log_probs
        self.values = values
        self.advantages = advantages
        self.returns = returns

    def __len__(self):
        # バッチサイズを返す
        return self.sequences.size(0)

    def __getitem__(self, index: int):
        # 指定されたインデックスのデータを、各テンソルから取り出してタプルで返す
        return (
            self.sequences[index],
            self.log_probs[index],
            self.values[index],
            self.advantages[index],
            self.returns[index]
        )


class MORTM5MemmapDataset(Dataset):
    """
    MidiVision (M3 / M4) でトークン化されたフラットバイナリ (.bin + _offsets.npy [+ _slots.npy]) を
    高速かつ省メモリに読み込む Dataset クラス。

    8通りのアブレーション設定および4スロット楽器条件付けに完全対応:
    - mode="flat":
        AblationTransformer1 用。1トークン単位の自己回帰。
        先頭に [BOS] (+ 楽器トークン4個 [Slot0..Slot3])、末尾に [EOS] を付与する。
    - mode="grouped":
        AblationTransformer2 用。vq_groups (G) トークンを1ソフトトークンとして束ねる自己回帰。
        シーケンス長が常に G の倍数になるよう、先頭に [BOS]*G (+ [Slot0]*G .. [Slot3]*G)、末尾に [EOS]*G を付与する。
    - separate_rvq_layers:
        False の場合は全 RVQ レイヤーで語彙 (G * K) を共有。
        True の場合は .bin を再トークン化することなく、オンザフライでレイヤー l ごとに l * G * K を加算し、
        全 RVQ レイヤー独立語彙 (L * G * K) に変換する。
    - use_instrument_tokens:
        True の場合、各ウィンドウの先頭 (<BOS> の直後) に 4 スロットの楽器条件付けトークンを付与し、
        RVQ トークンのオフセットを INST_RVQ_TOKEN_OFFSET (133) にシフトする。
    """

    def __init__(
        self,
        tokens_bin_path: str,
        offsets_npy_path: str,
        vq_groups: int = 8,
        vq_layers: int = 2,
        codebook_size: int = 2048,
        max_chunks: int = 128,
        min_chunks: int = 8,
        stride_chunks: Optional[int] = None,
        mode: str = "flat",
        bos_eos_mode: str = "always",
        separate_rvq_layers: bool = False,
        use_instrument_tokens: bool = False,
        slots_npy_path: Optional[str] = None,
        token_offset: Optional[int] = None,
        raw_bin_offset: int = RVQ_TOKEN_OFFSET,
        pad_token_id: int = PAD_TOKEN_ID,
        bos_token_id: int = BOS_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
    ):
        super().__init__()
        if mode not in ("flat", "grouped"):
            raise ValueError(f"mode must be 'flat' or 'grouped', got {mode}")
        if bos_eos_mode not in ("always", "song_boundary", "none"):
            raise ValueError(f"bos_eos_mode must be 'always', 'song_boundary', or 'none', got {bos_eos_mode}")

        self.tokens_bin_path = os.path.abspath(tokens_bin_path)
        self.offsets_npy_path = os.path.abspath(offsets_npy_path)
        self.vq_groups = int(vq_groups)
        self.vq_layers = int(vq_layers)
        self.codebook_size = int(codebook_size)
        if self.vq_groups <= 0 or self.vq_layers <= 0 or self.codebook_size <= 0:
            raise ValueError(
                f"vq_groups ({self.vq_groups}), vq_layers ({self.vq_layers}), and "
                f"codebook_size ({self.codebook_size}) must be positive."
            )
        self.tokens_per_chunk = self.vq_groups * self.vq_layers

        self.max_chunks = int(max_chunks)
        self.min_chunks = int(min_chunks)
        self.stride_chunks = int(stride_chunks) if stride_chunks is not None else int(max_chunks)
        if self.min_chunks <= 0 or self.max_chunks < self.min_chunks or self.stride_chunks <= 0:
            raise ValueError(
                f"Invalid chunk window settings: min_chunks={self.min_chunks}, "
                f"max_chunks={self.max_chunks}, stride_chunks={self.stride_chunks}."
            )
        self.mode = mode
        self.bos_eos_mode = bos_eos_mode
        self.separate_rvq_layers = bool(separate_rvq_layers)
        self.use_instrument_tokens = bool(use_instrument_tokens)
        self.raw_bin_offset = int(raw_bin_offset)

        if token_offset is None:
            self.token_offset = INST_RVQ_TOKEN_OFFSET if self.use_instrument_tokens else RVQ_TOKEN_OFFSET
        else:
            self.token_offset = int(token_offset)

        if self.use_instrument_tokens and self.token_offset < INST_RVQ_TOKEN_OFFSET:
            raise ValueError(
                f"use_instrument_tokens=True requires token_offset >= {INST_RVQ_TOKEN_OFFSET}, "
                f"got {self.token_offset}."
            )

        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)

        if self.tokens_bin_path.endswith("_tokens.bin"):
            complete_alt = self.tokens_bin_path[: -len("_tokens.bin")] + "_tokens_complete.bin"
            if os.path.exists(complete_alt):
                if not os.path.exists(self.tokens_bin_path) or os.path.getsize(complete_alt) > os.path.getsize(self.tokens_bin_path):
                    self.tokens_bin_path = complete_alt

        if not os.path.exists(self.tokens_bin_path):
            raise FileNotFoundError(f"Tokens binary file not found: {self.tokens_bin_path}")
        if not os.path.exists(self.offsets_npy_path):
            raise FileNotFoundError(f"Offsets npy file not found: {self.offsets_npy_path}")

        # もし隣接する {split}_state.json が存在すれば、vq_groups / vq_layers / codebook_size の設定不一致を検出する
        for suffix in ("_tokens.bin", "_tokens_complete.bin"):
            if self.tokens_bin_path.endswith(suffix):
                candidate_state_path = self.tokens_bin_path[: -len(suffix)] + "_state.json"
                if os.path.exists(candidate_state_path):
                    with open(candidate_state_path, "r", encoding="utf-8") as sf:
                        state_meta = json.load(sf)
                    meta_g = state_meta.get("vq_groups")
                    meta_l = state_meta.get("vq_layers")
                    meta_k = state_meta.get("codebook_size")
                    if (
                        (meta_g is not None and int(meta_g) != self.vq_groups)
                        or (meta_l is not None and int(meta_l) != self.vq_layers)
                        or (meta_k is not None and int(meta_k) != self.codebook_size)
                    ):
                        raise ValueError(
                            f"Tokenizer metadata mismatch for {self.tokens_bin_path}: "
                            f"state.json has (vq_groups={meta_g}, vq_layers={meta_l}, codebook_size={meta_k}), "
                            f"but MORTM5MemmapDataset was initialized with "
                            f"(vq_groups={self.vq_groups}, vq_layers={self.vq_layers}, codebook_size={self.codebook_size})."
                        )
                    if "token_offset" in state_meta and raw_bin_offset == RVQ_TOKEN_OFFSET:
                        self.raw_bin_offset = int(state_meta["token_offset"])

        # shape: [N_songs, 2] -> (start_token_offset, num_tokens)
        self.offsets = np.load(self.offsets_npy_path)
        if self.offsets.ndim != 2 or self.offsets.shape[1] != 2:
            raise ValueError(f"Expected offsets array of shape [N, 2], got {self.offsets.shape}")

        bin_size_bytes = os.path.getsize(self.tokens_bin_path)
        if bin_size_bytes % 2 != 0:
            raise ValueError(
                f"Tokens binary file size ({bin_size_bytes} bytes) is not aligned to uint16 (2 bytes): {self.tokens_bin_path}"
            )
        total_bin_tokens = bin_size_bytes // 2

        if len(self.offsets) > 0:
            if np.any(self.offsets[:, 0] < 0) or np.any(self.offsets[:, 1] <= 0):
                raise ValueError(f"Offsets contain negative start or non-positive length in {self.offsets_npy_path}")
            if np.any(self.offsets[:, 1] % self.tokens_per_chunk != 0):
                raise ValueError(
                    f"Offsets contain song token lengths not divisible by tokens_per_chunk ({self.tokens_per_chunk})"
                )
            required_tokens = int(np.max(self.offsets[:, 0] + self.offsets[:, 1]))
            if total_bin_tokens < required_tokens:
                raise ValueError(
                    f"Tokens binary file {self.tokens_bin_path} is truncated: has {total_bin_tokens} tokens, "
                    f"but offsets require at least {required_tokens} tokens."
                )

        # スロット楽器コード [N_songs, 4] のロードおよび楽器トークンIDへの事前変換
        self.slots_npy_path: Optional[str] = None
        self.slots: Optional[np.ndarray] = None
        self.slot_token_ids: Optional[np.ndarray] = None

        if slots_npy_path is not None:
            self.slots_npy_path = os.path.abspath(slots_npy_path)
            if not os.path.exists(self.slots_npy_path):
                raise FileNotFoundError(f"Slots npy file not found: {self.slots_npy_path}")
        elif self.offsets_npy_path.endswith("_offsets.npy"):
            candidate_slots = self.offsets_npy_path[: -len("_offsets.npy")] + "_slots.npy"
            if os.path.exists(candidate_slots):
                self.slots_npy_path = candidate_slots

        if self.slots_npy_path is not None:
            loaded_slots = np.load(self.slots_npy_path)
            if loaded_slots.ndim != 2 or loaded_slots.shape[1] != NUM_SLOT_TOKENS:
                raise ValueError(
                    f"Expected slots array of shape [N, {NUM_SLOT_TOKENS}], got {loaded_slots.shape} in {self.slots_npy_path}"
                )
            if len(loaded_slots) < len(self.offsets):
                raise ValueError(
                    f"Slots array length ({len(loaded_slots)}) < offsets length ({len(self.offsets)}) in {self.slots_npy_path}"
                )
            self.slots = np.asarray(loaded_slots[: len(self.offsets)], dtype=np.int16)
        elif self.use_instrument_tokens:
            # フォールバック: 隣接する {split}_manifest.jsonl に slots フィールドが含まれていれば復元する
            manifest_slots = self._try_load_slots_from_manifest()
            if manifest_slots is not None:
                self.slots = manifest_slots
            else:
                raise FileNotFoundError(
                    f"use_instrument_tokens=True requires a '{os.path.basename(self.offsets_npy_path).replace('_offsets.npy', '_slots.npy')}' "
                    f"file (or 'slots' field in manifest.jsonl) alongside {self.offsets_npy_path}. "
                    f"Run ConvertMidiVisionTokens.py with --build_slots_only to generate it."
                )

        if self.slots is not None:
            slots_i64 = self.slots.astype(np.int64, copy=False)
            valid_slots = (slots_i64 == -1) | ((slots_i64 >= 0) & (slots_i64 <= 128))
            if not bool(np.all(valid_slots)):
                bad_val = int(slots_i64[~valid_slots][0])
                slot_code_to_token_id(bad_val)  # 既存の ValueError メッセージを送出
            self.slot_token_ids = np.where(
                slots_i64 == 128,
                slot_code_to_token_id(128),
                np.where(slots_i64 == -1, slot_code_to_token_id(-1), 3 + slots_i64),
            ).astype(np.int64, copy=False)

        self._unit_width = 1 if self.mode == "flat" else self.vq_groups
        self._bos_arr = np.full(self._unit_width, self.bos_token_id, dtype=np.int64)
        self._eos_arr = np.full(self._unit_width, self.eos_token_id, dtype=np.int64)

        # 1チャンク (L * G トークン) 内の各位置に加算するシフト量を事前計算 (O(1) ベクトル加算用)
        offset_shift = self.token_offset - self.raw_bin_offset
        layer_ids = np.arange(self.vq_layers, dtype=np.int64)[:, None]  # [L, 1]
        layer_shift = (
            layer_ids * (self.vq_groups * self.codebook_size)
            if self.separate_rvq_layers
            else np.zeros_like(layer_ids)
        )
        self._intra_chunk_shift = (
            np.broadcast_to(offset_shift + layer_shift, (self.vq_layers, self.vq_groups))
            .reshape(1, self.tokens_per_chunk)
            .copy()
        )
        self._needs_token_shift = bool(np.any(self._intra_chunk_shift != 0))

        self._memmap: Optional[np.memmap] = None
        self.windows: np.ndarray = self._build_windows()

    @classmethod
    def from_args(
        cls,
        args: MORTM5Args,
        tokens_bin_path: str,
        offsets_npy_path: str,
        slots_npy_path: Optional[str] = None,
        max_chunks: int = 128,
        min_chunks: int = 8,
        stride_chunks: Optional[int] = None,
        bos_eos_mode: str = "always",
    ) -> "MORTM5MemmapDataset":
        """MORTM5Args の設定から整合性を保証して MORTM5MemmapDataset を生成するファクトリメソッド。"""
        return cls(
            tokens_bin_path=tokens_bin_path,
            offsets_npy_path=offsets_npy_path,
            vq_groups=args.vq_groups,
            vq_layers=args.vq_layers,
            codebook_size=args.codebook_size,
            max_chunks=max_chunks,
            min_chunks=min_chunks,
            stride_chunks=stride_chunks,
            mode=getattr(args, "model_mode", "flat"),
            bos_eos_mode=bos_eos_mode,
            separate_rvq_layers=getattr(args, "separate_rvq_layers", False),
            use_instrument_tokens=getattr(args, "use_instrument_tokens", False),
            slots_npy_path=slots_npy_path,
            token_offset=args.token_offset,
            pad_token_id=args.pad_token_id,
            bos_token_id=args.bos_token_id,
            eos_token_id=args.eos_token_id,
        )

    @classmethod
    def from_split_dir(
        cls,
        token_dir: str,
        split: str,
        args: MORTM5Args,
        max_chunks: Optional[int] = None,
        min_chunks: int = 8,
        stride_chunks: Optional[int] = None,
        stride: Optional[int] = None,
        bos_eos_mode: str = "always",
    ) -> "MORTM5MemmapDataset":
        """
        ディレクトリ `token_dir` と `split` ("train" or "val") および `MORTM5Args` から
        `max_chunks` を自動算出して `MORTM5MemmapDataset` を生成する。
        """
        if stride_chunks is None and stride is not None:
            stride_chunks = stride
        tokens_bin_path = os.path.join(token_dir, f"{split}_tokens.bin")
        offsets_npy_path = os.path.join(token_dir, f"{split}_offsets.npy")
        if not os.path.exists(offsets_npy_path):
            alt_index = os.path.join(token_dir, f"{split}_index.npy")
            if os.path.exists(alt_index):
                offsets_npy_path = alt_index
        slots_candidate = os.path.join(token_dir, f"{split}_slots.npy")
        slots_npy_path = slots_candidate if os.path.exists(slots_candidate) else None

        if max_chunks is None:
            mode = getattr(args, "model_mode", "flat")
            reserved_units = 2 + (NUM_SLOT_TOKENS if getattr(args, "use_instrument_tokens", False) else 0)
            if mode == "grouped":
                usable_soft = max(args.vq_layers, int(args.position_length) - reserved_units)
                max_chunks = max(1, usable_soft // int(args.vq_layers))
            else:
                tok_per_chunk = int(args.vq_groups) * int(args.vq_layers)
                usable_flat = max(tok_per_chunk, int(args.position_length) - reserved_units)
                max_chunks = max(1, usable_flat // tok_per_chunk)

        eff_min_chunks = min(int(min_chunks), int(max_chunks))
        ds = cls.from_args(
            args=args,
            tokens_bin_path=tokens_bin_path,
            offsets_npy_path=offsets_npy_path,
            slots_npy_path=slots_npy_path,
            max_chunks=max_chunks,
            min_chunks=eff_min_chunks,
            stride_chunks=stride_chunks,
            bos_eos_mode=bos_eos_mode,
        )
        ds.split = split
        return ds

    def _try_load_slots_from_manifest(self) -> Optional[np.ndarray]:
        if not self.offsets_npy_path.endswith("_offsets.npy"):
            return None
        manifest_path = self.offsets_npy_path[: -len("_offsets.npy")] + "_manifest.jsonl"
        if not os.path.exists(manifest_path):
            return None
        slots_rows: List[List[int]] = []
        with open(manifest_path, "r", encoding="utf-8") as mf:
            for _ in range(len(self.offsets)):
                line = mf.readline()
                if not line:
                    return None
                rec = json.loads(line)
                if "slots" not in rec or len(rec["slots"]) != NUM_SLOT_TOKENS:
                    return None
                slots_rows.append([int(x) for x in rec["slots"]])
        if len(slots_rows) != len(self.offsets):
            return None
        return np.asarray(slots_rows, dtype=np.int16)

    def _build_windows(self) -> np.ndarray:
        """
        各楽曲の (start_offset, num_tokens) から学習ウィンドウ
        (token_start, token_len, is_song_start, is_song_end, song_idx) のテーブルを構築する。
        """
        max_win_tokens = self.max_chunks * self.tokens_per_chunk
        min_win_tokens = self.min_chunks * self.tokens_per_chunk
        stride_tokens = self.stride_chunks * self.tokens_per_chunk

        windows = []
        for song_idx in range(len(self.offsets)):
            start_offset = int(self.offsets[song_idx, 0])
            num_tokens = int(self.offsets[song_idx, 1])

            if num_tokens < min_win_tokens:
                continue

            if num_tokens <= max_win_tokens:
                windows.append((start_offset, num_tokens, 1, 1, song_idx))
                continue

            pos = 0
            while pos < num_tokens:
                rem = num_tokens - pos
                if rem < min_win_tokens:
                    break
                win_len = min(max_win_tokens, rem)
                is_first = 1 if pos == 0 else 0
                is_last = 1 if (pos + win_len >= num_tokens) else 0
                windows.append((start_offset + pos, win_len, is_first, is_last, song_idx))
                if pos + win_len >= num_tokens:
                    break
                pos += stride_tokens

        if len(windows) == 0:
            return np.zeros((0, 5), dtype=np.int64)
        return np.asarray(windows, dtype=np.int64)

    def _get_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(self.tokens_bin_path, dtype=np.uint16, mode="r")
        return self._memmap

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_memmap"] = None
        return state

    def __len__(self) -> int:
        return len(self.windows)

    def get_window_slot_codes(self, index: int) -> Optional[np.ndarray]:
        """指定ウィンドウの楽曲に対応する 4 スロット楽器コード [4] (int16) を返す。"""
        if self.slots is None:
            return None
        song_idx = int(self.windows[index, 4])
        return self.slots[song_idx].copy()

    def __getitem__(self, index: int) -> Tensor:
        mmap = self._get_memmap()
        token_start, token_len, is_first, is_last, song_idx = self.windows[index]
        raw_tokens = np.asarray(mmap[token_start : token_start + token_len], dtype=np.int64)
        if raw_tokens.size != token_len:
            raise RuntimeError(
                f"Memmap slice length mismatch at [{token_start}:{token_start + token_len}]: "
                f"expected {token_len}, got {raw_tokens.size}"
            )

        if self._needs_token_shift:
            raw_tokens = (
                raw_tokens.reshape(-1, self.tokens_per_chunk) + self._intra_chunk_shift
            ).reshape(-1)

        add_bos = (
            self.bos_eos_mode == "always"
            or (self.bos_eos_mode == "song_boundary" and is_first == 1)
        )
        add_eos = (
            self.bos_eos_mode == "always"
            or (self.bos_eos_mode == "song_boundary" and is_last == 1)
        )

        unit_width = self._unit_width
        bos_len = unit_width if add_bos else 0
        inst_len = (NUM_SLOT_TOKENS * unit_width) if self.use_instrument_tokens else 0
        eos_len = unit_width if add_eos else 0

        if bos_len == 0 and inst_len == 0 and eos_len == 0:
            return torch.from_numpy(np.ascontiguousarray(raw_tokens, dtype=np.int64))

        seq = np.empty(bos_len + inst_len + token_len + eos_len, dtype=np.int64)
        cursor = 0
        if bos_len > 0:
            seq[cursor : cursor + bos_len] = self._bos_arr
            cursor += bos_len
        if inst_len > 0:
            if self.slot_token_ids is None:
                raise RuntimeError("use_instrument_tokens=True but slot_token_ids is not initialized.")
            slot_toks = self.slot_token_ids[song_idx]
            if unit_width == 1:
                seq[cursor : cursor + inst_len] = slot_toks
            else:
                seq[cursor : cursor + inst_len] = np.repeat(slot_toks, unit_width)
            cursor += inst_len
        seq[cursor : cursor + token_len] = raw_tokens
        cursor += token_len
        if eos_len > 0:
            seq[cursor : cursor + eos_len] = self._eos_arr
        return torch.from_numpy(seq)



def mortm5_collate_fn(
    batch: List[Tensor],
    mode: str = "flat",
    vq_groups: int = 8,
    pad_token_id: int = PAD_TOKEN_ID,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    MORTM5MemmapDataset のミニバッチから (input_ids, targets, padding_mask) を作成する。

    - mode="flat" (AblationTransformer1):
        shift = 1 トークン。
        input_ids: [B, S - 1]
        targets:   [B, S - 1]
        padding_mask: [B, S - 1] (True = 有効トークン, False = PAD)
    - mode="grouped" (AblationTransformer2):
        shift = vq_groups (G) トークン (= 1 ソフトトークン)。
        input_ids: [B, S - G]
        targets:   [B, S - G]
        padding_mask: [B, (S - G) // G] (True = 有効ソフトトークン, False = PAD)
    """
    from torch.nn.utils.rnn import pad_sequence

    if mode not in ("flat", "grouped"):
        raise ValueError(f"mode must be 'flat' or 'grouped', got {mode}")
    if not batch:
        raise ValueError("mortm5_collate_fn received an empty batch.")

    vq_groups = int(vq_groups)
    if vq_groups <= 0:
        raise ValueError(f"vq_groups must be positive, got {vq_groups}")

    shift = 1 if mode == "flat" else vq_groups
    for idx, seq in enumerate(batch):
        if seq.dim() != 1:
            raise ValueError(f"Batch item {idx} must be a 1D tensor, got shape {tuple(seq.shape)}")
        if seq.numel() <= shift:
            raise ValueError(
                f"Batch item {idx} length ({seq.numel()}) must be > shift ({shift}) for causal training."
            )
        if mode == "grouped" and seq.numel() % vq_groups != 0:
            raise ValueError(
                f"Batch item {idx} length ({seq.numel()}) is not divisible by vq_groups ({vq_groups}) in grouped mode."
            )

    padded = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)

    if mode == "grouped" and padded.size(1) % vq_groups != 0:
        rem = vq_groups - (padded.size(1) % vq_groups)
        pad_tail = torch.full(
            (padded.size(0), rem), pad_token_id, dtype=padded.dtype, device=padded.device
        )
        padded = torch.cat([padded, pad_tail], dim=1)

    input_ids = padded[:, :-shift].contiguous()
    targets = padded[:, shift:].contiguous()

    if mode == "flat":
        padding_mask = input_ids != pad_token_id
    else:
        # ソフトトークン単位のマスク: [B, S_soft]
        grouped_inputs = input_ids.reshape(input_ids.size(0), -1, vq_groups)
        padding_mask = (grouped_inputs != pad_token_id).any(dim=-1)

    return input_ids, targets, padding_mask


class MORTM5Collate:
    """
    DataLoader 用のシリアライズ可能な Collate ファンクタ。
    mortm5_collate_fn に mode / vq_groups / pad_token_id を束縛する。
    """

    def __init__(self, mode: str = "flat", vq_groups: int = 8, pad_token_id: int = PAD_TOKEN_ID):
        self.mode = mode
        self.vq_groups = int(vq_groups)
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        return mortm5_collate_fn(
            batch,
            mode=self.mode,
            vq_groups=self.vq_groups,
            pad_token_id=self.pad_token_id,
        )


class MORTM5OuterDataset(Dataset):
    """
    self_turing の 2段 DataLoader (Outer Loader -> _prefetched_outer_batches -> pre_processing -> Inner Loader)
    に MORTM5MemmapDataset を適合させる軽量インデックス Dataset。

    - Outer Loader は `big_batch_size` 個の `(split, window_idx)` タプルを 1 パックとして yield する。
    - `src_list` プロパティを提供し、`save_path_json` との完全な互換性を保つ。
    """

    def __init__(self, memmap_dataset: MORTM5MemmapDataset, split: Optional[str] = None):
        self.memmap_dataset = memmap_dataset
        self.split = split or getattr(memmap_dataset, "split", "train")
        paths = [str(memmap_dataset.tokens_bin_path), str(memmap_dataset.offsets_npy_path)]
        if getattr(memmap_dataset, "slots_npy_path", None) is not None:
            paths.append(str(memmap_dataset.slots_npy_path))
        self.src_list = paths

    def __len__(self) -> int:
        return len(self.memmap_dataset)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.split, int(index)


class MORTM5WindowSubsetDataset(Dataset):
    """
    MORTM5TrainSet.pre_processing がバックグラウンドスレッドで先読みした
    1 Outer Batch 分のウィンドウ Tensor 群、または (MORTM5MemmapDataset, indices) サブセットを保持する Dataset。
    """

    def __init__(self, windows: List[Tensor] | Dataset, indices: Optional[List[int]] = None):
        if indices is not None:
            self.windows = [windows[int(i)] for i in indices]
        else:
            self.windows = list(windows)  # type: ignore

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tensor:
        return self.windows[index]


def mortm5_outer_collate_fn(batch: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Outer DataLoader 用 Collate 関数: (split, window_idx) のリストをそのまま返す。"""
    return list(batch)

