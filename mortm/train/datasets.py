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


from mortm.models.modules.config import PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, RVQ_TOKEN_OFFSET


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
    MidiVision (M3 / M4) でトークン化されたフラットバイナリ (.bin + _offsets.npy) を
    高速かつ省メモリに読み込む Dataset クラス。

    - mode="flat":
        AblationTransformer1 用。1トークン単位の自己回帰。
        シーケンス先頭に [BOS] (1個)、末尾に [EOS] (1個) を付与する。
    - mode="grouped":
        AblationTransformer2 用。vq_groups (G) トークンを1ソフトトークンとして束ねる自己回帰。
        シーケンス長が常に G の倍数になるよう、先頭に [BOS] * G、末尾に [EOS] * G を付与する。
    """

    def __init__(
        self,
        tokens_bin_path: str,
        offsets_npy_path: str,
        vq_groups: int = 8,
        vq_layers: int = 2,
        max_chunks: int = 128,
        min_chunks: int = 8,
        stride_chunks: Optional[int] = None,
        mode: str = "flat",
        bos_eos_mode: str = "always",
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
        if self.vq_groups <= 0 or self.vq_layers <= 0:
            raise ValueError(f"vq_groups ({self.vq_groups}) and vq_layers ({self.vq_layers}) must be positive.")
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

        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)

        if not os.path.exists(self.tokens_bin_path):
            raise FileNotFoundError(f"Tokens binary file not found: {self.tokens_bin_path}")
        if not os.path.exists(self.offsets_npy_path):
            raise FileNotFoundError(f"Offsets npy file not found: {self.offsets_npy_path}")

        # もし隣接する {split}_state.json が存在すれば、vq_groups / vq_layers の設定不一致を検出する
        if self.tokens_bin_path.endswith("_tokens.bin"):
            candidate_state_path = self.tokens_bin_path[: -len("_tokens.bin")] + "_state.json"
            if os.path.exists(candidate_state_path):
                with open(candidate_state_path, "r", encoding="utf-8") as sf:
                    state_meta = json.load(sf)
                meta_g = state_meta.get("vq_groups")
                meta_l = state_meta.get("vq_layers")
                if (meta_g is not None and int(meta_g) != self.vq_groups) or (
                    meta_l is not None and int(meta_l) != self.vq_layers
                ):
                    raise ValueError(
                        f"Tokenizer metadata mismatch for {self.tokens_bin_path}: "
                        f"state.json has (vq_groups={meta_g}, vq_layers={meta_l}), "
                        f"but MORTM5MemmapDataset was initialized with (vq_groups={self.vq_groups}, vq_layers={self.vq_layers})."
                    )

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

        self._memmap: Optional[np.memmap] = None
        self.windows: np.ndarray = self._build_windows()

    def _build_windows(self) -> np.ndarray:
        """
        各楽曲の (start_offset, num_tokens) から学習ウィンドウ
        (token_start, token_len, is_song_start, is_song_end) のテーブルを構築する。
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
                windows.append((start_offset, num_tokens, 1, 1))
                continue

            pos = 0
            while pos < num_tokens:
                rem = num_tokens - pos
                if rem < min_win_tokens:
                    break
                win_len = min(max_win_tokens, rem)
                is_first = 1 if pos == 0 else 0
                is_last = 1 if (pos + win_len >= num_tokens) else 0
                windows.append((start_offset + pos, win_len, is_first, is_last))
                if pos + win_len >= num_tokens:
                    break
                pos += stride_tokens

        if len(windows) == 0:
            return np.zeros((0, 4), dtype=np.int64)
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

    def __getitem__(self, index: int) -> Tensor:
        mmap = self._get_memmap()
        token_start, token_len, is_first, is_last = self.windows[index]
        raw_tokens = np.asarray(mmap[token_start : token_start + token_len], dtype=np.int64)
        if raw_tokens.size != token_len:
            raise RuntimeError(
                f"Memmap slice length mismatch at [{token_start}:{token_start + token_len}]: "
                f"expected {token_len}, got {raw_tokens.size}"
            )

        add_bos = (
            self.bos_eos_mode == "always"
            or (self.bos_eos_mode == "song_boundary" and is_first == 1)
        )
        add_eos = (
            self.bos_eos_mode == "always"
            or (self.bos_eos_mode == "song_boundary" and is_last == 1)
        )

        unit_width = 1 if self.mode == "flat" else self.vq_groups
        parts = []
        if add_bos:
            parts.append(np.full(unit_width, self.bos_token_id, dtype=np.int64))
        parts.append(raw_tokens)
        if add_eos:
            parts.append(np.full(unit_width, self.eos_token_id, dtype=np.int64))

        seq = np.concatenate(parts, axis=0) if len(parts) > 1 else raw_tokens
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

