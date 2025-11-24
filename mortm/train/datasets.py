'''
Tokenizerで変換したシーケンスを全て保管します。
'''
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

    def add_data(self, music_seq: np.ndarray, *args):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            seq = music_seq[f'array{i + 1}'].tolist()
            if self.min_length < len(seq) < self.positional_length:
                self.seq.append(seq)
                suc_count += 1
        return suc_count
    def __getitem__(self, item):

        # 1. Pythonリスト(self.seq[item])を numpy.ndarray に変換
        sequence_array = np.array(self.seq[item])

        # AUGMENTATIONの適用条件
        apply_augmentation = (
                self.is_random_delete_key and
                self.system_tag is not None and
                self.program_token_id and
                self.sampling_inst_max > 0 and
                not self.mask_sample_task.isdisjoint(sequence_array)
        )

        if not apply_augmentation:
            # 条件を満たさない場合は、オリジナルのシーケンスを返す
            return torch.tensor(sequence_array, dtype=torch.long, device=self.progress.get_device())

        try:
            begin_tag_id, end_tag_id = self.system_tag

            footer_tokens = {626, 627}

            # --- 1. <SYSTEM> タグの開始インデックスを特定 ---
            # (get_tag内部でエラーチェックされるが、プレフィックス抽出のために先に見つける)
            ind_begin_tag_search = np.where(sequence_array == begin_tag_id)[0]
            if len(ind_begin_tag_search) == 0:
                # get_tag 同様のエラーを発生させる
                raise IndexError("Begin tag token not found in sequence.")
            ind_begin_tag = ind_begin_tag_search[0]

            # --- 2. プレフィックス (<EOS>など) を抽出 ---
            # <SYSTEM> よりも前の部分
            prefix_part = sequence_array[:ind_begin_tag].tolist()

            # --- 3. システムタグをフィルタリング ---
            # (get_tagは堅牢化されたものを使用)
            # system_seq_list = [8, 10, 9, ...]
            # ind_end_tag = (2 のインデックス)
            system_seq_list, ind_end_tag = get_tag(sequence_array, begin_tag_id, end_tag_id)

            present_inst_tokens = [
                token for token in system_seq_list
                if token in self.program_token_id
            ]
            inst_count = len(present_inst_tokens)

            # (楽器が1つ、またはサンプリング不要なケース)
            if inst_count <= 1:
                return torch.tensor(sequence_array, dtype=torch.long, device=self.progress.get_device())

            max_k = min(self.sampling_inst_max, inst_count)
            k = random.randint(1, max_k)
            kept_inst_tokens = set(random.sample(present_inst_tokens, k))

            if len(kept_inst_tokens) == inst_count:
                return torch.tensor(sequence_array, dtype=torch.long, device=self.progress.get_device())

            # 保持する楽器だけで新しいシステムタグを構築
            new_system_seq = [
                token for token in system_seq_list
                # 楽器トークンではない OR 保持する楽器トークンである
                if (token not in self.program_token_id) or (token in kept_inst_tokens)
            ]

            # --- 4. 音楽イベントをフィルタリング ---
            music_events_part = sequence_array[ind_end_tag + 1:]

            inst_token_indices = []
            for i, token in enumerate(music_events_part):
                if token in self.program_token_id:
                    inst_token_indices.append((i, token)) # (相対インデックス, トークンID)

            new_music_events = []

            if not inst_token_indices:
                # 楽器トークンがなければ、そのまま（<CGEN>タスクなど）
                new_music_events.extend(music_events_part.tolist())
            else:
                # 4a. 音楽プレフィックス (e.g., <MGEN>) を追加
                first_inst_idx = inst_token_indices[0][0]
                new_music_events.extend(music_events_part[:first_inst_idx].tolist())

                # 4b. 保持する楽器チャンクを追加
                for i, (chunk_start_idx, inst_token_id) in enumerate(inst_token_indices):
                    if inst_token_id not in kept_inst_tokens:
                        continue

                    # チャンクの終了位置を特定
                    if i + 1 < len(inst_token_indices):
                        # 次の楽器トークンの手前まで
                        next_chunk_start_idx = inst_token_indices[i+1][0]
                        chunk_end_idx = next_chunk_start_idx
                    else:
                        # 最後の楽器チャンク。フッターを探す
                        footer_start_idx = -1
                        # 最後のチャンク開始位置からフッターを探す
                        for j, tok in enumerate(music_events_part[chunk_start_idx:]):
                            if tok in footer_tokens:
                                # フッターの相対インデックスを発見
                                footer_start_idx = chunk_start_idx + j
                                break

                        if footer_start_idx != -1:
                            chunk_end_idx = footer_start_idx # フッターの手前まで
                        else:
                            chunk_end_idx = len(music_events_part) # フッターがなければ最後まで

                    chunk = music_events_part[chunk_start_idx : chunk_end_idx]
                    new_music_events.extend(chunk.tolist())

                # 4c. 音楽サフィックス (e.g., <TE>) を追加
                # 最後の楽器チャンクの開始位置からフッターを探す
                last_inst_start_idx = inst_token_indices[-1][0]
                footer_start_idx = -1
                for j, tok in enumerate(music_events_part[last_inst_start_idx:]):
                    if tok in footer_tokens:
                        footer_start_idx = last_inst_start_idx + j
                        break

                # フッターが見つかった場合、その位置から最後までを追加
                if footer_start_idx != -1:
                    new_music_events.extend(music_events_part[footer_start_idx:].tolist())

            # --- 5. 最終シーケンスの結合 ---
            # [プレフィックス] + [システムタグ] + [音楽イベント]
            final_sequence_list = prefix_part + new_system_seq + new_music_events

            #print(final_sequence_list) # デバッグ用
            return torch.tensor(final_sequence_list, dtype=torch.long, device=self.progress.get_device())

        except Exception as e:
            # ログを拡張
            print(f"Warning: Augmentation failed for item {item}, returning original. Error: {e}  {sequence_array}")
            return torch.tensor(sequence_array, dtype=torch.long, device=self.progress.get_device())


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
        return (torch.tensor(v, dtype=torch.long, device=self.progress.get_device()),
                torch.tensor(self.value[item], dtype=torch.long, device=self.progress.get_device()))

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
        return torch.tensor(self.src_list[item], device=self.progress.get_device())

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

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, item: int) :
        return self.src_list[item]


    def add_data(self, directory: List[str], filename: List[str]):
        for i in range(len(directory)):
            self.src_list.append(os.path.join(directory[i], filename[i]))



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