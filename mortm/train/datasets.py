'''
Tokenizerで変換したシーケンスを全て保管します。
'''
import os.path
import random
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from mortm.models.modules.progress import LearningProgress
from einops import rearrange

class MORTM_SEQDataset(Dataset):
    def __init__(self, progress: LearningProgress, positional_length, min_length, is_random_delete_key = False):
        self.seq: list = list()
        self.progress = progress
        self.positional_length = positional_length
        self.min_length = min_length
        self.is_random_delete_key = is_random_delete_key

    def __len__(self):
        return len(self.seq)

    def add_data(self, music_seq: np.ndarray, *args):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            seq = music_seq[f'array{i + 1}'].tolist()
            if (self.min_length if seq[0] != 7 else 42) < len(seq) < self.positional_length and seq.count(4) < 3:
                self.seq.append(seq)
                suc_count += 1
        return suc_count
    def __getitem__(self, item):
        if self.is_random_delete_key:
            i = random.random()
        else:
            i = 1
        if i < 0.5:
            return torch.tensor(self.seq[item][1:], dtype=torch.long, device=self.progress.get_device())
        else:
            return torch.tensor(self.seq[item], dtype=torch.long, device=self.progress.get_device())


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
        return torch.tensor(self.src_list[item], dtype=torch.float32, device=self.progress.get_device())

    def add_data(self, piano_roll: np.ndarray, *args):
            self.src_list.append(piano_roll)

    def set_tokenizer_dataset(self):
        new_src_dataset = np.array([])
        for i in range(len(self.src_list)):
            if len(self.src_list[i]) > 0:
                if len(new_src_dataset) == 0:
                    new_src_dataset = self.src_list[i]
                else:
                    new_src_dataset = np.concatenate((new_src_dataset, self.src_list[i]), axis=0)
        self.src_list = new_src_dataset


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