'''
Tokenizerで変換したシーケンスを全て保管します。
'''

import torch
from torch.utils.data import Dataset
import numpy as np
from mortm.models.modules.progress import LearningProgress


class MORTM_SEQDataset(Dataset):
    def __init__(self, progress: LearningProgress, positional_length):
        self.seq: list = list()
        self.progress = progress
        self.positional_length = positional_length

    def __len__(self):
        return len(self.seq)

    def add_data(self, music_seq: np.ndarray):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            seq = music_seq[f'array{i + 1}'].tolist()
            if 90 < len(seq) < self.positional_length and seq.count(4) < 3:
                self.seq.append(seq)
                suc_count += 1
        return suc_count
    def __getitem__(self, item):
        return torch.tensor(self.seq[item], dtype=torch.long, device=self.progress.get_device())


class MORTM_DataSets(Dataset):
    def __init__(self, progress: LearningProgress, positional_length):
        self.key: list = list()
        self.value: list = list()
        self.progress = progress
        self.positional_length = positional_length

    def __len__(self):
        return len(self.key)

    def __getitem__(self, item):
        return (torch.tensor(self.key[item], dtype=torch.long, device=self.progress.get_device()),
                torch.tensor(self.value[item], dtype=torch.long, device=self.progress.get_device()))

    def add_data(self, music_seq: np.ndarray):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            seq = music_seq[f'array_{i + 1}'].tolist()
            if (90 < len(seq['key']) < self.positional_length) or (4 in seq['key'] and len(seq['key']) < 90):
                if len(seq['value']) < self.positional_length:
                    self.key.append(seq['key'].tolist())
                    self.value.append(seq['value'].tolist())
                    suc_count += 1

        return suc_count