'''
Tokenizerで変換したシーケンスを全て保管します。
'''
import random
import time

import torch
from torch.utils.data import Dataset
import numpy as np
from .progress import LearningProgress


class MORTM_DataSets(Dataset):
    def __init__(self, progress: LearningProgress):
        self.key: list = list()
        self.value: list = list()
        self.progress = progress

    def __len__(self):
        return len(self.key)

    def __getitem__(self, item):
        return (torch.tensor(self.key[item], dtype=torch.long, device=self.progress.get_device()),
                torch.tensor(self.value[item], dtype=torch.long, device=self.progress.get_device()))

    def add_data(self, music_seq: np.ndarray):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            seq = music_seq[f'array_{i + 1}'].tolist()
            if 0 < len(seq['key']) < 220:
                self.key.append(seq['key'].tolist())
                self.value.append(seq['value'].tolist())
                suc_count += 1

        return suc_count