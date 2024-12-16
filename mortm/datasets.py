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
        self.musics_seq: list = []
        self.progress = progress

    def __len__(self):
        return len(self.musics_seq)

    def __getitem__(self, item):
        return torch.tensor(self.musics_seq[item], dtype=torch.long, device=self.progress.get_device())

    def add_data(self, music_seq: np.ndarray):
        suc_count = 0
        for i in range(len(music_seq) - 1):
            aya_node = music_seq[f'array{i + 1}']
            if 100 < len(aya_node) < 1000 or 2 in aya_node:
                self.musics_seq = self.musics_seq + [aya_node.tolist()]
                suc_count += 1
        return suc_count

    def split_seq_data(self):
        new_music_seq = [[]]
        for i in range(len(self.musics_seq)):
            result = []
            current_sublist = []
            for value in self.musics_seq[i]:
                current_sublist.append(value)
                if value == 2:
                    result.append(current_sublist)
                    current_sublist = []
            new_music_seq = new_music_seq + result

        self.musics_seq = new_music_seq

        pass

    def set_train_data(self):
        print(f"Set train data....{self._get_shape(self.musics_seq)}")
        for music_seq in self.musics_seq:
            tgt_data = []
            for i in range(len(music_seq) - 1):
                tgt_data.append(music_seq[i + 1])
            if self.tgt_seq is None:
                self.tgt_seq = [tgt_data]
            else:
                self.tgt_seq = self.tgt_seq + [tgt_data]

        print(f"clear! train shape is:{self._get_shape(self.tgt_seq)}")

    def _get_shape(self, lst):
        if isinstance(lst, list):
            return [len(lst)] + self._get_shape(lst[0]) if lst else []
        return []

    def set_padding(self):
        #        self._padding(self.tgt_seq)
        self._padding(self.musics_seq)

        pass

    def get_max_length(self, target):
        max_lengths = []
        for t in target:
            max_lengths.append(len(t))
        max_length = max(max_lengths)
        return max_length

    def _padding(self, target: list):
        max_lengths = []
        for t in target:
            max_lengths.append(len(t))
        max_length = max(max_lengths)
        print(f"Max length is {max_length}")
        for t in target:
            if len(t) < max_length:
                for _ in range(max_length - len(t)):
                    t.append(0)
        pass


class MORTMTuringDataset(Dataset):
    def __init__(self, progress: LearningProgress):
        self.musics_seq: list = []
        self.tgt_seq: list = []
        self.progress = progress

    def __len__(self):
        return len(self.musics_seq)

    def __getitem__(self, item):
        return (torch.tensor(self.musics_seq[item], dtype=torch.long, device=self.progress.get_device()),
                torch.tensor(self.tgt_seq[item], dtype=torch.long, device=self.progress.get_device()))

    def add_data(self, music_seq: np.ndarray):
        for i in range(len(music_seq)):
            aya_dict = music_seq[f'array{i}']
            if isinstance(aya_dict, np.ndarray):
                aya_dict = aya_dict.item()
            self.musics_seq = self.musics_seq + [aya_dict['src']]
            self.tgt_seq = self.tgt_seq + [aya_dict['tgt']]

        pass
