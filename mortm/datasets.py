import random
import time

import torch
from torch.utils.data import Dataset
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class MORTM_DataSets(Dataset):
    def __init__(self):
        self.musics_seq = None
        self.tgt_seq = None

    def __len__(self):
        return len(self.musics_seq)

    def __getitem__(self, item):
        return torch.tensor(self.musics_seq[item], dtype=torch.int).to(get_device()), torch.tensor(self.tgt_seq[item], dtype=torch.int).to(get_device())

    def add_data(self, music_seq: np.ndarray):
        if self.musics_seq is None:
            if len(music_seq.shape) == 1:
                self.musics_seq = [music_seq.tolist()]
            else:
                self.musics_seq = music_seq.tolist()
        else:
            if len(music_seq.shape) == 1:
                music_seq: list = [music_seq.tolist()]
                self.musics_seq = self.musics_seq + music_seq
            else:
                self.musics_seq = self.musics_seq + music_seq.tolist()

        print(self._get_shape(self.musics_seq))
        pass

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