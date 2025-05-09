import numpy as np
import torch
from torch import Tensor
from mortm.models.modules.progress import LearningProgress


class EpochObserver:
    def __init__(self, max:int):
        self.loss = np.array([])
        self.max = max
        pass

    def add(self, l):
        if len(self.loss) >= self.max:
            self.loss = self.loss[1:]
        self.loss = np.append(self.loss, l)

    def get(self):
        return self.loss.sum() / len(self.loss)

