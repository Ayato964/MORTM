import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List, Optional

from einops import rearrange
from torch.nn.modules.transformer import _get_clones


class Vision(nn.Module):
    def __init__(self, d_spect, patch_size, dropout:float):
        super(Vision, self).__init__()
        self.d_spect = d_spect
        self.patch_size = patch_size
        self.dt = nn.Dropout(dropout)
        self.linear = nn.Linear(d_spect * patch_size, d_spect * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        assert S % self.patch_size == 0, f"Input length {S} must be divisible by patch size {self.patch_size}"
        x = rearrange(x, 'b (s p) d -> b s p d', s=S // self.patch_size, p=self.patch_size)
        x = rearrange(x, 'b s p d -> b s (p d)')
        x = self.linear(x)
        x = self.dt(x)
        return x

class UnVision(nn.Module):
    def __init__(self, d_spect, patch_size, dropout):
        super(UnVision, self).__init__()
        self.d_spect = d_spect
        self.patch_size = patch_size
        self.linear = nn.Linear(d_spect * patch_size, d_spect * patch_size)
        self.Wout = nn.Linear(d_spect, d_spect)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.dropout(x)
        x = rearrange(x, 'b s (p d) -> b s p d', p=self.patch_size, d=self.d_spect)
        x = rearrange(x, 'b s p d -> b (s p) d')
        x = self.Wout(x)
        return self.dropout2(x)