import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List, Optional
from einops import rearrange

from .modules.config import V_MORTMArgs
from .modules.layers import MORTMDecoder
from .modules.audio_patch import Vision, UnVision
from .modules.progress import LearningProgress

class V_MORTM(nn.Module):
    def __init__(self, args: V_MORTMArgs, progress:LearningProgress):
        super(V_MORTM, self).__init__()

        self.vision = Vision(args.d_spect, args.patch_size, args.dropout)

        self.conv_d_model = nn.Linear(args.d_spect * args.patch_size, args.d_model)
        self.decoder = MORTMDecoder(args, batch_first=True, bias=True, layer_norm_eps=1e-5, progress=progress)
        self.conv_d_spect = nn.Linear(args.d_model, args.d_spect * args.patch_size)
        self.unvision = UnVision(args.d_spect, args.patch_size, args.dropout)

        self.Wout = nn.Linear(args.d_spect, args.d_spect)

    def forward(self, src: Tensor) -> Tensor:
        print(src.shape)
        v_spect = self.vision(src)
        x = self.conv_d_model(v_spect)
        x = self.decoder(x, memory=None, tgt_is_causal=True)
        x = self.conv_d_spect(x)
        x = self.unvision(x)
        x = self.Wout(F.gelu(x))
        return x

    def top_p_sampling_measure(self, src: Tensor, p=0.9, temperature=1.0) -> List[Tensor]:
        """
        トークンを生成するためのメソッドです。

        Args:
            src (Tensor): 入力テンソル
            p (float): 確率の閾値
            max_measure (int): 最大生成長
            temperature (float): 温度パラメータ

        Returns:
            List[Tensor]: 生成されたトークンのリスト
        """
        if isinstance(src, np.ndarray):
            src = torch.tensor(src, device=self.progress.get_device())
        src = src.unsqueeze(0)

        while len(src) < 1722:
            if len(src) >= self.vision.patch_size:
                src = self.vision(src)
                src = self.conv_d_model(src)
                src = self.decoder(src, memory=None, tgt_is_causal=True)
                src = self.conv_d_spect(src)
                src = self.unvision(src)
                src = self.Wout(F.gelu(src))

                # Apply softmax to the output
                src = F.softmax(src, dim=-1)

