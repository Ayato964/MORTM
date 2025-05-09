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
        v_spect = self.vision(src)
        x = self.conv_d_model(v_spect)
        x = self.decoder(x, memory=None, tgt_is_causal=True)
        x = self.conv_d_spect(x)
        x = self.unvision(x)
        x = self.Wout(F.gelu(x))
        return x