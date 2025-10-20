import json
from typing import Optional, Literal

import numpy
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from .modules.config import MORTM_LIVE_Args
from .modules.layers import *


class Vision(nn.Module):
    def __init__(self, args: MORTM_LIVE_Args):
        super().__init__()
        encoder_output_dim = args.instrument_num * 8 * (args.pianoroll_time_step // 4) * 16
        self.encoder = VisionEncoder(args, encoder_output_dim)
        self.decoder = VisionDecoder(args, encoder_output_dim=encoder_output_dim, encoder_output_shape=(args.instrument_num * 2, args.pianoroll_time_step // 4, 16))

    def forward(self, pianoroll: Tensor):
        pianoroll = re.arrange(pianoroll, 'b t p c -> b c t p')
        encoded, mu, log_var = self.encoder(pianoroll)
        decoded = self.decoder(encoded)

        return decoded, mu, log_var


class MORTMLive(nn.Module):
    def __init__(self, args: MORTM_LIVE_Args, progress=None):
        super(MORTMLive, self).__init__()
        self.args = args
        self.vision = Vision(args)
        self.mortm = MORTMDecoder(args=args, progress=progress)

