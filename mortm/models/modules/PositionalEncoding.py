import torch
import torch.nn as nn
import math

from .progress import LearningProgress


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, progress: LearningProgress, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encodingのテンソルを生成
        pe = torch.zeros(max_len, d_model, device=progress.get_device())
        position = torch.arange(0, max_len, dtype=torch.float, device=progress.get_device()).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(progress.get_device())

        pe[:, 0::2] = torch.sin(position * div_term,)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
