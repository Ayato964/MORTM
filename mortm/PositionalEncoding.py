import torch
import torch.nn as nn
import math

from .progress import LearningProgress


class RelativePositionalEncoding(nn.Module):

    def __init__(self, n_head, d_model, progress: LearningProgress, dropout=0.1, max_len=8000):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Relative Positional Encodingのテンソルを生成
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_len - 1, d_model), device=progress.get_device())
        )
        self.register_buffer(
            "relative_position_index", self._generate_relative_position_index(max_len)
        )
        self.n_head = n_head  # ヘッドの数を設定
        self._reset_parameters()

    def forward(self, x):
        # Relative Positional Encodingを適用
        batch_size, seq_len, d_model = x.size()
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:seq_len, :seq_len]
        ].permute(2, 0, 1).unsqueeze(0)
        relative_position_bias = relative_position_bias.repeat(batch_size, 1, 1, 1)

        # ヘッドの数に合わせて形状を変更
        num_heads = self.n_head  # ヘッドの数を修正
        head_dim = d_model // num_heads
        relative_position_bias = relative_position_bias.reshape(batch_size, num_heads, seq_len, seq_len)  # 要素数を調整

        x = x + relative_position_bias

        return self.dropout(x)

    def _generate_relative_position_index(self, max_len):
        """
        相対位置インデックスを生成する関数
        """
        range_vec = torch.arange(max_len)
        range_mat = range_vec.unsqueeze(0).repeat(max_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(distance_mat, -max_len + 1, max_len - 1)
        relative_position_index = distance_mat_clipped + max_len - 1
        return relative_position_index

    def _reset_parameters(self):
        """
        学習可能なパラメータを初期化する関数
        """
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)


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
