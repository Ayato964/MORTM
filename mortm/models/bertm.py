import torch
from torch import nn
from torch import Tensor
from .modules.layers import MORTMEncoder
from .modules.PositionalEncoding import PositionalEncoding


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = self.attn(x)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)  # 各トークンに重み付け
        pooled = (weights * x).sum(dim=1)  # 重み付き平均 → (B, D)
        return pooled


class BERTM(nn.Module):

    def __init__(self, vocab_size, d_model, dim_ff, num_head, num_layer,
                 dropout, batch_first, bias, layer_norm_eps, progress):
        super(BERTM, self).__init__()
        self.encoder = MORTMEncoder(d_model=d_model, dim_ff=dim_ff, num_layer=num_layer,
                                     num_head=num_head, dropout=dropout,
                                     batch_first=batch_first, bias=bias,
                                     layer_norm_eps=layer_norm_eps,
                                     progress=progress)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, progress, dropout, max_len=5000)
        self.Wout = nn.Linear(d_model, vocab_size)

    def forward(self, src: Tensor, input_padding_mask=None):
        """
        src: 入力テンソル (バッチサイズ, シーケンス長, 特徴量次元)
        input_padding_mask: パディングマスク (バッチサイズ, シーケンス長)
        """
        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        src = self.positional(src)

        # Encoderのforwardメソッドを呼び出す
        out = self.encoder(src=src, src_mask=None,
                           src_key_padding_mask=input_padding_mask,
                           src_is_causal=False)

        out = self.Wout(out)

        return out