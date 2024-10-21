import torch
import torch.nn as nn
import math
from torch.nn.modules.transformer import _get_clones


class RelativePositionMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device, dropout=0.1, max_len=5000):
        super(RelativePositionMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model).to(device)
        self.W_k = nn.Linear(d_model, d_model).to(device)
        self.W_v = nn.Linear(d_model, d_model).to(device)

        self.W_o = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        # 相対位置エンコーディングのパラメータ
        self.Er = nn.Parameter(torch.Tensor(max_len, self.d_k).to(device)).to(device)
        nn.init.xavier_uniform_(self.Er).to(device)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.size()
        if seq_len > self.Er.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum relative positions {self.Er.size(0)}")

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, L, Dk)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 相対位置エンコーディングの取得
        Er = self.Er[:seq_len, :]  # (L, Dk)

        # アテンションスコアの計算
        AC = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L, L)

        # 相対位置のインデックスを計算
        rel_pos = torch.arange(seq_len - 1, -seq_len, -1, device=x.device)
        rel_pos = rel_pos + (self.Er.size(0) - 1)  # シフトして正のインデックスに
        Er_embed = self.Er.index_select(0, rel_pos)  # (2L -1, Dk)

        BD = torch.matmul(Q, Er_embed.transpose(0, 1))  # (B, H, L, 2L -1)
        BD = self._relative_shift(BD)

        # スコアの合計
        scores = (AC + BD) / math.sqrt(self.d_k)

        # マスクの適用
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(key_padding_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, Dk)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)

        return out

    def _relative_shift(self, x):
        # x: (B, H, L, 2L -1)
        batch_size, heads, length, _ = x.size()
        x = x.view(batch_size, heads, length, 2 * length -1)

        x = x[:, :, :, length -1:]

        return x


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward , dropout, max_len, device):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = RelativePositionMultiHeadAttention(d_model, n_head, device, dropout=dropout, max_len=max_len).to(device)
        self.linear1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model).to(device)

        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)

        self.activation = nn.ReLU().to(device)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, attn_mask=mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, dim_feedforward , dropout, max_len, device):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = _get_clones(CustomTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, max_len, device), num_layers).to(device=device)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):

        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, mask=mask,
                                    src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)

        if self.norm:
            output = self.norm(output)

        return output
