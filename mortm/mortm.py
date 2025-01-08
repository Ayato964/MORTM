from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, LayerNorm, MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
import numpy as np
from .PositionalEncoding import PositionalEncoding
from .rpr import MultiHeadAttentionRPR
from .progress import LearningProgress
from torch.distributions import Categorical

def generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


class MORTM(nn.Module):
    def __init__(self, vocab_size, progress: LearningProgress, d_layer=12, e_layer=9, num_heads=16, d_model=1024,
                 dim_feedforward=4096, dropout=0.2,
                 position_length=8500):
        super(MORTM, self).__init__()

        self.progress = progress
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, progress, dropout, position_length).to(
            self.progress.get_device())

        #Transformerの設定
        self.decoder = MORTMDecoder(d_model=d_model, dim_ff=dim_feedforward,
                               num_head=num_heads, dropout=dropout,
                               batch_first=False, bias=True,
                               layer_norm_eps=1e-5, num_decoder_layer=d_layer)
        encoder_layer = MORTMEncoderLayer(
            d_model=d_model, dim_ff=dim_feedforward,
            num_head=num_heads, dropout=dropout,
            batch_first=False, bias=True,
            layer_norm_eps=1e-5
        )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=e_layer, norm=LayerNorm(d_model, 1e-5, bias=True))

        print("Use RPR Transformer")
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, vocab_size).to(self.progress.get_device())

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(self.progress.get_device())
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, input_padding_mask=None,
                tgt_padding_mask=None):
        if tgt_mask is None:
            mask = generate_square_subsequent_mask(tgt.shape[1]).to(self.progress.get_device())
        else:
            mask = tgt_mask

        sec_e: Tensor = self.embedding(src)
        sec_e = sec_e.permute(1, 0, 2)

        src_p: Tensor = self.positional(sec_e)

        if tgt is not None:
            tgt_e = self.embedding(tgt)
            tgt_e = tgt_e.permute(1, 0, 2)
            tgt_p = self.positional(tgt_e)
        else:
            tgt_p = src_p

        memory = self.encoder(src=src_p, mask=src_mask, src_key_padding_mask=input_padding_mask)

        out = self.decoder(tgt=tgt_p, memory=memory, tgt_mask=mask, tgt_key_padding_mask=tgt_padding_mask)

        out = out.permute(1, 0, 2)

        score: Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

    def top_p_sampling_measure(self, input_seq, p=0.9, max_measure=20, temperature=1.0):
        self.eval()
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=self.progress.get_device())
        seg: Tensor = self.split_tensor_at_value(input_seq, 3, include_split=True)
        tgt = torch.tensor([2], dtype=torch.long, device=self.progress.get_device())
        tgt = torch.concatenate((tgt, seg[-1])).to(self.progress.get_device())
        point = 0 if len(seg[:-1]) - 8 <= 0 else len(seg[:-1]) - 8

        src = torch.tensor([], dtype=torch.long, device=self.progress.get_device())

        for i in range(point, len(seg[point:-1])):
            src = torch.concatenate((src, seg[i]))
        generated = src.clone()

        for i in range(max_measure):
            while not (tgt[-1] == 391 or tgt[-1] == 392):
                logit = self(src=src.unsqueeze(0), tgt=tgt.unsqueeze(0))
                outputs = logit.view(-1, logit.size(-1)).to(self.progress.get_device())
                token = self.top_p_sampling(outputs[-1], p=p, temperature=temperature)
                tgt = torch.concatenate((tgt, torch.tensor([token], dtype=torch.long,
                                                           device=self.progress.get_device())), dim=0)

            if tgt[-1] == 392:
                break
            generated = torch.concatenate((generated, tgt[1: -1]))
            src = torch.concatenate((src, tgt[1:-1]))
            tgt = torch.tensor([2], device=self.progress.get_device())
            seg = self.split_tensor_at_value(src, 3, include_split=True)
            if len(seg) > 8 :
                src = torch.tensor([], dtype=torch.long, device=self.progress.get_device())
                for i in seg[1:]:
                    src = torch.concatenate((src, i))

        return generated

    def top_p_sampling(self, logits, p=0.9, temperature=1.0) -> int:

        logits = logits / temperature
        # logitsをソフトマックスで確率分布に変換
        probs = self.softmax(logits)
        # 確率の降順に並べ替え、そのインデックスを取得
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 累積確率を計算
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 累積確率がpを超えるインデックスを取得
        cutoff_index = torch.where(cumulative_probs > p)[0][0]

        # 上位pに入らないトークンの確率を0にする
        sorted_probs[cutoff_index + 1:] = 0

        # 確率を再正規化
        sorted_probs /= torch.sum(sorted_probs)

        # トークンをサンプリング
        sampled_index = torch.multinomial(sorted_probs, 1)

        # インデックスを元の順序に戻す
        return sorted_indices[sampled_index].item()

    def split_tensor_at_value(self, tensor: Tensor, split_value, include_split=True):
        """
        指定した値を基準にテンソルを分割します。

        Args:
            tensor (torch.Tensor): 1次元のテンソルを想定しています。
            split_value (int or float): 分割の基準となる値。
            include_split (bool, optional): 分割値を各セグメントに含めるかどうか。デフォルトは True。

        Returns:
            List[torch.Tensor]: 分割されたテンソルのリスト。
        """
        if tensor.dim() != 1:
            raise ValueError("この関数は1次元のテンソルに対してのみ動作します。")

        # 分割値が存在するインデックスを取得
        split_indices = (tensor == split_value).nonzero(as_tuple=True)[0]

        if len(split_indices) == 0:
            # 分割値が見つからない場合、元のテンソルをそのまま返す
            return [tensor]

        segments = []
        num_splits = len(split_indices)

        for i in range(num_splits):
            start = split_indices[i]
            if include_split:
                start = start  # 分割値を含める場合
            else:
                start = split_indices[i] + 1  # 分割値を含めない場合

            if i + 1 < num_splits:
                end = split_indices[i + 1]
            else:
                end = len(tensor)

            if include_split:
                end = end  # 次の分割値の位置まで含める
            else:
                end = end  # 次の分割値の位置まで含めない

            segment = tensor[start:end]
            segments.append(segment)

        return segments


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory


class MORTMEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, dim_ff, num_head, dropout, batch_first, bias, layer_norm_eps):
        super(MORTMEncoderLayer, self).__init__(d_model=d_model, dim_feedforward=dim_ff, nhead=num_head, dropout=dropout,
                                                batch_first=batch_first, bias=bias, layer_norm_eps=layer_norm_eps)
        self.self_attn = MultiHeadAttentionRPR(embed_dim=d_model, num_heads=num_head, dropout=dropout, bias=bias)


class MORTMDecoder(nn.Module):
    def __init__(self,d_model, dim_ff, num_head, dropout, batch_first, bias, layer_norm_eps,  num_decoder_layer:int):
        super(MORTMDecoder, self).__init__()
        self.num_layer = num_decoder_layer
        self.layers = _get_clones(MORTMDecoderLayer(d_model=d_model, dim_ff=dim_ff,
                                                    num_head=num_head, dropout=dropout,
                                                    batch_first=batch_first, bias=bias,
                                                    layer_norm_eps=layer_norm_eps), self.num_layer)
        self.norm = LayerNorm(d_model, eps=1e-5, bias=True)
    def forward(self, tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False, **kwargs) -> Tensor:

        output = tgt
        for mod in self.layers:
            mod: MORTMDecoderLayer
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            pass
        return self.norm(output)


class MORTMDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_ff, num_head, dropout, batch_first, bias, layer_norm_eps):
        super(MORTMDecoderLayer, self).__init__()
        self.multi_head_attention: MultiheadAttention = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_head,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias
        )
        self.rpr_attention: MultiHeadAttentionRPR = MultiHeadAttentionRPR(
            embed_dim=d_model,
            num_heads=num_head,
            dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model, bias=bias)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        )-> Tensor:
        if tgt_is_causal is None:
            tgt_is_causal = False

        y = tgt

        y = y + self.rpr_block(self.norm1(y), tgt_mask, tgt_key_padding_mask, tgt_is_causal) #相対位置マルチヘッドアテンションを適用

        y = y + self.multi_block(self.norm2(y), memory, memory_mask, memory_key_padding_mask, memory_is_causal) # マルチヘッドアテンションを適用

        y = y + self.ff_block(self.norm3(y)) # フィードフォワード層を適用

        return y

    def rpr_block(self,
                    y: Tensor,
                    attn_mask: Optional[Tensor],
                    key_padding_mask: Optional[Tensor],
                    is_causal: bool = False,
                    ):
        y = self.rpr_attention(y,y,y, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]

        return self.dropout1(y)

    def multi_block(self,
                  y: Tensor,
                  mem: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False,
                  ):
        y = self.multi_head_attention(y, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask,need_weights=False, is_causal=is_causal)[0]
        return self.dropout2(y)

    def ff_block(self, y: Tensor):
        y = self.linear1(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return self.dropout3(y)
