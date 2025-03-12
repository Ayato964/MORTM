from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, LayerNorm, MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, _generate_square_subsequent_mask
from typing import Tuple, List
import numpy as np
from .PositionalEncoding import PositionalEncoding
from .attention import FlashSelfAttentionM, FlashCrossAttentionM, MultiHeadAttentionRPR
from .progress import LearningProgress



class MORTM(nn.Module):
    def __init__(self, vocab_size, progress: LearningProgress, d_layer=15, e_layer=15, num_heads=12, d_model=768,
                 dim_feedforward=3072, dropout=0.2, decoder_only:bool=False,
                 position_length=400):
        super(MORTM, self).__init__()
        self.progress = progress
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.decoder_only = decoder_only
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, progress, dropout, position_length * 4).to(
            self.progress.get_device())
        #Transformerの設定
        if not decoder_only:
            self.decoder = MORTMDecoder(d_model=d_model, dim_ff=dim_feedforward,
                                   num_head=num_heads, dropout=dropout,
                                   batch_first=True, bias=True,
                                   layer_norm_eps=1e-5, num_decoder_layer=d_layer)
        encoder_layer = MORTMEncoderLayer(
            d_model=d_model, dim_ff=dim_feedforward,
            num_head=num_heads, dropout=dropout,
            batch_first=True, bias=True,
            layer_norm_eps=1e-5
        )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=e_layer, norm=LayerNorm(d_model, 1e-5, bias=True))

        print("Use RPR Transformer")
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, vocab_size).to(self.progress.get_device())

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(self.progress.get_device())
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, input_padding_mask=None,
                tgt_padding_mask=None, src_is_causal=False, tgt_is_causal=False):
        if tgt_mask is None and tgt_is_causal:
            tgt_mask = _generate_square_subsequent_mask(tgt.size(1)).to(self.progress.get_device())

        sec_e: Tensor = self.embedding(src)
        sec_e = sec_e.permute(1, 0, 2)

        src_p: Tensor = self.positional(sec_e)
        src_p = src_p.permute(1, 0, 2)

        if tgt is not None:
            tgt_e = self.embedding(tgt)
            tgt_e = tgt_e.permute(1, 0, 2)
            tgt_p = self.positional(tgt_e)
            tgt_p = tgt_p.permute(1, 0, 2)
        else:
            tgt_p = src_p

        if not self.decoder_only:
            memory = self.encoder(src=src_p, mask=src_mask, src_key_padding_mask=input_padding_mask, is_causal=src_is_causal)

            out = self.decoder(tgt=tgt_p, memory=memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=input_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask, memory_is_causal=src_is_causal, tgt_is_causal=tgt_is_causal)
        else:
            out = self.encoder(src=src_p, mask=tgt_mask, src_key_padding_mask=input_padding_mask, is_casual=src_is_causal)


        out = out.permute(1, 0, 2)
        score: Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

    def top_p_sampling_measure_decoder(self, input_seq, p=0.9, max_measure=20, temperature=1.0):
        self.eval()
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=self.progress.get_device())
        seg: Tensor = self.split_tensor_at_value(input_seq, 3, include_split=True)
        gen = torch.tensor([2], dtype=torch.long, device=self.progress.get_device())


        generated = torch.cat([torch.tensor(s) for s in seg[:-1]]).to(self.progress.get_device())
        src = torch.concatenate([s for s in seg[:-1]])
        src = torch.concatenate((src, gen, seg[-1])).to(self.progress.get_device())

        for i in range(max_measure):
            while not (src[-1] == 391 or src[-1] == 392):
                logit = self(src=src.unsqueeze(0))
                outputs = logit.view(-1, logit.size(-1)).to(self.progress.get_device())
                token = self.top_p_sampling(outputs[-1], p=p, temperature=temperature)
                src = torch.concatenate((src, torch.tensor([token], dtype=torch.long, device=self.progress.get_device())))
            if src[-1] == 392:
                break

            seg = self.split_tensor_at_value(src, 3, include_split=True)
            seg[-2] = seg[-2][:-1]
            seg[-1] = seg[-1][:-1]
            generated = torch.cat((generated, seg[-1])).to(self.progress.get_device())
            #print(f"seg:{seg}   generated:{generated}    src:{src}")
            if len(seg) > 8:
                src = torch.cat([seq for seq in seg[len(seg) - 8:]]).to(self.progress.get_device())
            else:
                src = torch.cat([seq for seq in seg]).to(self.progress.get_device())
            src = torch.cat((src, torch.tensor([2], dtype=torch.long, device=self.progress.get_device()))).to(self.progress.get_device())


        return generated
    def top_p_sampling_measure(self, input_seq, p=0.9, max_measure=20, temperature=1.0, context_measure=8):
        self.eval()
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=self.progress.get_device())
        seg: Tensor = self.split_tensor_at_value(input_seq, 3, include_split=True)
        tgt = torch.tensor([2], dtype=torch.long, device=self.progress.get_device())
        tgt = torch.concatenate((tgt, seg[-1])).to(self.progress.get_device())
        point = 0 if len(seg[:-1]) - context_measure <= 0 else len(seg[:-1]) - context_measure

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
            if len(seg) > context_measure:
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
        self.self_attn =FlashSelfAttentionM(d_model, num_head, dropout)
        #self.self_attn = MultiheadAttention(d_model, num_head, dropout, batch_first=True)


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
        self.n_head = num_head
        self.d_model = d_model
        self.cross_attention: FlashCrossAttentionM = FlashCrossAttentionM(d_model, num_head, dropout)
        #self.cross_attention = MultiheadAttention(d_model, num_head, dropout, batch_first=False)
        self.self_attention: FlashSelfAttentionM =FlashSelfAttentionM(d_model, num_head, dropout)
        #self.self_attention = MultiheadAttention(d_model, num_head, dropout, batch_first=True)

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

        y = tgt

        y = y + self.self_block(self.norm1(y), tgt_mask, tgt_key_padding_mask, tgt_is_causal) #相対位置マルチヘッドアテンションを適用

        y = y + self.cross_block(self.norm2(y), memory, memory_mask,
                                 memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                 is_causal=memory_is_causal) # マルチヘッドアテンションを適用

        y = y + self.ff_block(self.norm3(y)) # フィードフォワード層を適用

        return y

    def self_block(self,
                   y: Tensor,
                   attn_mask: Optional[Tensor],
                   tgt_key_padding_mask: Optional[Tensor],
                   is_causal: bool = False,
                   ):

        #print(y.shape)
        y, _ = self.self_attention(y, y, y, key_padding_mask=tgt_key_padding_mask,
                                   need_weights=True, attn_mask=attn_mask, is_causal=is_causal)
        #print(y.shape)

        return self.dropout1(y)

    def cross_block(self,
                    y: Tensor,
                    mem: Tensor,
                    attn_mask: Optional[Tensor],
                    memory_key_padding_mask: Optional[Tensor],
                    tgt_key_padding_mask: Optional[Tensor],
                    is_causal: bool = False,
                    ):
        y, _ = self.cross_attention(y, mem, mem, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                    attn_mask=attn_mask, is_causal=is_causal)

        #y, _ = self.cross_attention(y, mem, mem, key_padding_mask=memory_key_padding_mask,
        #                            attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout2(y)

    def ff_block(self, y: Tensor):
        y = self.linear1(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return self.dropout3(y)
