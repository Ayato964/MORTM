from typing import Optional, Literal

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, LayerNorm, MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, _generate_square_subsequent_mask
from typing import Tuple, List
import numpy as np
from .PositionalEncoding import PositionalEncoding
from .attention import FlashSelfAttentionM, FlashCrossAttentionM, MultiHeadAttentionRPR, linear
from .progress import LearningProgress


world_size = 1
rank = 0
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


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
                                   layer_norm_eps=1e-5, num_decoder_layer=d_layer, progress=progress)

        self.encoder = MORTMEncoder(d_model=d_model, dim_ff=dim_feedforward, num_layer=e_layer,
                                    num_head=num_heads, dropout=dropout,
                                    batch_first=True, bias=True,
                                    layer_norm_eps=1e-5,
                                    progress=progress)

        print("Use RPR Transformer")
        print(f"Input Vocab Size:{vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, vocab_size).to(self.progress.get_device())

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0).to(self.progress.get_device())
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

        #out = out.permute(1, 0, 2)
        score: Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

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


class MORTMEncoder(nn.Module):
    def __init__(self, d_model, dim_ff, num_head, num_layer, dropout, batch_first, bias, layer_norm_eps, progress):
        super(MORTMEncoder, self).__init__()
        self.num_layer = num_layer
        self.layers = _get_clones(MORTMEncoderLayer(d_model=d_model, dim_ff=dim_ff, num_head=num_head, dropout=dropout, batch_first=batch_first,
                                                    bias=bias, layer_norm_eps=layer_norm_eps, progress=progress), self.num_layer)

        self.norm = LayerNorm(d_model, eps=1e-5, bias=True, dtype=torch.float32)

    def forward(self, src, mask, src_key_padding_mask, is_causal):
        memory = src

        for mod in self.layers:
            memory = mod(
                memory,
                mask,
                src_key_padding_mask,
                is_causal
            )

        return self.norm(memory)


class MORTMEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_ff, num_head, dropout, batch_first, bias, layer_norm_eps, progress):
        super(MORTMEncoderLayer, self).__init__()

        self.d_model = d_model
        self.dim_ff = dim_ff
        self.dropout = dropout


        self.self_attn =FlashSelfAttentionM(d_model, num_head, dropout, progress=progress)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=True, dtype=torch.float32)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=True, dtype=torch.float32)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.f_linear = nn.Linear(self.d_model, self.dim_ff)
        self.f_drop = nn.Dropout(dropout)
        self.ff_linear = nn.Linear(self.dim_ff, self.d_model)

    def forward(self, memory, mask, src_key_padding_mask, is_causal):
        y = memory

        y = y + self.self_block(self.norm1(y), mask, src_key_padding_mask, is_causal)

        y = y + self.ff_block(self.norm2(y))

        return y

    def self_block(self, y, mask, src_key_padding_mask, is_causal):

        y,  _ = self.self_attn(y, key_padding_mask=src_key_padding_mask,
                               need_weights=True, attn_mask=mask, is_causal=is_causal)

        return self.dropout1(y)

    def ff_block(self, y: Tensor):
        y = self.f_linear(y)
        y = F.relu(y)
        y = self.f_drop(y)
        y = self.ff_linear(y)
        return self.dropout2(y)


class MORTMDecoder(nn.Module):
    def __init__(self,d_model, dim_ff, num_head, dropout, batch_first, bias, layer_norm_eps,  num_decoder_layer:int, progress):
        super(MORTMDecoder, self).__init__()
        self.num_layer = num_decoder_layer
        self.layers = _get_clones(MORTMDecoderLayer(d_model=d_model, dim_ff=dim_ff,
                                                    num_head=num_head, dropout=dropout,
                                                    batch_first=batch_first, bias=bias,
                                                    layer_norm_eps=layer_norm_eps, progress=progress), self.num_layer)
        self.norm = LayerNorm(d_model, eps=1e-5, bias=True, dtype=torch.float32)
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

        return self.norm(output)


class MORTMDecoderLayer(nn.Module):

    def __init__(self, d_model, dim_ff, num_head, dropout, batch_first, bias, layer_norm_eps, progress):
        super(MORTMDecoderLayer, self).__init__()
        self.n_head = num_head
        self.d_model = d_model
        self.cross_attention: FlashCrossAttentionM = FlashCrossAttentionM(d_model, num_head, dropout)
        self.self_attention: FlashSelfAttentionM =FlashSelfAttentionM(d_model, num_head, dropout, progress=progress)

        #self.ffn = FFN(d_model, dim_ff, dropout)
        self.ffn = MoE(d_model, dim_ff, 5, 2, 1, 1, )

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=torch.float32)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=torch.float32)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, dtype=torch.float32)

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
        y, _ = self.self_attention(y, key_padding_mask=tgt_key_padding_mask,
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
        y, _ = self.cross_attention(y, mem, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                    attn_mask=attn_mask, is_causal=is_causal)

        #y, _ = self.cross_attention(y, mem, mem, key_padding_mask=memory_key_padding_mask,
        #                            is_causal=is_causal)
        return self.dropout2(y)

    def ff_block(self, y: Tensor):
        return self.dropout3(self.ffn(y))


class FFN(nn.Module):

    def __init__(self, d_model, ff_d, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, ff_d)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_d, d_model)

    def forward(self, x: Tensor):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        return y


class MLP(nn.Module):

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):

    def __init__(self, d_model, num_experts, activated_experts, num_groups, top_k_groups, route_scale=1, score_type="softmax"):
        """
        :param d_model: 埋め込み次元数
        :param num_experts: 専門家の数
        :param activated_experts: 選ばれる専門家の数(top_k)
        :param num_groups:　専門家のグループ数
        :param top_k_groups:　選ばれるグループの数(top_k)
        :param route_scale: スケーリング係数
        :param score_type:　スケールのタイプ
        """
        super().__init__()
        self.dim = d_model
        self.topk = activated_experts
        self.n_groups = num_groups
        self.topk_groups = top_k_groups
        self.score_func = score_type
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, d_model))
        self.bias = nn.Parameter(torch.empty(num_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=torch.bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, d_model, dim_ff, num_experts, topk_experts, num_group, topk_groups, route_scale=1):
        """
        :param d_model: 埋め込み次元数
        :param dim_ff: FFNの次元数
        :param num_experts: 専門家の数
        :param topk_experts: 選択される専門家の数(top_k)
        :param num_group: 専門家のグループの数
        :param topk_groups: 選択される専門家のグループの数(top_k)
        :param route_scale: スケーリングの値
        """
        super().__init__()
        self.dim = d_model
        self.n_routed_experts = num_experts
        self.n_local_experts = self.n_routed_experts // world_size
        self.n_activated_experts = topk_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(d_model, num_experts, topk_experts, num_group, topk_groups, route_scale=route_scale)

        self.experts = nn.ModuleList([Expert(d_model, dim_ff) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(d_model, dim_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)