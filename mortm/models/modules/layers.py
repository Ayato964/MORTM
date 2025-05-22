import json
from typing import Optional, Literal

import numpy
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, LayerNorm, MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, _generate_square_subsequent_mask
from typing import Tuple, List
import numpy as np

from .attention import FlashSelfAttentionM, FlashCrossAttentionM, MultiHeadAttentionRPR, linear
from .config import MORTMArgs


world_size = 1
rank = 0
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory


class MORTMEncoder(nn.Module):
    def __init__(self, args: MORTMArgs, layer_norm_eps, progress):
        super(MORTMEncoder, self).__init__()
        self.num_layer = args.e_layer
        self.layers = _get_clones(MORTMEncoderLayer(args, layer_norm_eps=layer_norm_eps, progress=progress), self.num_layer)

        self.norm = LayerNorm(args.d_model, eps=1e-5, bias=True, dtype=torch.float32)

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
    def __init__(self, args: MORTMArgs, layer_norm_eps, progress):
        super(MORTMEncoderLayer, self).__init__()

        self.d_model = args.d_model
        self.dim_ff = args.dim_feedforward
        self.dropout = args.dropout


        self.self_attn =FlashSelfAttentionM(args.d_model, args.num_heads, args.dropout, progress=progress)
        if args.use_moe_encoder == True:
            self.ffn = MoE(args.d_model, args.dim_feedforward, args.num_experts, args.topk_experts, args.num_groups, args.topk_groups)
        else:
            self.ffn = self.mlp
            self.ff_linear = nn.Linear(args.d_model, args.dim_feedforward)
            self.ff_linear2 = nn.Linear(args.dim_feedforward, args.d_model)

        self.norm1 = LayerNorm(args.d_model, eps=layer_norm_eps, bias=True, dtype=torch.float32)
        self.norm2 = LayerNorm(args.d_model, eps=layer_norm_eps, bias=True, dtype=torch.float32)

        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)

    def forward(self, memory, mask, src_key_padding_mask, is_causal):
        y = memory

        y = y + self.self_block(self.norm1(y), mask, src_key_padding_mask, is_causal)

        y = y + self.ff_block(self.norm2(y))

        return y

    def mlp(self, x:  Tensor):
        x = self.ff_linear(x)
        x = F.gelu(x)
        return self.ff_linear2(x)

    def self_block(self, y, mask, src_key_padding_mask, is_causal):

        y,  _ = self.self_attn(y, key_padding_mask=src_key_padding_mask,
                               need_weights=True, attn_mask=mask, is_causal=is_causal)

        return self.dropout1(y)

    def ff_block(self, y: Tensor):
        return self.dropout2(self.ffn(y))


class MORTMDecoder(nn.Module):
    def __init__(self, args: MORTMArgs, batch_first, bias, layer_norm_eps, progress):
        super(MORTMDecoder, self).__init__()
        self.num_layer = args.d_layer
        self.layers = _get_clones(MORTMDecoderLayer(args,
                                                    batch_first=batch_first, bias=bias,
                                                    layer_norm_eps=layer_norm_eps, progress=progress), self.num_layer)
        self.norm = LayerNorm(args.d_model, eps=1e-5, bias=True, dtype=torch.float32)
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

    def __init__(self, args: MORTMArgs, batch_first, bias, layer_norm_eps, progress):
        super(MORTMDecoderLayer, self).__init__()
        self.n_head = args.num_heads
        self.d_model = args.d_model
        self.cross_attention: FlashCrossAttentionM = FlashCrossAttentionM(args.d_model, args.num_heads, args.dropout)
        self.self_attention: FlashSelfAttentionM =FlashSelfAttentionM(args.d_model, args.num_heads, args.dropout, progress=progress)

        #self.ffn = FFN(d_model, dim_ff, dropout)
        if args.use_moe_decoder == True:
            self.ffn = MoE(args.d_model, args.dim_feedforward,
                           args.num_experts, args.topk_experts, args.num_groups, args.topk_groups, )
        else:
            self.ffn = FFN(args.d_model, args.dim_feedforward, args.dropout)


        self.norm1 = LayerNorm(args.d_model, eps=layer_norm_eps, bias=bias, dtype=torch.float32)
        self.norm2 = LayerNorm(args.d_model, eps=layer_norm_eps, bias=bias, dtype=torch.float32)
        self.norm3 = LayerNorm(args.d_model, eps=layer_norm_eps, bias=bias, dtype=torch.float32)

        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)

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

        if memory is not None:
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