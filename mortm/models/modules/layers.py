import json
from typing import Optional, Literal

import numpy
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, LayerNorm, MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, _generate_square_subsequent_mask
from einops import rearrange
import loralib.layers as lora
from typing import Tuple, List
import numpy as np

from .attention import FlashSelfAttentionM, FlashCrossAttentionM, linear
from .config import MORTMArgs


world_size = 1
rank = 0
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

class Pool(nn.Module):
    """Attention Poolingによるシーケンス集約モジュール"""
    def __init__(self, args: MORTMArgs):
        super().__init__()
        self.attention_scorer = nn.Linear(args.d_model, 1)

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        attention_scores = self.attention_scorer(x)

        batch_size = len(cu_seqlens) - 1
        output_vectors = []
        for i in range(batch_size):
            start_idx, end_idx = cu_seqlens[i], cu_seqlens[i+1]
            if start_idx == end_idx: continue

            seq_x = x[start_idx:end_idx]
            seq_scores = attention_scores[start_idx:end_idx]
            attention_weights = torch.softmax(seq_scores, dim=0)
            context_vector = torch.sum(seq_x * attention_weights, dim=0)
            output_vectors.append(context_vector)

        return torch.stack(output_vectors) # shape: [B, d_model]


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
    def __init__(self, args: MORTMArgs, progress):
        super(MORTMDecoder, self).__init__()
        self.num_layer = args.d_layer
        self.layers = _get_clones(MORTMDecoderLayer(args, progress=progress), self.num_layer)
        self.norm = DynamicTanh(args.d_model)

    def forward(self, tgt: Tensor, tgt_is_causal: Optional[bool] = None,
                cu_seqlens=None, max_seqlen=None, batch_size=None, indices=None, is_save_cache=False) -> Tensor:

        output = tgt
        for mod in self.layers:
            mod: MORTMDecoderLayer
            output = mod(
                output,
                tgt_is_causal=tgt_is_causal,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                batch_size=batch_size, indices=indices, is_save_cache=is_save_cache
            )

        return self.norm(output)


class MORTMDecoderLayer(nn.Module):

    def __init__(self, args: MORTMArgs, progress):
        super(MORTMDecoderLayer, self).__init__()
        self.n_head = args.num_heads
        self.d_model = args.d_model
        self.self_attention: FlashSelfAttentionM =FlashSelfAttentionM(args, progress=progress)

        if args.use_moe_decoder == True:
            self.ffn = MoE(args)
        else:
            self.ffn = FFN(args.d_model, args.dim_feedforward, args.dropout)


        self.norm1 = DynamicTanh(args.d_model)
        self.norm2 = DynamicTanh(args.d_model)
        self.norm3 = DynamicTanh(args.d_model)

        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)

    def forward(self,tgt: Tensor,tgt_is_causal: bool = False, cu_seqlens=None, max_seqlen=None, batch_size=None, indices=None, is_save_cache=False)-> Tensor:

        y = tgt
        y = y + self.self_block(self.norm1(y), tgt_is_causal, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, batch_size=batch_size, indices=indices, is_save_cache=is_save_cache) # 自己注意機構を適用

        y = y + self.ff_block(self.norm3(y)) # フィードフォワード層を適用

        return y

    def self_block(self, y: Tensor, is_causal: bool, cu_seqlens=None, max_seqlen=None, batch_size=None, indices=None, is_save_cache=False):
        y = self.self_attention(y, is_causal=is_causal, cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen, batch_size=batch_size, indices=indices, is_save_cache=is_save_cache)

        return self.dropout1(y)

    def ff_block(self, y: Tensor):

        return self.dropout3(self.ffn(y))


class SelectiveAttentionDecoder(nn.Module):
    def __init__(self, args: MORTMArgs):
        super().__init__()
        self.layers = _get_clones(SelectiveAttentionDecoderLayer(args), args.d_layer)
        self.norm = DynamicTanh(args.d_model)

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SelectiveAttentionDecoderLayer(nn.Module):
    def __init__(self, args: MORTMArgs):
        super().__init__()
        self.expand = 2
        self.d_inner = int(args.d_model * self.expand)

        self.in_proj = nn.Module(args.d_model, args.d_model * 2)
        self.conv1d = nn.Conv1d(args.d_model, args.d_model, kernel_size=4)

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)

        A = rearrange(torch.arange(1, self.d_state + 1), "n -> 1 n")
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_block = MoE(args)

        self.dropout = nn.Dropout(args.dropout)
        self.norm = DynamicTanh(args.d_model)

    def forward(self, x: Tensor):
        ss_out = self.ssm_block(self.norm(x))

        y = x + self.dropout(self.out_block(ss_out))

        return y

    def ssm_block(self, x: Tensor):
        x_and_res: Tensor = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, "b l d -> b d l")
        x_conv = self.conv1d(x)[:, :, :x.shape[-1]]
        x_conv = rearrange(x_conv, "b d l -> b l d")

        x = F.silu(x_conv)
        y = self.ssm(x)

        res = F.silu(res)
        gated_output = y * res

        return gated_output

    def ssm(self, x):
        # A, D は学習可能なパラメータ
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # Δ, B, C は入力xに応じて動的に決まる
        x_doubled = self.x_proj(x)
        delta, B = x_doubled.split(split_size=[self.d_state, self.d_state], dim=-1)
        C = B # 簡略化のためBとCを共有

        delta = F.softplus(self.dt_proj(delta))

        # ★★★ ここでCUDAカーネルを呼び出す ★★★
        #y = selective_scan_fn(
        #    x, delta, A, B, C, D=D, delta_softplus=True
        #)

        return delta

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

    def __init__(self, args: MORTMArgs):
        super().__init__()
        if not args.use_lora:
            self.w1 = nn.Linear(args.d_model, args.dim_feedforward)
            self.w2 = nn.Linear(args.dim_feedforward, args.d_model)
            self.w3 = nn.Linear(args.d_model, args.dim_feedforward)
        else:
            self.w1 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha)
            self.w2 = lora.Linear(args.dim_feedforward, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha)
            self.w3 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha)

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
        with torch.autocast(device_type=x.device.type):
            scores = linear(x, self.weight)
            if self.score_func == "softmax":
                scores = scores.softmax(dim=-1)
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
        return weights.type_as(x).to(dtype=x.dtype), indices


class Expert(nn.Module):

    def __init__(self, args: MORTMArgs):
        super().__init__()
        if not args.use_lora:
            self.w1 = nn.Linear(args.d_model, args.dim_feedforward)
            self.w2 = nn.Linear(args.dim_feedforward, args.d_model)
            self.w3 = nn.Linear(args.d_model, args.dim_feedforward)
        else:
            self.w1 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha)
            self.w2 = lora.Linear(args.dim_feedforward, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha)
            self.w3 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: MORTMArgs, route_scale=1):
        super().__init__()
        self.dim = args.d_model
        self.n_routed_experts = args.num_experts
        self.n_local_experts = self.n_routed_experts // world_size
        self.n_activated_experts = args.topk_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args.d_model, args.num_experts, args.topk_experts, args.num_groups, args.topk_groups, route_scale=route_scale)

        self.experts = nn.ModuleList([Expert(args) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
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
        return (y + z)


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        dtype = x.dtype
        x = torch.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        return x.to(dtype=dtype)