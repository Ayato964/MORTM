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
from .config import MORTMArgs, MORTM_LIVE_Args

gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = nn.GroupNorm(num_groups=out_channels // 8, num_channels=out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.silu(x)

class VisionEncoder(nn.Module):
    def __init__(self, args: MORTM_LIVE_Args, encoder_output_dim):
        super(VisionEncoder, self).__init__()

        # stride=(2, 2) -> 出力サイズ (64, 8)
        self.conv1 = CNNBlock(in_channels=args.instrument_num * 2, out_channels=args.instrument_num * 4, kernel_size=3, stride=(2, 2), padding=1, groups=1)

        # stride=(2, 2) -> 出力サイズ (32, 4)
        self.conv2 = CNNBlock(args.instrument_num * 4, args.instrument_num * 4, kernel_size=3, stride=(2, 2), padding=1, groups=1)

        # stride=(2, 1) -> 出力サイズ (16, 4)
        self.conv3 = CNNBlock(args.instrument_num * 4, args.instrument_num * 8, kernel_size=3, stride=(2, 1), padding=1, groups=1)

        self.fc_mu = nn.Linear(encoder_output_dim, args.d_model)
        self.fc_log_var = nn.Linear(encoder_output_dim, args.d_model)

        self.out_channel = args.instrument_num * 8
        self.width =  args.pianoroll_time_step // 4
        self.height = 16


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # 標準正規分布からノイズをサンプリング
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print(x.shape)
        h_flat = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(h_flat)
        log_var = self.fc_log_var(h_flat)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class VisionDecoder(nn.Module):
    def __init__(self, args: MORTM_LIVE_Args, encoder_output_dim:int,  encoder_output_shape = (64, 2, 32)):
        super().__init__()

        # エンコーダーの最終出力次元 (平坦化前)
        self.encoder_output_dim = encoder_output_dim
        self.encoder_output_shape = encoder_output_shape # (C, H, W)

        # 1. d_model次元の潜在変数zを、転置畳み込みできる形に復元する全結合層
        self.fc = nn.Linear(args.d_model, self.encoder_output_dim)

        # 2. 転置畳み込みで画像サイズを大きくしていく層 (エンコーダーの逆)
        # 入力: (64, 16, 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_output_shape[0], self.encoder_output_shape[0] // 2, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0), bias=False),
            nn.GroupNorm(self.encoder_output_shape[0] // 2 // 8, self.encoder_output_shape[0] // 2),
            nn.SiLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_output_shape[0] // 2, self.encoder_output_shape[0] // 2, kernel_size=3, stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.GroupNorm(self.encoder_output_shape[0] // 2 // 8, self.encoder_output_shape[0] // 2),
            nn.SiLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_output_shape[0] // 2, args.instrument_num * 2, kernel_size=3, stride=(2, 2), padding=1, output_padding=1, bias=False),
        )
        # 出力: (16, 128, 16) - 元のピアノロールサイズ

    def forward(self, z):
        # (N, d_model) -> (N, 4096)
        h = self.fc(z)
        h_reshaped = h.view(-1, *self.encoder_output_shape)
        #print(self.encoder_output_shape,  h_reshaped.shape)

        x_recon = self.deconv3(self.deconv2(self.deconv1(h_reshaped)))

        return x_recon

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



class MORTMDecoder(nn.Module):
    def __init__(self, args: MORTMArgs, progress):
        super(MORTMDecoder, self).__init__()
        self.num_layer = args.d_layer
        self.layers = _get_clones(MORTMDecoderLayer(args, progress=progress), self.num_layer)
        if args.normalize_type == "tanh":
            self.norm = NormTanh(args.d_model)
        elif args.normalize_type == "layernorm":
            self.norm = LayerNorm(args.d_model, eps=1e-5, bias=True)
        elif args.normalize_type == "rmsnorm":
            # 演算効率と警告解消のためdtypeを明示
            self.norm = nn.RMSNorm(args.d_model, eps=1e-5, dtype=torch.bfloat16)

    def forward(self,tgt: Tensor,tgt_is_causal: bool = False, cu_seqlens=None, max_seqlen=None, batch_size=None, indices=None, is_save_cache=False,
                encoder_x: Tensor = None, cu_seqlens_k=None, max_seqlen_k=None) -> Tensor:

        output = tgt
        for mod in self.layers:
            mod: MORTMDecoderLayer
            output = mod(
                output,
                tgt_is_causal=tgt_is_causal,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                batch_size=batch_size, indices=indices, is_save_cache=is_save_cache,
                encoder_x=encoder_x,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k
            )

        return self.norm(output)


class MORTMDecoderLayer(nn.Module):

    def __init__(self, args: MORTMArgs, progress):
        super(MORTMDecoderLayer, self).__init__()
        self.n_head = args.num_heads
        self.args = args
        self.d_model = args.d_model
        self.self_attention: FlashSelfAttentionM =FlashSelfAttentionM(args, progress=progress)

        if args.use_cross_attention:
            self.cross_attention: FlashCrossAttentionM =FlashCrossAttentionM(args, progress=progress)

        if args.use_moe_decoder:
            print("FFN TYPE: Gate Network")
            self.ffn = MoE(args)
        else:
            if args.use_silu:
                print("FFN TYPE: MLP with SiLU")
                self.ffn = Expert(args)
            else:
                print("FFN TYPE: Standard FFN")
                self.ffn = FFN(args.d_model, args.dim_feedforward, args.dropout)

        if args.normalize_type == "tanh":
            print("NORM TYPE: NormTanh")
            self.norm1 = NormTanh(args.d_model)
            if args.use_cross_attention:
                self.norm2 = NormTanh(args.d_model)
            self.norm3 = NormTanh(args.d_model)
        elif args.normalize_type == "layernorm":
            print("NORM TYPE: LayerNorm")
            self.norm1 = LayerNorm(args.d_model, eps=1e-5, bias=True)
            if args.use_cross_attention:
                self.norm2 = LayerNorm(args.d_model, eps=1e-5, bias=True)
            self.norm3 = LayerNorm(args.d_model, eps=1e-5, bias=True)
        elif args.normalize_type == "rmsnorm":
            print("NORM TYPE: RMSNorm")
            self.norm1 = nn.RMSNorm(args.d_model, eps=1e-5, dtype=torch.bfloat16)
            if args.use_cross_attention:
                self.norm2 = nn.RMSNorm(args.d_model, eps=1e-5, dtype=torch.bfloat16)
            self.norm3 = nn.RMSNorm(args.d_model, eps=1e-5, dtype=torch.bfloat16)

        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)

    def forward(self,tgt: Tensor,tgt_is_causal: bool = False, cu_seqlens=None, max_seqlen=None, batch_size=None, indices=None, is_save_cache=False,
                encoder_x: Tensor = None, cu_seqlens_k=None, max_seqlen_k=None)-> Tensor:
        y = tgt
        #print("ATTENTION START")
        y = y + self.self_block(self.norm1(y), tgt_is_causal, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, batch_size=batch_size, indices=indices, is_save_cache=is_save_cache) # 自己注意機構を適用

        if self.args.use_cross_attention:
            y = y + self.cross_block(self.norm2(y), encoder_x, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_k,
                                         max_seqlen_q=max_seqlen,
                                         max_seqlen_k=max_seqlen_k)
        #print("FFN START")
        y = y + self.ff_block(self.norm3(y)) # フィードフォワード層を適用
        #print("FFN END")
        return y

    def self_block(self, y: Tensor, is_causal: bool, cu_seqlens=None, max_seqlen=None, batch_size=None, indices=None, is_save_cache=False):
        y = self.self_attention(y, is_causal=is_causal, cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen, batch_size=batch_size, indices=indices, is_save_cache=is_save_cache)

        return self.dropout1(y)

    def cross_block(self,  x: Tensor, encoder_x: Tensor,cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                        max_seqlen_k=None):
        y = self.cross_attention(x, encoder_x,cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                    max_seqlen_q=max_seqlen_q,
                                    max_seqlen_k=max_seqlen_k)

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

    def __init__(self, args: MORTMArgs):
        super().__init__()
        if not args.use_ffn_lora:
            self.w1 = nn.Linear(args.d_model, args.dim_feedforward, bias=args.use_bias)
            self.w2 = nn.Linear(args.dim_feedforward, args.d_model, bias=args.use_bias)
            self.w3 = nn.Linear(args.d_model, args.dim_feedforward, bias=args.use_bias)
        else:
            self.w1 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w2 = lora.Linear(args.dim_feedforward, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w3 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):

    def __init__(self, args: MORTMArgs, route_scale=1, score_type="softmax"):
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
        self.dim = args.d_model
        self.topk = args.topk_experts
        self.n_groups = args.num_groups
        self.topk_groups = args.topk_groups
        self.score_func = score_type
        self.route_scale = route_scale
        #self.weight = nn.Parameter(torch.empty(num_experts, d_model))
        #self.bias = nn.Parameter(torch.empty(num_experts)) if self.dim == 7168 else None
        if args.use_gate_lora:
            self.gate_proj = lora.Linear(args.d_model, args.num_experts, r=args.lora_r, lora_alpha=args.lora_alpha, bias=False)
        else:
            self.gate_proj = nn.Linear(args.d_model, args.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x.device.type):
            scores = self.gate_proj(x)
            if self.score_func == "softmax":
                scores = scores.softmax(dim=-1)
            else:
                scores = scores.sigmoid()
            original_scores = scores

            if self.n_groups > 1:
                scores = scores.view(x.size(0), self.n_groups, -1)
                group_scores = scores.amax(dim=-1)
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
        if not args.use_ffn_lora:
            self.w1 = nn.Linear(args.d_model, args.dim_feedforward, bias=args.use_bias)
            self.w2 = nn.Linear(args.dim_feedforward, args.d_model, bias=args.use_bias)
            self.w3 = nn.Linear(args.d_model, args.dim_feedforward, bias=args.use_bias)
        else:
            self.w1 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w2 = lora.Linear(args.dim_feedforward, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w3 = lora.Linear(args.d_model, args.dim_feedforward, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: MORTMArgs, route_scale=1):
        super().__init__()
        self.dim = args.d_model
        self.n_routed_experts = args.num_experts
        
        # モジュールレベルではなく、初期化時に動的に取得することでインポート順の影響を排除
        curr_world_size = dist.get_world_size() if dist.is_initialized() else 1
        curr_rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.n_local_experts = self.n_routed_experts // curr_world_size
        self.n_activated_experts = args.topk_experts
        self.experts_start_idx = curr_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args, route_scale=route_scale)

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
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)
        return (y + z)


class NormTanh(nn.Module):
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