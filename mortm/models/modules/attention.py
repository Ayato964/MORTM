from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import linear, softmax, dropout
from einops import rearrange

# lora, config, flash_attn 等のインポートは環境に合わせて維持
import loralib.layers as lora
from .config import MORTMArgs

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_with_kvcache
    )
except ImportError:
    print("FlashAttention not found. Ensure it is installed for CUDA acceleration.")

# ==========================================
# Helper Functions (RoPE / ALiBi)
# ==========================================

def get_alibi_slopes(n_heads):
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_alibi_slopes(2 * closest_power_of_2)[0::2]
        slopes.extend(extra[: n_heads - closest_power_of_2])
    return slopes

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos, sin: [seq_len, head_dim] or [1, head_dim] depending on usage
    # position_ids: [batch] or [total_tokens]

    # 形状を合わせるためのunsqueeze処理
    # cos: [..., 1, head_dim]
    cos = cos[position_ids].unsqueeze(-2)
    sin = sin[position_ids].unsqueeze(-2)

    # unsqueeze logic for broadcasting if necessary
    # q, k shape: [batch, seq, heads, head_dim] or [total, heads, head_dim]
    # cos shape after idx: [batch/total, 1, head_dim] -> Broadcast to heads

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_position_embeddings = seq_len
        t = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# ==========================================
# Modules
# ==========================================

class QKVLinear(nn.Module):
    def __init__(self, args: MORTMArgs, use_cross_attention: bool=False):
        super(QKVLinear, self).__init__()
        self.num_heads = args.num_heads
        self.head_dim = args.d_model // args.num_heads
        self.use_cross_attention = use_cross_attention

        if not use_cross_attention:
            if not args.use_attn_lora:
                self.qkv_weight = nn.Linear(args.d_model, 3 * args.d_model, bias=False, dtype=torch.bfloat16)
                self.W_o = nn.Linear(args.d_model, args.d_model, dtype=torch.bfloat16, bias=args.use_bias)
            else:
                self.qkv_weight = lora.Linear(args.d_model, 3 * args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=False, dtype=torch.bfloat16)
                self.W_o = lora.Linear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias, dtype=torch.bfloat16)
        else:
            self.q_weight = nn.Linear(args.d_model, args.d_model, bias=True, dtype=torch.bfloat16)
            self.kv_weight = nn.Linear(args.d_model, 2 * args.d_model, bias=True, dtype=torch.bfloat16)
            self.W_o = nn.Linear(args.d_model, args.d_model, dtype=torch.bfloat16)

    def forward(self, q: Tensor, kv: Tensor = None):
        if not self.use_cross_attention:
            # 【修正点】: q.size()の展開をやめ、動的にreshapeする
            qkv = self.qkv_weight(q)
            # qkv shape: [..., 3 * d_model]

            # Reshape to [..., 3, num_heads, head_dim]
            new_shape = qkv.shape[:-1] + (3, self.num_heads, self.head_dim)
            qkv = qkv.view(new_shape)
            return qkv
        else:
            # Cross Attention用の処理（既存ロジックをshape対応版に修正）
            q = self.q_weight(q)
            q_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
            q = q.view(q_shape)

            kv = self.kv_weight(kv)
            kv_shape = kv.shape[:-1] + (2, self.num_heads, self.head_dim)
            kv = kv.view(kv_shape)
            return q, kv

    def comp(self, o: Tensor):
        return self.W_o(o)


class FlashSelfAttentionM(nn.Module):
    def __init__(self, args: MORTMArgs, progress=None):
        super(FlashSelfAttentionM, self).__init__()
        self.args = args
        self.head_dim = args.d_model // args.num_heads
        self.drop = args.dropout

        self.qkv_block = QKVLinear(args)

        # KV Cache init
        self.kv_cache: Optional[Tuple[Tensor, Tensor]] = None
        self.cache_seqlens: Optional[Tensor] = None

        device = progress.get_device() if progress else torch.device("cuda")

        if self.args.use_rope:
            print(f"FlashSelfAttentionM: Using RoPE (dim={self.head_dim})")
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=args.position_length, device=device)
            self.alibi_slopes = None
        else:
            print("FlashSelfAttentionM: Using ALiBi")
            self.alibi_slopes = torch.tensor(get_alibi_slopes(args.num_heads), dtype=torch.float32, device=device)
            self.rotary_emb = None

    def _init_kv_cache(self, batch_size, device, dtype):
        max_seq_len = self.args.position_length + 512
        shape = (batch_size, max_seq_len, self.args.num_heads, self.head_dim)
        self.kv_cache = (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype)
        )
        self.cache_seqlens = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def forward(self, x: Tensor, is_causal=True, cu_seqlens=None, max_seqlen=None,
                batch_size=None, indices=None, is_save_cache=False):

        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # ==========================================
        # Phase 1: Prefill / Training (Parallel Processing)
        # ==========================================
        if cu_seqlens is not None:
            # Linear Projection & Reshape
            qkv = self.qkv_block(q=x)

            if self.args.use_rope:
                q, k, v = qkv.unbind(1) # [total_tokens, heads, dim]

                # Position IDs Gen
                position_ids_list = []
                for i in range(len(cu_seqlens) - 1):
                    seq_len_i = cu_seqlens[i+1] - cu_seqlens[i]
                    position_ids_list.append(torch.arange(0, seq_len_i, device=x.device, dtype=torch.long))
                position_ids = torch.cat(position_ids_list)

                # Apply RoPE
                cos, sin = self.rotary_emb(v, seq_len=max_seqlen)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                # Re-pack for FlashAttention
                qkv = torch.stack([q, k, v], dim=1)

                if is_save_cache:
                    if self.kv_cache is None or self.kv_cache[0].shape[0] != batch_size:
                        self._init_kv_cache(batch_size, x.device, x.dtype)

                    with torch.no_grad():
                        for i in range(batch_size):
                            start, end = cu_seqlens[i], cu_seqlens[i+1]
                            l = end - start
                            self.kv_cache[0][i, :l] = k[start:end]
                            self.kv_cache[1][i, :l] = v[start:end]
                            self.cache_seqlens[i] = l

                out = flash_attn_varlen_qkvpacked_func(
                    qkv, dropout_p=self.drop, causal=is_causal,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                )
                # out shape: [total_tokens, num_heads, head_dim]

            else:
                # ALiBi case
                out = flash_attn_varlen_qkvpacked_func(
                    qkv, dropout_p=self.drop, causal=is_causal,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    alibi_slopes=self.alibi_slopes
                )
                if is_save_cache:
                    if self.kv_cache is None or self.kv_cache[0].shape[0] != batch_size:
                        self._init_kv_cache(batch_size, x.device, x.dtype)

                    _, k, v = qkv.unbind(1)
                    with torch.no_grad():
                        for i in range(batch_size):
                            start, end = cu_seqlens[i], cu_seqlens[i+1]
                            l = end - start
                            self.kv_cache[0][i, :l] = k[start:end]
                            self.kv_cache[1][i, :l] = v[start:end]
                            self.cache_seqlens[i] = l

        # ==========================================
        # Phase 2: Decoding (Step-by-Step)
        # ==========================================
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1)

            qkv = self.qkv_block(q=x)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

            if is_save_cache:
                if self.args.use_rope:
                    position_ids = self.cache_seqlens.clone()
                    max_pos = position_ids.max().item()
                    cos, sin = self.rotary_emb(v, seq_len=max_pos + 1)

                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                    out = flash_attn_with_kvcache(
                        q, self.kv_cache[0], self.kv_cache[1],
                        k=k, v=v,
                        cache_seqlens=self.cache_seqlens,
                        causal=True
                    )
                else:
                    out = flash_attn_with_kvcache(
                        q, self.kv_cache[0], self.kv_cache[1],
                        k=k, v=v,
                        cache_seqlens=self.cache_seqlens,
                        alibi_slopes=self.alibi_slopes,
                        causal=True
                    )
                self.cache_seqlens += 1
            else:
                raise NotImplementedError("Decoding without cache not implemented.")

            # out shape: [batch, 1, num_heads, head_dim] (usually)

        # ==========================================
        # Unified Output Reshape
        # ==========================================
        # Prefill: [total, heads, dim] -> [total, heads*dim]
        # Decoding: [batch, seq, heads, dim] -> [batch, seq, heads*dim]
        # "..." が任意の次元（batch, seq, totalなど）を吸収し、
        # 最後の2次元 (heads, head_dim) を結合します。
        out = rearrange(out, "... h d -> ... (h d)")

        return self.qkv_block.comp(out)

class FlashCrossAttentionM(nn.Module):
    def __init__(self, args: MORTMArgs, progress=None):
        super(FlashCrossAttentionM, self).__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None
        self.args = args

        self.embed_dim = args.d_model
        self.qkv_block = QKVLinear(args, use_cross_attention=True)
        self.drop = args.dropout


    def forward(self, x: Tensor, encoder_x: Tensor,cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                    max_seqlen_k=None):
        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)
        if encoder_x.dtype == torch.float32:
            encoder_x = encoder_x.to(torch.bfloat16)

        # --- フェーズ1: 学習 または 推論のプロンプト処理 ---
        if cu_seqlens_q is not None:
            q, kv = self.qkv_block(q=x, kv=encoder_x)

            out = flash_attn_varlen_kvpacked_func(
                q=q,
                kv=kv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=False,
                dropout_p=self.drop
            )
        else:
            q, kv = self.qkv_block(q=x, kv=encoder_x)
            q = q.unsqueeze(0)
            kv = kv.unsqueeze(0)
            out = flash_attn_kvpacked_func(q=q, kv=kv, dropout_p=self.drop, causal=False)
            out = rearrange(out, "b s h d -> (b s) (h d)")
            return self.qkv_block.comp(out)

        # 最終的な出力層
        out = rearrange(out, "total h d -> total (h d)")
        return self.qkv_block.comp(out)