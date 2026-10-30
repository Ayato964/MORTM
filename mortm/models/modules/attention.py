from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        flash_attn_varlen_qkvpacked_func as _cuda_flash_attn_varlen_qkvpacked_func,
        flash_attn_kvpacked_func as _cuda_flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func as _cuda_flash_attn_varlen_kvpacked_func,
        flash_attn_with_kvcache as _cuda_flash_attn_with_kvcache,
    )
    from flash_attn.bert_padding import (
        pad_input as _cuda_pad_input,
        unpad_input as _cuda_unpad_input,
    )
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


@dataclass(frozen=True)
class PackedVarlenLayout:
    """Immutable value object encapsulating varlen packed sequence layout without GPU->CPU syncs."""

    cu_seqlens: Tensor
    max_seqlen: int
    total_tokens: int

    def __post_init__(self) -> None:
        if self.max_seqlen < 0:
            raise ValueError(f"max_seqlen must be >= 0, got {self.max_seqlen}")
        if self.total_tokens < 0:
            raise ValueError(f"total_tokens must be >= 0, got {self.total_tokens}")

    @property
    def batch_size(self) -> int:
        return max(0, int(self.cu_seqlens.size(0)) - 1)

    @property
    def is_uniform(self) -> bool:
        return (
            self.total_tokens > 0
            and self.max_seqlen > 0
            and self.total_tokens == self.batch_size * self.max_seqlen
        )

    def seqlens(self, device: Optional[torch.device] = None) -> Tensor:
        cu = self.cu_seqlens if device is None else self.cu_seqlens.to(device=device)
        return (cu[1:] - cu[:-1]).to(dtype=torch.long)

    def position_ids(self, device: torch.device) -> Tensor:
        if self.total_tokens == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        if self.is_uniform:
            return torch.arange(self.max_seqlen, device=device, dtype=torch.long).repeat(self.batch_size)
        cu = self.cu_seqlens.to(device=device, dtype=torch.long)
        seqlens = cu[1:] - cu[:-1]
        starts = torch.repeat_interleave(
            cu[:-1],
            seqlens,
            output_size=self.total_tokens,
        )
        return torch.arange(self.total_tokens, device=device, dtype=torch.long) - starts

    def flat_padded_indices(self, device: torch.device) -> Tensor:
        if self.total_tokens == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        cu = self.cu_seqlens.to(device=device, dtype=torch.long)
        seqlens = cu[1:] - cu[:-1]
        seq_ids = torch.repeat_interleave(
            torch.arange(self.batch_size, device=device, dtype=torch.long),
            seqlens,
            output_size=self.total_tokens,
        )
        starts = torch.repeat_interleave(
            cu[:-1],
            seqlens,
            output_size=self.total_tokens,
        )
        pos_ids = torch.arange(self.total_tokens, device=device, dtype=torch.long) - starts
        return seq_ids * self.max_seqlen + pos_ids


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unpad_input(hidden_states: Tensor, attention_mask: Tensor):
    if _HAS_FLASH_ATTN and hidden_states.is_cuda:
        return _cuda_unpad_input(hidden_states, attention_mask)
    batch_size, seqlen = hidden_states.shape[0], hidden_states.shape[1]
    total_elems = batch_size * seqlen
    if total_elems == 0:
        cu_seqlens = torch.zeros(batch_size + 1, device=hidden_states.device, dtype=torch.int32)
        seqlens_in_batch = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.int32)
        indices = torch.empty((0,), device=hidden_states.device, dtype=torch.long)
        return hidden_states.reshape(0, *hidden_states.shape[2:]), indices, cu_seqlens, 0, seqlens_in_batch

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    min_max = torch.stack([seqlens_in_batch.min(), seqlens_in_batch.max()]).tolist()
    min_seqlen_in_batch, max_seqlen_in_batch = int(min_max[0]), int(min_max[1])

    if min_seqlen_in_batch == seqlen:
        indices = torch.arange(total_elems, device=hidden_states.device, dtype=torch.long)
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seqlen, step=seqlen, device=hidden_states.device, dtype=torch.int32
        )
        unpadded = hidden_states.reshape(total_elems, *hidden_states.shape[2:])
        return unpadded, indices, cu_seqlens, seqlen, seqlens_in_batch

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    unpadded = hidden_states.reshape(total_elems, *hidden_states.shape[2:])[indices]
    return unpadded, indices, cu_seqlens, max_seqlen_in_batch, seqlens_in_batch


def pad_input(hidden_states: Tensor, indices: Tensor, batch: int, seqlen: int) -> Tensor:
    if _HAS_FLASH_ATTN and hidden_states.is_cuda:
        return _cuda_pad_input(hidden_states, indices, batch, seqlen)
    total_elems = batch * seqlen
    if hidden_states.size(0) == total_elems and indices.numel() == total_elems:
        return hidden_states.reshape(batch, seqlen, *hidden_states.shape[1:])
    output = torch.zeros(
        total_elems,
        *hidden_states.shape[1:],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = output.index_put((indices,), hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def _build_alibi_bias(
    q_len: int,
    k_len: int,
    alibi_slopes: Tensor,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    q_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    rel_pos = -torch.abs(q_pos - k_pos).to(torch.float32)
    bias = alibi_slopes.to(device=device, dtype=torch.float32)[:, None, None] * rel_pos[None, :, :]
    if causal:
        causal_mask = k_pos > q_pos
        bias = bias.masked_fill(causal_mask[None, :, :], float("-inf"))
    return bias.unsqueeze(0).to(dtype)


class VarlenQkvAttentionStrategy(ABC):
    """Strategy interface for packed varlen QKV self-attention execution."""

    @abstractmethod
    def execute(
        self,
        qkv: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[Tensor] = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> Tensor: ...


class CudaFlashVarlenQkvStrategy(VarlenQkvAttentionStrategy):
    """Hardware-accelerated flash_attn CUDA C++ extension strategy."""

    def execute(
        self,
        qkv: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[Tensor] = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> Tensor:
        return _cuda_flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )


class BatchedSdpaVarlenQkvStrategy(VarlenQkvAttentionStrategy):
    """Zero-sync batched PyTorch 2.x SDPA (FlashAttention-2 / CuDNN) strategy."""

    _MAX_DENSE_EXPANSION_RATIO: int = 4

    def execute(
        self,
        qkv: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[Tensor] = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> Tensor:
        total_tokens = int(qkv.size(0))
        s_max = int(max_seqlen) if max_seqlen is not None else 0
        layout = PackedVarlenLayout(cu_seqlens=cu_seqlens, max_seqlen=s_max, total_tokens=total_tokens)
        n_heads, head_dim = int(qkv.size(2)), int(qkv.size(3))

        if total_tokens == 0 or s_max == 0 or layout.batch_size == 0:
            return torch.empty((0, n_heads, head_dim), device=qkv.device, dtype=qkv.dtype)

        # Fast Path 1: Uniform sequence lengths (zero padding, zero scatter/gather, 1 fused SDPA kernel)
        if layout.is_uniform and window_size == (-1, -1):
            bsz = layout.batch_size
            q, k, v = qkv.view(bsz, s_max, 3, n_heads, head_dim).unbind(dim=2)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            if alibi_slopes is not None:
                attn_mask = _build_alibi_bias(s_max, s_max, alibi_slopes, causal, qkv.device, qkv.dtype)
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale
                )
            else:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_t, k_t, v_t, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
                )
            return out.transpose(1, 2).reshape(total_tokens, n_heads, head_dim)

        # Fast Path 2: Variable-length causal batch without ALiBi (right-padded causal invariance -> 1 fused SDPA kernel)
        bsz = layout.batch_size
        dense_tokens = bsz * s_max
        if (
            causal
            and alibi_slopes is None
            and window_size == (-1, -1)
            and dense_tokens <= max(total_tokens * self._MAX_DENSE_EXPANSION_RATIO, 8192)
        ):
            flat_idx = layout.flat_padded_indices(qkv.device)
            padded_qkv = qkv.new_zeros((dense_tokens, 3, n_heads, head_dim)).index_put((flat_idx,), qkv)
            q, k, v = padded_qkv.view(bsz, s_max, 3, n_heads, head_dim).unbind(dim=2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                is_causal=True,
                scale=softmax_scale,
            )
            return out.transpose(1, 2).reshape(dense_tokens, n_heads, head_dim)[flat_idx]

        # Fallback Path: Extreme sparse varlen / non-causal varlen / ALiBi varlen
        cu = cu_seqlens.detach().cpu().tolist()
        outs = []
        for i in range(len(cu) - 1):
            s, e = int(cu[i]), int(cu[i + 1])
            if e <= s:
                continue
            q_i, k_i, v_i = qkv[s:e].unbind(dim=1)
            q_i = q_i.permute(1, 0, 2).unsqueeze(0)
            k_i = k_i.permute(1, 0, 2).unsqueeze(0)
            v_i = v_i.permute(1, 0, 2).unsqueeze(0)
            seq_l = e - s
            if alibi_slopes is not None:
                attn_mask = _build_alibi_bias(seq_l, seq_l, alibi_slopes, causal, qkv.device, qkv.dtype)
                o_i = torch.nn.functional.scaled_dot_product_attention(
                    q_i, k_i, v_i, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale
                )
            else:
                o_i = torch.nn.functional.scaled_dot_product_attention(
                    q_i, k_i, v_i, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
                )
            outs.append(o_i.squeeze(0).permute(1, 0, 2))
        if not outs:
            return torch.empty((0, n_heads, head_dim), device=qkv.device, dtype=qkv.dtype)
        return torch.cat(outs, dim=0)


_CUDA_FLASH_VARLEN_QKV_STRATEGY = CudaFlashVarlenQkvStrategy()
_BATCHED_SDPA_VARLEN_QKV_STRATEGY = BatchedSdpaVarlenQkvStrategy()


def flash_attn_varlen_qkvpacked_func(
    qkv: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tensor:
    strategy: VarlenQkvAttentionStrategy = (
        _CUDA_FLASH_VARLEN_QKV_STRATEGY
        if (_HAS_FLASH_ATTN and qkv.is_cuda)
        else _BATCHED_SDPA_VARLEN_QKV_STRATEGY
    )
    return strategy.execute(
        qkv=qkv,
        cu_seqlens=cu_seqlens,
        max_seqlen=int(max_seqlen) if max_seqlen is not None else 0,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_varlen_kvpacked_func(
    q: Tensor,
    kv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor:
    if _HAS_FLASH_ATTN and q.is_cuda:
        return _cuda_flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    outs = []
    for i in range(len(cu_q) - 1):
        sq, eq = int(cu_q[i]), int(cu_q[i + 1])
        sk, ek = int(cu_k[i]), int(cu_k[i + 1])
        if eq <= sq:
            continue
        if ek <= sk:
            outs.append(torch.zeros((eq - sq, q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype))
            continue
        q_i = q[sq:eq].permute(1, 0, 2).unsqueeze(0)
        k_i, v_i = kv[sk:ek].unbind(dim=1)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0)
        Lq, Lk = eq - sq, ek - sk
        if alibi_slopes is not None:
            attn_mask = _build_alibi_bias(Lq, Lk, alibi_slopes, causal, q.device, q.dtype)
            o_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale
            )
        else:
            o_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            )
        outs.append(o_i.squeeze(0).permute(1, 0, 2))
    if not outs:
        return torch.empty_like(q)
    return torch.cat(outs, dim=0)


def flash_attn_kvpacked_func(
    q: Tensor,
    kv: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor:
    if _HAS_FLASH_ATTN and q.is_cuda:
        return _cuda_flash_attn_kvpacked_func(
            q,
            kv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
    k, v = kv.unbind(dim=2)
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    if alibi_slopes is not None:
        attn_mask = _build_alibi_bias(q.shape[1], k.shape[1], alibi_slopes, causal, q.device, q.dtype)
        o_t = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale
        )
    else:
        o_t = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
    return o_t.permute(0, 2, 1, 3)


def flash_attn_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Optional[Tensor] = None,
    v: Optional[Tensor] = None,
    rotary_cos: Optional[Tensor] = None,
    rotary_sin: Optional[Tensor] = None,
    cache_seqlens: Optional[Tensor] = None,
    cache_batch_idx: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[Tensor] = None,
    num_splits: int = 0,
) -> Tensor:
    if _HAS_FLASH_ATTN and q.is_cuda:
        return _cuda_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=num_splits,
        )
    B, Sq, H, D = q.shape
    outs = []
    for i in range(B):
        pos = int(cache_seqlens[i].item()) if cache_seqlens is not None else 0
        if k is not None and v is not None:
            k_cache[i, pos : pos + Sq] = k[i]
            v_cache[i, pos : pos + Sq] = v[i]
            end_pos = pos + Sq
        else:
            end_pos = pos
        q_i = q[i].permute(1, 0, 2).unsqueeze(0)
        k_i = k_cache[i, :end_pos].permute(1, 0, 2).unsqueeze(0)
        v_i = v_cache[i, :end_pos].permute(1, 0, 2).unsqueeze(0)
        if alibi_slopes is not None:
            attn_mask = _build_alibi_bias(Sq, end_pos, alibi_slopes, causal and Sq > 1, q.device, q.dtype)
            o_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=softmax_scale
            )
        else:
            o_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, dropout_p=0.0, is_causal=(causal and Sq > 1), scale=softmax_scale
            )
        outs.append(o_i.squeeze(0).permute(1, 0, 2))
    return torch.stack(outs, dim=0)

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
    # position_ids に基づいて取得
    # cos, sin: [max_seq, dim] -> [Batch, Dim] (Decoding時) または [Total, Dim] (Prefill時)
    cos = cos[position_ids]
    sin = sin[position_ids]

    # 次元拡張（ここが修正箇所）
    # q との計算時に [Batch, Batch, ...] にならないよう、明示的に次元を合わせます
    if q.dim() == 4:
        # Decoding時: qは [Batch, 1, Heads, Dim]
        # cos/sinを [Batch, 1, 1, Dim] に変形
        # これで Batch次元同士が正しく対応し、Heads次元にブロードキャストされます
        cos = cos.unsqueeze(1).unsqueeze(1)
        sin = sin.unsqueeze(1).unsqueeze(1)
    else:
        # Prefill時: qは [Total, Heads, Dim]
        # cos/sinを [Total, 1, Dim] に変形
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # 型変換は行わず、そのまま計算 (bfloat16 なら bfloat16 のまま)
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
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        req_len = int(seq_len) if seq_len is not None else int(x.shape[-2])
        if req_len > self.max_position_embeddings or self.cos_cached.device != x.device:
            new_max = max(req_len, self.max_position_embeddings * 2)
            self._set_cos_sin_cache(new_max, x.device, x.dtype)
        return (
            self.cos_cached[:req_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:req_len].to(device=x.device, dtype=x.dtype),
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

        device = progress.get_device() if progress else _get_default_device()

        # RoPE / ALiBi の初期化
        if self.args.use_rope:
            # print(f"FlashSelfAttentionM: Using RoPE (dim={self.head_dim})")
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=args.position_length, device=device)
            self.alibi_slopes = None
        else:
            # print("FlashSelfAttentionM: Using ALiBi")
            self.register_buffer(
                "alibi_slopes",
                torch.tensor(get_alibi_slopes(args.num_heads), dtype=torch.float32, device=device),
                persistent=False,
            )
            self.rotary_emb = None

    def _init_kv_cache(self, batch_size, device, dtype, min_capacity: int = 0):
        max_seq_len = max(self.args.position_length + 512, int(min_capacity))
        shape = (batch_size, max_seq_len, self.args.num_heads, self.head_dim)
        self.kv_cache = (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype)
        )
        self.cache_seqlens = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def _ensure_kv_cache_capacity(
        self, batch_size: int, required_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self.kv_cache is None
            or self.cache_seqlens is None
            or self.kv_cache[0].shape[0] != batch_size
            or self.kv_cache[0].device != device
            or self.kv_cache[0].dtype != dtype
        ):
            self._init_kv_cache(batch_size, device, dtype, min_capacity=required_len)
            return

        cur_cap = int(self.kv_cache[0].shape[1])
        if required_len <= cur_cap:
            return

        new_cap = max(int(required_len), cur_cap * 2)
        new_shape = (batch_size, new_cap, self.args.num_heads, self.head_dim)
        new_k = torch.zeros(new_shape, device=device, dtype=dtype)
        new_v = torch.zeros(new_shape, device=device, dtype=dtype)
        new_k[:, :cur_cap].copy_(self.kv_cache[0])
        new_v[:, :cur_cap].copy_(self.kv_cache[1])
        self.kv_cache = (new_k, new_v)

    @torch.no_grad()
    def _write_prefill_kv_cache(
        self, k: Tensor, v: Tensor, layout: PackedVarlenLayout, batch_size: int
    ) -> None:
        eff_batch = int(batch_size) if batch_size is not None else layout.batch_size
        self._ensure_kv_cache_capacity(eff_batch, layout.max_seqlen, k.device, k.dtype)
        if layout.is_uniform and eff_batch == layout.batch_size:
            s_len = layout.max_seqlen
            self.kv_cache[0][:, :s_len] = k.view(eff_batch, s_len, self.args.num_heads, self.head_dim)
            self.kv_cache[1][:, :s_len] = v.view(eff_batch, s_len, self.args.num_heads, self.head_dim)
            self.cache_seqlens.fill_(s_len)
            return

        cu_list = layout.cu_seqlens.detach().cpu().tolist()
        for i in range(eff_batch):
            start, end = int(cu_list[i]), int(cu_list[i + 1])
            l = end - start
            if l > 0:
                self.kv_cache[0][i, :l] = k[start:end]
                self.kv_cache[1][i, :l] = v[start:end]
            self.cache_seqlens[i] = l

    def forward(self, x: Tensor, is_causal=True, cu_seqlens=None, max_seqlen=None,
                batch_size=None, indices=None, is_save_cache=False):

        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # ==========================================
        # Phase 1: Prefill / Training (Parallel)
        # ==========================================
        if cu_seqlens is not None:
            # [Total_Tokens, Dim] -> [Total_Tokens, 3, Heads, HeadDim]
            qkv = self.qkv_block(q=x)
            s_max = int(max_seqlen) if max_seqlen is not None else 0
            layout = PackedVarlenLayout(
                cu_seqlens=cu_seqlens,
                max_seqlen=s_max,
                total_tokens=int(x.size(0)),
            )

            if self.args.use_rope:
                q, k, v = qkv.unbind(1)

                # Zero-sync vectorized Position IDs Gen
                position_ids = layout.position_ids(x.device)

                # Apply RoPE (Safe float32)
                cos, sin = self.rotary_emb(v, seq_len=s_max)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                qkv = torch.stack([q, k, v], dim=1)

                # Cache Update (if needed)
                if is_save_cache:
                    self._write_prefill_kv_cache(k, v, layout, batch_size)

                out = flash_attn_varlen_qkvpacked_func(
                    qkv, dropout_p=(self.drop if self.training else 0.0), causal=is_causal,
                    cu_seqlens=cu_seqlens, max_seqlen=s_max
                )
            else:
                # ALiBi
                out = flash_attn_varlen_qkvpacked_func(
                    qkv, dropout_p=(self.drop if self.training else 0.0), causal=is_causal,
                    cu_seqlens=cu_seqlens, max_seqlen=s_max,
                    alibi_slopes=self.alibi_slopes
                )
                if is_save_cache:
                    _, k, v = qkv.unbind(1)
                    self._write_prefill_kv_cache(k, v, layout, batch_size)

            # Prefill Output: [Total, Heads, Dim] -> [Total, Heads*Dim]
            # (Prefill時はもともと2次元[Total, Dim]で返すのが正解)
            out = rearrange(out, "t h d -> t (h d)")

        # ==========================================
        # Phase 2: Decoding (Step-by-Step)
        # ==========================================
        else:
            # 1. 入力を [Batch, 1, Dim] に統一
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # 2. QKV計算 [Batch, 1, 3, Heads, Dim]
            qkv = self.qkv_block(q=x)

            # 3の次元で分割し、それぞれ [Batch, Heads, Dim] を取得
            q, k, v = qkv.unbind(2)


            # Seq次元 (=1) を復元して [Batch, 1, Heads, Dim] にする
            # ※ この unbind -> unsqueeze の過程でメモリが完全に独立・整列します
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()


            if is_save_cache:
                if self.cache_seqlens is None:
                    self._init_kv_cache(q.size(0), q.device, k.dtype, min_capacity=q.size(1))
                max_pos = int(self.cache_seqlens.max().item()) if self.cache_seqlens.numel() > 0 else 0
                self._ensure_kv_cache_capacity(q.size(0), max_pos + q.size(1), q.device, k.dtype)

                if self.args.use_rope:
                    position_ids = self.cache_seqlens.clone()
                    cos, sin = self.rotary_emb(v, seq_len=max_pos + q.size(1))

                    # RoPE適用 (q, k はここで新しく計算されるため安全)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                if getattr(self.args, 'debug_attention', False):
                    # ======================================================
                    # 観察用Attention: Flash Attentionをバイパスし、
                    # softmax(QK^T/√d)V を手動計算してweightを保存する
                    # ======================================================
                    B = q.shape[0]
                    cur_lens = self.cache_seqlens  # [B]

                    # 新しい k/v をキャッシュに書き込む
                    for i in range(B):
                        pos = cur_lens[i].item()
                        self.kv_cache[0][i, pos] = k[i, 0]   # k: [B,1,H,D]
                        self.kv_cache[1][i, pos] = v[i, 0]

                    self.cache_seqlens += 1
                    new_lens = self.cache_seqlens  # [B]

                    # 各バッチで attention を計算 (B=1 前提で最適化)
                    S = new_lens[0].item()
                    k_ctx = self.kv_cache[0][0, :S]   # [S, H, D]
                    v_ctx = self.kv_cache[1][0, :S]   # [S, H, D]

                    # q[0,0]: [H, D]  →  [H, 1, D]
                    q0 = q[0, 0].unsqueeze(1)          # [H, 1, D]
                    k0 = k_ctx.permute(1, 0, 2)        # [H, S, D]
                    v0 = v_ctx.permute(1, 0, 2)        # [H, S, D]

                    scale = self.head_dim ** -0.5
                    scores = torch.matmul(q0, k0.transpose(-2, -1)) * scale  # [H, 1, S]
                    attn_w = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [H, 1, S]

                    # weight を保存 [H, S] (head 次元を保持)
                    self.last_attn_weights = attn_w[:, 0, :].detach().cpu()

                    out_0 = torch.matmul(attn_w, v0)   # [H, 1, D]
                    out_0 = out_0.squeeze(1)            # [H, D]
                    out = out_0.unsqueeze(0)            # [1, H, D]  (batch=1)

                    # [B, H, D] -> [B, H*D]
                    out = rearrange(out, "b h d -> b (h d)")

                elif self.args.use_rope:
                    out = flash_attn_with_kvcache(
                        q,
                        self.kv_cache[0],
                        self.kv_cache[1],
                        k=k,
                        v=v,
                        cache_seqlens=self.cache_seqlens,
                        causal=True
                    )
                    self.cache_seqlens += 1
                    # Decoding Output: [Batch, 1, Heads, Dim] -> [Batch, Heads*Dim]
                    out = out.squeeze(1)
                    out = rearrange(out, "b h d -> b (h d)")
                else:
                    out = flash_attn_with_kvcache(
                        q,
                        self.kv_cache[0],
                        self.kv_cache[1],
                        k=k,
                        v=v,
                        cache_seqlens=self.cache_seqlens,
                        alibi_slopes=self.alibi_slopes,
                        causal=True
                    )
                    self.cache_seqlens += 1
                    out = out.squeeze(1)
                    out = rearrange(out, "b h d -> b (h d)")
            else:
                raise NotImplementedError("Decoding without cache is not implemented.")

        # 最終出力
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

            drop_p = self.drop if self.training else 0.0
            out = flash_attn_varlen_kvpacked_func(
                q=q,
                kv=kv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=False,
                dropout_p=(self.drop if self.training else 0.0)
            )
        else:
            q, kv = self.qkv_block(q=x, kv=encoder_x)
            q = q.unsqueeze(0)
            kv = kv.unsqueeze(0)
            out = flash_attn_kvpacked_func(q=q, kv=kv, dropout_p=(self.drop if self.training else 0.0), causal=False)
            out = rearrange(out, "b s h d -> (b s) (h d)")
            return self.qkv_block.comp(out)

        # 最終的な出力層
        out = rearrange(out, "total h d -> total (h d)")
        return self.qkv_block.comp(out)