from typing import Optional

from torch.nn.parameter import Parameter
from torch.nn.init import *
from typing import Optional, Tuple
import loralib.layers as lora

from torch.nn.functional import linear, softmax, dropout

import torch
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange

from .config import MORTMArgs
try:
    from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb
    IS_NOT_LINUX = True #本当はFlase
except ImportError as i:
    IS_NOT_LINUX = True
    print(f"モジュールをインストールできませんでした。（WindowsではFlashを利用できません）\n {i.name}")

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import *
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func, flash_attn_qkvpacked_func
except ImportError as i:
    print(f"モジュールをインストールできませんでした。\n {i.name}")

# FlashAttention2 の関数（flash_attn_func）をインポート
# （ライブラリがダウンロード済みであると仮定）



def marge_cache(kv_cache: Optional[Tuple[Tensor, Tensor]], cache_seqlens: Optional[Tensor],
                k: Tensor, v: Tensor) -> Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
    for i in range(k.shape[0]):
        pos = cache_seqlens[i] # シーケンス内の位置
        if pos >= kv_cache[0].shape[1]:
            kv_cache[0] = torch.cat([kv_cache[0],torch.zeros_like(kv_cache[0][:, :1])], dim=1)
            kv_cache[1] = torch.cat([kv_cache[1],torch.zeros_like(kv_cache[1][:, :1])], dim=1)

        kv_cache[0][i, pos, :, :] = k[i, 0]  # バッチi, スロットposに格納
        kv_cache[1][i, pos, :, :] = v[i, 0]
        cache_seqlens[i] += 1

    return kv_cache, cache_seqlens

def get_alibi_slopes(n_heads):
    """
    ALiBi のスロープを計算する関数。
    n_heads が 2 のべき乗の場合はシンプルな幾何級数になり、
    そうでない場合は補間してスロープを拡張します。
    """
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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # キャッシュ（sin/cosテーブル）の初期化
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 初回のキャッシュ構築
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

def rotate_half(x):
    """xの後半の符号を反転して前半と入れ替える（[-x2, x1]を作る操作）"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    q, k: [batch, seq_len, head_dim] or [total_tokens, head_dim]
    cos, sin: [max_seq, head_dim] -> position_idsに従って取得
    position_ids: [batch, seq_len] or [total_tokens]
    """
    cos = cos[position_ids].unsqueeze(-2) # [..., 1, head_dim] (head次元用)
    sin = sin[position_ids].unsqueeze(-2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QKVLinear(nn.Module):
    def __init__(self, args: MORTMArgs, use_cross_attention: bool=False):
        super(QKVLinear, self).__init__()
        self.num_heads = args.num_heads
        self.drop_out = nn.Dropout(args.dropout)
        self.use_cross_attention  = use_cross_attention

        if not use_cross_attention:
            if not  args.use_attn_lora:
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
            total, D = q.size()
            qkv = self.qkv_weight(q).view(total, 3, self.num_heads, D // self.num_heads)
            return qkv
        else:
            total_q, D_q = q.size()
            total_kv, D_kv = kv.size()

            q = self.q_weight(q).view(total_q, self.num_heads, D_q // self.num_heads)
            kv = self.kv_weight(kv).view(total_kv, 2, self.num_heads, D_kv // self.num_heads)
            return q, kv

    def comp(self, o: Tensor):
        out: Tensor = self.W_o(o)

        return out


class FlashSelfAttentionM(nn.Module):
    def __init__(self, args: MORTMArgs, progress=None):
        super(FlashSelfAttentionM, self).__init__()
        self.args = args
        self.head_dim = args.d_model // args.num_heads
        self.drop = args.dropout

        self.qkv_block = QKVLinear(args)

        # KV Cache の初期化
        self.kv_cache: Optional[Tuple[Tensor, Tensor]] = None
        self.cache_seqlens: Optional[Tensor] = None

        # Positional Embedding の選択
        if self.args.use_rope:
            print(f"FlashSelfAttentionM: Using Pure PyTorch RoPE (dim={self.head_dim})")
            device = progress.get_device() if progress else torch.device("cuda")
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=args.position_length, device=device)
            self.alibi_slopes = None
        else:
            print("FlashSelfAttentionM: Using ALiBi")
            device = progress.get_device() if progress else torch.device("cuda")
            self.alibi_slopes = torch.tensor(get_alibi_slopes(args.num_heads), dtype=torch.float32, device=device)
            self.rotary_emb = None

    def _init_kv_cache(self, batch_size, device, dtype):
        max_seq_len = self.args.position_length + 512 # マージンを持たせる
        shape = (batch_size, max_seq_len, self.args.num_heads, self.head_dim)

        self.kv_cache = (
            torch.zeros(shape, device=device, dtype=dtype), # 安全のためzeros推奨
            torch.zeros(shape, device=device, dtype=dtype)
        )
        self.cache_seqlens = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def forward(self, x: Tensor, is_causal=True, cu_seqlens=None, max_seqlen=None,
                batch_size=None, indices=None, is_save_cache=False):

        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # ==========================================
        # Phase 1: Prefill (Prompt Processing / Training)
        # ==========================================
        if cu_seqlens is not None:
            # キャッシュ初期化
            if is_save_cache and (self.kv_cache is None or self.kv_cache[0].shape[0] != batch_size):
                self._init_kv_cache(batch_size, x.device, x.dtype)

            # Linear projection
            qkv = self.qkv_block(q=x) # [total_tokens, 3 * d_model]

            # FlashAttention varlen packed 用に整形: [total_tokens, 3, num_heads, head_dim]
            total_tokens = x.shape[0]
            qkv = qkv.view(total_tokens, 3, self.args.num_heads, self.head_dim)

            # --- RoPE 適用 (ここが重要) ---
            if self.args.use_rope:

                q, k, v = qkv.unbind(1) # 各 [total_tokens, num_heads, head_dim]

                position_ids_list = []
                for i in range(len(cu_seqlens) - 1):
                    seq_len_i = cu_seqlens[i+1] - cu_seqlens[i]
                    position_ids_list.append(torch.arange(0, seq_len_i, device=x.device, dtype=torch.long))
                position_ids = torch.cat(position_ids_list) # [total_tokens]

                cos, sin = self.rotary_emb(v, seq_len=max_seqlen)
                q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                qkv = torch.stack([q_rot, k_rot, v], dim=1)

                out = flash_attn_varlen_qkvpacked_func(
                    qkv, dropout_p=self.drop, causal=is_causal,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                )

                if is_save_cache:
                    with torch.no_grad():
                        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
                        for i in range(batch_size):
                            start, end = cu_seqlens[i], cu_seqlens[i+1]
                            l = end - start
                            self.kv_cache[0][i, :l] = k_rot[start:end] # 回転済み
                            self.kv_cache[1][i, :l] = v[start:end]
                        self.cache_seqlens = seqlens

            # --- ALiBi の場合 ---
            else:
                out = flash_attn_varlen_qkvpacked_func(
                    qkv, dropout_p=self.drop, causal=is_causal,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    alibi_slopes=self.alibi_slopes
                )
                if is_save_cache:
                    with torch.no_grad():
                        _, k, v = qkv.unbind(1)
                        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
                        for i in range(batch_size):
                            start, end = cu_seqlens[i], cu_seqlens[i+1]
                            l = end - start
                            self.kv_cache[0][i, :l] = k[start:end]
                            self.kv_cache[1][i, :l] = v[start:end]
                        self.cache_seqlens = seqlens

        # ==========================================
        # Phase 2: Decoding (Token by Token)
        # ==========================================
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1) # [batch, 1, d_model]

            qkv = self.qkv_block(q=x) # [batch, 1, 3*d_model]
            qkv = qkv.view(x.shape[0], 1, 3, self.args.num_heads, self.head_dim)

            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

            if is_save_cache:
                if self.args.use_rope:
                    position_ids = self.cache_seqlens.clone() # [batch]

                    cos, sin = self.rotary_emb(v, seq_len=position_ids.max().item() + 1)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids.unsqueeze(1))

                    out = flash_attn_with_kvcache(
                        q,
                        self.kv_cache[0],
                        self.kv_cache[1],
                        k=k, # 回転済み
                        v=v,
                        cache_seqlens=self.cache_seqlens,
                        causal=True,
                        # rotary_... は渡さない！
                    )
                else:
                    # ALiBi
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

                # カウンタ更新
                self.cache_seqlens += 1

            else:
                pass

            out = out.squeeze(1) # [batch, heads, dim]

        # out: [total, heads, dim] or [batch, heads, dim]
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