from typing import Optional

from torch.nn.parameter import Parameter
from torch.nn.init import *
from typing import Optional, Tuple
import loralib.layers as lora

from torch.nn.functional import linear, softmax, dropout

import torch
import torch.nn as nn
import math
from einops import rearrange

from .config import MORTMArgs
try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import (flash_attn_varlen_qkvpacked_func,
                                                 flash_attn_qkvpacked_func,
                                                 flash_attn_varlen_kvpacked_func,
                                                 flash_attn_kvpacked_func)
    from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
except ImportError as i:
    print(f"モジュールをインストールできませんでした。\n {i.name}")

# FlashAttention2 の関数（flash_attn_func）をインポート
# （ライブラリがダウンロード済みであると仮定）
try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError("FlashAttention2 のライブラリが必要です。インストールしてください。")





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


class QKVLinear(nn.Module):
    def __init__(self, args: MORTMArgs):
        super(QKVLinear, self).__init__()
        self.num_heads = args.num_heads
        self.drop_out = nn.Dropout(args.dropout)

        if not  args.use_lora:
            self.qkv_weight = nn.Linear(args.d_model, 3 * args.d_model, bias=True, dtype=torch.bfloat16)
            self.W_o = nn.Linear(args.d_model, args.d_model, dtype=torch.bfloat16)
        else:
            self.qkv_weight = lora.Linear(args.d_model, 3 * args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=True, dtype=torch.bfloat16)
            self.W_o = lora.Linear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=True, dtype=torch.bfloat16)


    def forward(self, q: Tensor, k: Tensor=None, v: Tensor=None, ):
        total, D = q.size()
        qkv = self.qkv_weight(q).view(total, 3, self.num_heads, D // self.num_heads)

        return qkv

    def comp(self, o: Tensor):
        out: Tensor = self.W_o(o)

        return out.to(dtype=torch.float32)


class FlashSelfAttentionM(nn.Module):
    def __init__(self, args: MORTMArgs, progress=None):
        super(FlashSelfAttentionM, self).__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None

        self.embed_dim = args.d_model
        self.qkv_block = QKVLinear(args)
        self.drop = args.dropout

        self.alibi_slopes = torch.tensor(get_alibi_slopes(args.num_heads), dtype=torch.float32, device=progress.get_device())

    def forward(self, x, is_causal=False, cu_seqlens=None, max_seqlen=None):

        x = x.to(dtype=torch.bfloat16)
        qkv: Tensor = self.qkv_block(q=x)


        if cu_seqlens is not None:
            out = flash_attn_varlen_qkvpacked_func(qkv, dropout_p=self.drop, causal=is_causal,
                                                   cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                                                   alibi_slopes=self.alibi_slopes) # OK
        else:
            qkv = qkv.unsqueeze(0)
            out: Tensor = flash_attn_qkvpacked_func(qkv, causal=is_causal, dropout_p=0,
                                            alibi_slopes=self.alibi_slopes)
            out = out.squeeze(0)

        out = rearrange(out, "total h d -> total (h d)")
        out = self.qkv_block.comp(out)
        return out, None


class FlashCrossAttentionM(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(FlashCrossAttentionM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop = dropout
        self.qkv_block = QKVLinear(embed_dim, 256, 128, num_heads, dropout)

    def forward(self, tgt, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):
        batch, tgt_len, embed_dim = tgt.size()
        assert embed_dim == self.embed_dim
        assert list(tgt.size()) == [batch, tgt_len, embed_dim]
        tgt = tgt.to(dtype=torch.bfloat16)
        memory = memory.to(dtype=torch.bfloat16)

        q, k, v, cu_seqlens, max_s, indices, cu_seqlens_k, max_s_k = self.qkv_block(q=tgt, k=memory, v=memory,
                                                                                    key_padding_mask=tgt_key_padding_mask,
                                                                                    memory_padding_mask=memory_key_padding_mask)

        k_unpad = torch.stack([k, v], dim=1 if tgt_key_padding_mask is not None else 2)
        if tgt_key_padding_mask is not None:
            out = flash_attn_varlen_kvpacked_func(q, k_unpad, causal=is_causal, dropout_p=self.drop,
                                                  cu_seqlens_q=cu_seqlens,
                                                  max_seqlen_q=max_s,
                                                  cu_seqlens_k=cu_seqlens_k,
                                                  max_seqlen_k=max_s_k)
        else:
            out = flash_attn_kvpacked_func(q, k_unpad, causal=is_causal, dropout_p=0)

        if tgt_key_padding_mask is not None:
            out = rearrange(out, "total h d -> total (h d)")
            out: Tensor = pad_input(out, indices, batch, tgt_len)
        else:
            out: Tensor = rearrange(out, "b s h d -> b s (h d)")

        out = self.qkv_block.comp(out)
        return out, None
