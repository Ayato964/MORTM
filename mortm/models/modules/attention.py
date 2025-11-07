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


class QKVLinear(nn.Module):
    def __init__(self, args: MORTMArgs, use_cross_attention: bool=False):
        super(QKVLinear, self).__init__()
        self.num_heads = args.num_heads
        self.drop_out = nn.Dropout(args.dropout)
        self.use_cross_attention  = use_cross_attention

        if not use_cross_attention:
            if not  args.use_lora:
                self.qkv_weight = nn.Linear(args.d_model, 3 * args.d_model, bias=True, dtype=torch.bfloat16)
                self.W_o = nn.Linear(args.d_model, args.d_model, dtype=torch.bfloat16)
            else:
                self.qkv_weight = lora.Linear(args.d_model, 3 * args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=True, dtype=torch.bfloat16)
                self.W_o = lora.Linear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=True, dtype=torch.bfloat16)
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
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None
        self.args = args

        self.embed_dim = args.d_model
        self.qkv_block = QKVLinear(args)
        self.drop = args.dropout
        self.kv_cache: Optional[Tuple[Tensor, Tensor]] = None
        self.cache_seqlens: Tensor = None

        if not self.args.use_rope:
            print("FlashAttention2のALiBiを使用します。")
            self.alibi_slopes = torch.tensor(get_alibi_slopes(args.num_heads), dtype=torch.float32, device=progress.get_device())
        else:
            print("FlashAttention2のRoPEを使用します。")
            head_dim = args.d_model // args.num_heads
            device = progress.get_device() if progress else None
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=10000.0, interleaved=False, device=device)

    def _init_kv_cache(self, batch_size, device, dtype):
        """最初の呼び出し時に、バッチサイズに合わせてキャッシュを初期化する"""
        max_seq_len = self.args.position_length + 100 # 設定ファイルなどから最大長を取得
        head_dim = self.args.d_model // self.args.num_heads
        shape = (batch_size, max_seq_len, self.args.num_heads, head_dim)

        # torch.emptyでメモリを確保するだけ。0で埋める必要はない
        self.kv_cache = (
            torch.empty(shape, device=device, dtype=dtype),
            torch.empty(shape, device=device, dtype=dtype)
        )
        self.cache_seqlens = torch.zeros(batch_size, device=device, dtype=torch.int32)

    def forward(self, x: Tensor, is_causal=False, cu_seqlens=None, max_seqlen=None,
                batch_size=None, indices=None, is_save_cache=False):
        if x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        # --- フェーズ1: 学習 または 推論のプロンプト処理 ---
        if cu_seqlens is not None:
            # プロンプト処理時にはキャッシュを初期化
            if is_save_cache and (self.kv_cache is None or self.kv_cache[0].shape[0] != batch_size):
                self._init_kv_cache(batch_size, x.device, x.dtype)

            qkv: Tensor = self.qkv_block(q=x)

            # RoPE/ALiBiの適用とアテンション計算 (この部分は元のロジックを維持)
            if not self.args.use_rope:
                out = flash_attn_varlen_qkvpacked_func(qkv, dropout_p=self.drop, causal=is_causal,
                                                       cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                                                       alibi_slopes=self.alibi_slopes)
            else:
                q, k, v = qkv.unbind(1)
                self.rotary_emb._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
                q = apply_rotary_emb(q, self.rotary_emb._cos_cached, self.rotary_emb._sin_cached, interleaved=False, cu_seqlens=cu_seqlens)
                k = apply_rotary_emb(k, self.rotary_emb._cos_cached, self.rotary_emb._sin_cached, interleaved=False, cu_seqlens=cu_seqlens)
                qkv_rotated = torch.stack([q, k, v], dim=1)
                out = flash_attn_varlen_qkvpacked_func(qkv_rotated, dropout_p=self.drop, causal=is_causal,
                                                       cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

            # is_save_cacheがTrueの場合、計算結果を事前確保したキャッシュに書き込む
            if is_save_cache:
                with torch.no_grad():
                    # RoPE適用済みのk,vをキャッシュするのが望ましい場合があるが、ここではqkvから取得
                    _, k_unpad, v_unpad = qkv.unbind(dim=1)

                    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)

                    # 各シーケンスのK,Vを、事前確保したキャッシュの先頭に書き込む
                    for i in range(batch_size):
                        start, end = cu_seqlens[i], cu_seqlens[i+1]
                        seq_len = end - start
                        self.kv_cache[0][i, :seq_len] = k_unpad[start:end]
                        self.kv_cache[1][i, :seq_len] = v_unpad[start:end]

                    self.cache_seqlens = seqlens

        # --- フェーズ2: 1トークンずつの推論 ---
        else:
            if is_save_cache:
                # このパスでは、xは (batch_size, d_model) の形状を想定
                qkv: Tensor = self.qkv_block(q=x)
                # (batch_size, 3, num_heads, head_dim) -> (3, batch_size, num_heads, head_dim)
                qkv = qkv.permute(1, 0, 2, 3)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # (batch_size, num_heads, head_dim) -> (batch_size, 1, num_heads, head_dim)
                # flash_attn_with_kvcache の入力形状に合わせる
                q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)

                # RoPE / ALiBi の引数を準備
                rotary_kwargs = {}
                if not IS_NOT_LINUX:
                    self.rotary_emb._update_cos_sin_cache(self.args.position_length, device=x.device, dtype=x.dtype)
                    rotary_kwargs = {
                        "rotary_cos": self.rotary_emb.cos_cached,
                        "rotary_sin": self.rotary_emb.sin_cached,
                        "rotary_interleaved": False
                    }

                # flash_attn_with_kvcache を呼び出すだけで、計算とキャッシュ更新が完了
                out = flash_attn_with_kvcache(
                    q,
                    k_cache=self.kv_cache[0],
                    v_cache=self.kv_cache[1],
                    k=k,
                    v=v,
                    cache_seqlens=self.cache_seqlens,
                    alibi_slopes=self.alibi_slopes if IS_NOT_LINUX else None,
                    causal=True,
                    **rotary_kwargs
                )

                # キャッシュの有効長をインクリメント
                self.cache_seqlens += 1

                # (batch_size, 1, h, d_model) -> (batch_size, h, d_model)
                out = out.squeeze(1)
            else:
                qkv = self.qkv_block(q=x)
                qkv = qkv.unsqueeze(0)
                out = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.drop, causal=is_causal)
                out = rearrange(out, "b s h d -> (b s) (h d)")
                return self.qkv_block.comp(out)

        # 最終的な出力層
        out = rearrange(out, "total h d -> total (h d)")
        return self.qkv_block.comp(out)

    def compute_cache_seqlens(self, k: torch.Tensor) -> torch.Tensor:
        """
        k: Tensor of shape [batch_size, max_seq_len, num_heads, head_dim]
        Returns:
            cache_seqlens: Tensor of shape [batch_size]  (実際のシーケンス長)
        """
        # 各タイムステップが "all-zero" かどうかを判定
        is_nonzero = k.abs().sum(dim=(-1, -2)) != 0  # shape: [batch_size, max_seq_len]

        # True/False → int に変換して累積和で長さを求める（ただし最初の False 位置でもOK）
        seqlens = is_nonzero.sum(dim=1)  # shape: [batch_size]

        return seqlens


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