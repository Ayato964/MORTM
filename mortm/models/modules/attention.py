from typing import Optional

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import *
from typing import Optional, Tuple

from torch.nn.functional import linear, softmax, dropout

import torch
import torch.nn as nn
import math
from einops import rearrange

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


def print_stats(name, tensor):
    print(f"{name}: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}")


def zero_pad_kv(K: torch.Tensor, V: torch.Tensor, pad_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    K, V : torch.Tensor
        キーおよびバリューのテンソル。形状は (seq_len, batch_size, hidden_dim) と仮定する。
    pad_mask : torch.Tensor
        パディングマスク。形状は (batch_size, seq_len) または (seq_len, batch_size) の Boolean テンソルとする。
        True: 有効トークン、False: パディングトークン

    Returns
    -------
    (K_zeroed, V_zeroed) : tuple of torch.Tensor
        パディング位置の値を0に置き換えた K と V。
    """
    if pad_mask is None:
        return K, V
    # もし pad_mask の shape が (batch_size, seq_len) となっている場合、(seq_len, batch_size) に転置する
    if pad_mask.shape[0] == K.shape[1]:
        pad_mask = pad_mask.transpose(0, 1)  # 転置して (seq_len, batch_size) にする

    # ここで pad_mask の shape は (seq_len, batch_size) になっている前提
    # 最後の次元に unsqueeze して (seq_len, batch_size, 1) にする（hidden_dim 軸へのブロードキャスト用）
    mask = pad_mask.to(K.dtype).unsqueeze(-1)

    # マスクが True の位置は 1, False の位置は 0 となるので、それを乗じるとパディング部分は全て 0 になる
    K_zeroed = K * mask
    V_zeroed = V * mask

    return K_zeroed, V_zeroed


def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    return True


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
            for x in torch_dispatch_mode_stack
        )
    else:
        return False


class MultiHeadAttentionRPR(Module):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/activation.html#MultiheadAttention

    Modification to add RPR embedding Er and call custom multi_head_attention_forward_rpr
    ----------
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, er_len=None):
        super(MultiHeadAttentionRPR, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = False
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Adding RPR embedding matrix
        if (er_len is not None):
            self.Er = Parameter(torch.rand((er_len, self.head_dim), dtype=torch.float32))
        else:
            self.Er = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):

        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            # return F.multi_head_attention_forward(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask, use_separate_proj_weight=True,
            #     q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            #     v_proj_weight=self.v_proj_weight)

            return multi_head_attention_forward_rpr(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, rpr_mat=self.Er)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            # return F.multi_head_attention_forward(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask)

            return multi_head_attention_forward_rpr(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, rpr_mat=self.Er)


# multi_head_attention_forward_rpr
def multi_head_attention_forward_rpr(query,  # type: Tensor
                                     key,  # type: Tensor
                                     value,  # type: Tensor
                                     embed_dim_to_check,  # type: int
                                     num_heads,  # type: int
                                     in_proj_weight,  # type: Tensor
                                     in_proj_bias,  # type: Tensor
                                     bias_k,  # type: Optional[Tensor]
                                     bias_v,  # type: Optional[Tensor]
                                     add_zero_attn,  # type: bool
                                     dropout_p,  # type: float
                                     out_proj_weight,  # type: Tensor
                                     out_proj_bias,  # type: Tensor
                                     training=True,  # type: bool
                                     key_padding_mask=None,  # type: Optional[Tensor]
                                     need_weights=True,  # type: bool
                                     attn_mask=None,  # type: Optional[Tensor]
                                     use_separate_proj_weight=False,  # type: bool
                                     q_proj_weight=None,  # type: Optional[Tensor]
                                     k_proj_weight=None,  # type: Optional[Tensor]
                                     v_proj_weight=None,  # type: Optional[Tensor]
                                     static_k=None,  # type: Optional[Tensor]
                                     static_v=None,  # type: Optional[Tensor]
                                     rpr_mat=None
                                     ):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/functional.html

    Modification to take RPR embedding matrix and perform skew optimized RPR (https://arxiv.org/abs/1809.04281)
    ----------
    type: (...) -> Tuple[Tensor, Optional[Tensor]]
    """



    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    ######### ADDITION OF RPR ###########
    if (rpr_mat is not None):
        rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
        qe = torch.einsum("hld,md->hlm", q, rpr_mat)
        srel = _skew(qe)

        attn_output_weights += srel

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        key_padding_mask = ~key_padding_mask.bool()

        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        key_padding_mask = key_padding_mask.type(dtype=torch.float32)
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)

    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def _get_valid_embedding(Er, len_q, len_k):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Gets valid embeddings based on max length of RPR attention
    ----------
    """

    len_e = Er.shape[0]
    start = max(0, len_e - len_q)
    return Er[start:, :]


def _skew(qe):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Performs the skew optimized RPR computation (https://arxiv.org/abs/1809.04281)
    ----------
    """

    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)

    qe = mask * qe
    qe = F.pad(qe, (1, 0, 0, 0, 0, 0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))

    srel = qe[:, 1:, :]
    return srel


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
    def __init__(self, d_model, d_q, d_kv, num_heads, drop_out):
        super(QKVLinear, self).__init__()
        self.num_heads = num_heads
        self.drop_out = nn.Dropout(drop_out)

        self.qkv_weight = Parameter(torch.empty(3 * d_model, d_model, dtype=torch.bfloat16)).to(dtype=torch.bfloat16)
        self.qkv_bias = Parameter(torch.empty(3 * d_model, dtype=torch.bfloat16)).to(dtype=torch.bfloat16)

        self.W_o = nn.Linear(d_model, d_model, dtype=torch.bfloat16)
        self.reset_()


    def reset_(self):
        #if not self.is_cross_attn:
        xavier_uniform_(self.qkv_weight)
        constant_(self.qkv_bias, 0)
        '''
        else:
            xavier_uniform_(self.q_weight)
            xavier_uniform_(self.kv_weight)
        
            constant_(self.q_bias, 0)
            constant_(self.kv_bias, 0)
        '''


    def forward(self, q: Tensor, k: Tensor, v: Tensor, memory_padding_mask: Tensor=None, key_padding_mask: Tensor=None):
        '''
        dkv = self.W_dkv(k)
        dq = self.W_dq(q)

        q = self.W_uq(dq)
        k = self.W_uk(dkv)
        v = self.W_uv(dkv)
        '''
        q, k, v = linear(q, self.qkv_weight, self.qkv_bias).chunk(3, dim=-1)

        if key_padding_mask is not None:
            q_unpad, indices, cu_seqlens, max_s, used_seqlens = unpad_input(q, key_padding_mask)
        else:
            q_unpad = q
            cu_seqlens = None
            max_s = None
            indices = None

        if key_padding_mask is not None:
            k_unpad, _, cu_seqlens_k, max_s_k, _ = unpad_input(k, memory_padding_mask if memory_padding_mask is not None else key_padding_mask)
            v_unpad, _, _, _, _ = unpad_input(v, memory_padding_mask if memory_padding_mask is not None else key_padding_mask)
        else:
            k_unpad = k
            v_unpad = v
            cu_seqlens_k, max_s_k = (None, None)

        #Q = self.W_q(q_unpad)
        #K = self.W_k(k_unpad)
        #V = self.W_v(v_unpad)
        Q = q_unpad
        K = k_unpad
        V = v_unpad

        if key_padding_mask is not None:
            Q = rearrange(Q, "total (h d) -> total h d", h=self.num_heads)
            K = rearrange(K, "total (h d) -> total h d", h=self.num_heads)
            V = rearrange(V, "total (h d) -> total h d", h=self.num_heads)
        else:
            Q = rearrange(Q, "b s (h d) -> b s h d", h=self.num_heads)
            K = rearrange(K, "b s (h d) -> b s h d", h=self.num_heads)
            V = rearrange(V, "b s (h d) -> b s h d", h=self.num_heads)

        return Q, K, V, cu_seqlens, max_s, indices, cu_seqlens_k, max_s_k

    def comp(self, o: Tensor):
        out: Tensor = self.W_o(o)

        return out.to(dtype=torch.float32)


class FlashSelfAttentionM(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, progress=None):
        super(FlashSelfAttentionM, self).__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None

        self.embed_dim = embed_dim
        self.qkv_block = QKVLinear(embed_dim, 256, 128, num_heads, dropout)
        self.drop = dropout

        self.alibi_slopes = torch.tensor(get_alibi_slopes(num_heads), dtype=torch.float32, device=progress.get_device())

    def forward(self, memory, key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):
        batch, tgt_len, embed_dim = memory.size()
        assert embed_dim == self.embed_dim
        assert list(memory.size()) == [batch, tgt_len, embed_dim]
        memory = memory.to(dtype=torch.bfloat16)
        q, k, v, cu_seqlens, max_s, indices, cu_seqlens_k, max_s_k = self.qkv_block(q=memory, k=memory, v=memory,
                                                                                    key_padding_mask=key_padding_mask)

        qkv_unpad = torch.stack([q, k, v], dim=1 if key_padding_mask is not None else 2)

        if key_padding_mask is not None:
            out = flash_attn_varlen_qkvpacked_func(qkv_unpad, dropout_p=self.drop, causal=is_causal,
                                                   cu_seqlens=cu_seqlens, max_seqlen=max_s,
                                                   alibi_slopes=self.alibi_slopes) # OK
        else:
            out = flash_attn_qkvpacked_func(qkv_unpad, causal=is_causal, dropout_p=0,
                                            alibi_slopes=self.alibi_slopes)

        if key_padding_mask is not None:
            out = rearrange(out, "total h d -> total (h d)")
            out = pad_input(out, indices, batch, tgt_len)

        else:
            out = rearrange(out, "b s h d -> b s (h d)")
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
