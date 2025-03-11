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

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, er_len=None):
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
        if(er_len is not None):
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
def multi_head_attention_forward_rpr(query,                       # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None,                   # type: Optional[Tensor]
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
    if(rpr_mat is not None):
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
        key_padding_mask = key_padding_mask.type(dtype=torch.float)
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
    qe = F.pad(qe, (1,0, 0,0, 0,0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))

    srel = qe[:, 1:, :]
    return srel


class QKVLinear(nn.Module):
    def __init__(self, d_model, num_heads, drop_out):
        super(QKVLinear, self).__init__()
        self.num_heads = num_heads
        self.drop_out = nn.Dropout(drop_out)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        if torch.all(k == v):
            raise "KeyとValueは常に同じ値になる必要があります。"

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        Q = rearrange(Q, "b s (h d) -> b s h d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b s h d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b s h d", h=self.num_heads)

        return Q, K, V

    def comp(self, o: Tensor):
        out = self.W_o(o)

        return self.drop_out(out)




class FlashSelfAttentionM(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(FlashSelfAttentionM, self).__init__()
        self.embed_dim = embed_dim
        self.qkv_block = QKVLinear(embed_dim, num_heads, dropout)
        self.drop = dropout

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):

        batch, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [batch, tgt_len, embed_dim]
        assert key.size() == value.size()

        q, k, v = self.qkv_block(q=query, k=key, v=value)

        if  query.dtype not in [torch.float16, torch.bfloat16]:
            q = q.half()
            k = k.half()
            v = v.half()


        qkv = torch.stack([q, k, v], dim=2)  # 形状: (batch, tgt_len, 3, H, D)

        #qkv = rearrange(qkv, "b s three h d -> b s (three h d)", three=3)  # 形状: (batch, tgt_len, 3*h*d)
        if key_padding_mask is not None:
            qkv_unpad, indices, cu_seqlens, max_s, used_seqlens = unpad_input(qkv, key_padding_mask)
        else:
            qkv_unpad = qkv
            cu_seqlens = None
            max_s = None
            indices = None

        out = self.fsa(qkv_unpad, causal=is_causal, cu_seqlens=cu_seqlens, max_seqlen=max_s)
        if key_padding_mask is not None:
            out_flat = rearrange(out, "nnz h d -> nnz (h d)")
            out = pad_input(out_flat, indices, batch, tgt_len)
        else:
            out = rearrange(out, "b s h d -> b s (h d)")

        out = self.qkv_block.comp(out)

        return out, None


class FlashCrossAttentionM(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, ):
        super(FlashCrossAttentionM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop = dropout
        self.qkv_block = QKVLinear(embed_dim, num_heads, dropout)

    def forward(self, query, key, value, memory_key_padding_mask=None, tgt_key_padding_mask=None,
                need_weights=True, attn_mask=None, is_causal=False):


        batch, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [ batch, tgt_len, embed_dim]
        assert key.size() == value.size()

        q, k, v = self.qkv_block(query, key, value)
        if  query.dtype not in [torch.float16, torch.bfloat16]:
            q = q.half()
            k = k.half()
            v = v.half()
        #print_stats("linear: Q", query)
        #print_stats("linear: K", key)
        #print_stats("linear: V", value)


        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        if tgt_key_padding_mask is not None:
            q_unpad, indices_q, cu_seqlens_q, max_s_q, used_seqlens_q = unpad_input(q, tgt_key_padding_mask)
        else:
            q_unpad = q
            cu_seqlens_q = None
            max_s_q = None
            indices_q = None

        kv = torch.stack([k, v], dim=2)  # 形状: (batch, tgt_len, 3, H, D)
        #kv = rearrange(kv, "b s three h d -> b s (three h d)", three=2)  # 形状: (batch, tgt_len, 3*h*d)
        if memory_key_padding_mask is not None:
            kv_unpad, indices, cu_seqlens, max_s, used_seqlens = unpad_input(kv, memory_key_padding_mask)
        else:
            kv_unpad = kv
            cu_seqlens = None
            max_s = None


        out = self.fca(q_unpad, kv_unpad, causal=is_causal,
                       cu_seqlens=cu_seqlens_q,
                       max_seqlen=max_s_q,
                       cu_seqlens_k=cu_seqlens,
                       max_seqlen_k=max_s)


        if tgt_key_padding_mask is not None:
            out_flat = rearrange(out, "nnz h d -> nnz (h d)")
            out = pad_input(out_flat, indices_q, batch, tgt_len)
        else:
            out = rearrange(out, "b s h d -> b s (h d)")

        out = self.qkv_block.comp(out)

        return out, None