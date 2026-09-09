import json
import math
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

from .attention import FlashSelfAttentionM, FlashCrossAttentionM, linear, flash_attn_varlen_kvpacked_func
from .config import MORTMArgs, MORTM5Args

gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


class ResNetBlock(nn.Module):
    """
    ConvNeXt スタイルの現代的 Inverted Residual (SoTA) ブロック
    - 7x7 Depthwise Conv による広い受容野（ピアノロールの時間・音高の文脈抽出）
    - GroupNorm (1グループ = LayerNormと同等) によるバッチサイズ非依存の安定した正規化
    - 1x1 Conv でチャンネルを expand_ratio 倍（既定: 4倍）に広げ、GELU活性化後に圧縮する Inverted Bottleneck 構造
    - クリーンな残差接続
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | Tuple[int, int] = 1,
        expand_ratio: int = 4,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 1. 空間特徴抽出 (Depthwise Conv: チャンネル独立で大受容野を効率的に計算)
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # 2. チャンネル特徴混合 (Inverted Bottleneck: 4倍に拡大して非線形活性化後、圧縮)
        hidden_dim = in_channels * expand_ratio
        self.pwconv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

        # 3. 残差パス (解像度やチャンネル数が変化する場合のみ 1x1 Conv で整合)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        return out + identity


class ResNetUpBlock(nn.Module):
    """
    ConvNeXt スタイルの現代的アップサンプリングブロック (SoTA)
    - 転置畳み込み特有のチェッカーボードアーティファクト（格子ノイズ）を防止するため、
      Nearest Upsample -> 7x7 Depthwise ConvNeXt ResNetBlock の構成を採用
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int | Tuple[int, int] = 1,
        expand_ratio: int = 4,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        if scale_factor != 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        else:
            self.upsample = nn.Identity()
        self.block = ResNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            expand_ratio=expand_ratio,
            kernel_size=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        return self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, args: MORTM5Args):
        super().__init__()
        self.track_size = args.track_size
        self.input_channel = self.track_size * 2
        self.resnet = nn.Sequential(
            ResNetBlock(self.input_channel, args.first_channel, stride=1),
            ResNetBlock(args.first_channel, args.first_channel, stride=2),
            ResNetBlock(args.first_channel, args.max_channel, stride=(2, 1)),
            ResNetBlock(args.max_channel, args.max_channel, stride=2)

        )

        self.init_h = 16
        self.init_w = 6
        flatten_dim = args.max_channel * self.init_h * self.init_w
        self.w_out = nn.Linear(flatten_dim, args.encoder_wout)
    
    def forward(self, x: Tensor) -> Tensor:
        
        if x.dim() == 5: #Shape: [Batch, Sequence, Track * 2, height, width]
            b, s = x.shape[0], x.shape[1]
            x = x.reshape(b * s, x.shape[2], x.shape[3], x.shape[4])
            x = self.resnet(x)
            x = x.reshape(x.shape[0], -1)    
            x = self.w_out(x)
            return x.reshape(b, s, -1)


        if x.dim() == 4: # Shape: [Batch, Track * 2, height, width]
            x = self.resnet(x)
            x = x.reshape(x.shape[0], -1)    
            x = self.w_out(x)
            return x


class ResNetDecoder(nn.Module):
    """
    ResNet (エンコーダ) と完全に対称・鏡像となる SoTA デコーダ
    - 潜在ベクトル [B, 2048] (または [B, S, 2048]) を元のピアノロール [B, 8, 128, 24] (または [B, S, 8, 128, 24]) に完全復元
    - 4段階の対称アップサンプリング: (16x6 -> 32x12 -> 64x12 -> 128x24)
    - チャンネル推移もエンコーダの完全逆順: (128 -> 128 -> 64 -> 64 -> 8)
    """
    def __init__(self, args: MORTM5Args):
        super().__init__()
        self.track_size = args.track_size
        self.output_channel = self.track_size * 2
        self.max_channel = args.max_channel
        self.first_channel = args.first_channel
        self.roll_h = args.roll_h
        self.roll_w = args.roll_w

        # 1. 特徴マップ形状への復元プロジェクション (2048 -> 128 * 16 * 6 = 12288)
        self.init_h = 16
        self.init_w = 6
        flatten_dim = self.max_channel * self.init_h * self.init_w
        self.w_in = nn.Linear(args.encoder_wout, flatten_dim)

        # 2. エンコーダと完全に対称な4段デコーダ
        # Encoder: (8->64, s=1) -> (64->64, s=2) -> (64->128, s=(2,1)) -> (128->128, s=2)
        # Decoder: (128->128, scale=2) -> (128->64, scale=(2,1)) -> (64->64, scale=2) -> (64->8, scale=1)
        self.decoder = nn.Sequential(
            ResNetUpBlock(self.max_channel, self.max_channel, scale_factor=2),           # (16, 6)   -> (32, 12), 128ch -> 128ch
            ResNetUpBlock(self.max_channel, self.first_channel, scale_factor=(2, 1)),   # (32, 12)  -> (64, 12), 128ch -> 64ch
            ResNetUpBlock(self.first_channel, self.first_channel, scale_factor=2),       # (64, 12)  -> (128, 24), 64ch -> 64ch
            ResNetUpBlock(self.first_channel, self.output_channel, scale_factor=1),      # (128, 24) -> (128, 24), 64ch -> 8ch
        )

    def forward(self, x: Tensor) -> Tensor:
        # 5次元の場合: 入力 [Batch, Sequence, 2048] -> 出力 [Batch, Sequence, Track*2, 128, 24]
        if x.dim() == 3:
            b, s = x.shape[0], x.shape[1]
            x = x.reshape(b * s, -1)
            x = self.w_in(x)
            x = x.reshape(b * s, self.max_channel, self.init_h, self.init_w)
            x = self.decoder(x)
            return x.reshape(b, s, self.output_channel, self.roll_h, self.roll_w)

        # 4次元の場合: 入力 [Batch, 2048] -> 出力 [Batch, Track*2, 128, 24]
        if x.dim() == 2:
            x = self.w_in(x)
            x = x.reshape(x.shape[0], self.max_channel, self.init_h, self.init_w)
            x = self.decoder(x)
            return x

        raise ValueError(f"Expected 2D [Batch, Dim] or 3D [Batch, Seq, Dim] tensor, got {x.dim()}D (shape: {x.shape})")

    

class AttentionPool(nn.Module):
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


class PMA(nn.Module):
    def __init__(self, d_model: int, out_dim: int, h_dim=64):
        super().__init__()
        assert out_dim % h_dim == 0
        self.num_heads = out_dim // h_dim
        self.head_dim = h_dim
        self.out_dim = out_dim
        self.d_model = d_model

        self.q_seed = nn.Parameter(torch.empty(1, out_dim))
        nn.init.normal_(self.q_seed, std=math.sqrt(1.0 / d_model))

        self.w_kv = nn.Linear(d_model, out_dim * 2)
        self.w_o = nn.Linear(out_dim, out_dim)

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        # x: (total_tokens, d_model), packed varlen
        batch = cu_seqlens.numel() - 1
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen_k = int(seqlens.max().item())

        kv = self.w_kv(x)
        kv = kv.view(-1, 2, self.num_heads, self.head_dim)

        # 1 seed query per sequence
        q = self.q_seed.expand(batch, -1).view(batch, self.num_heads, self.head_dim)

        # FlashAttention は fp16/bf16 のみ対応。q_seed(fp32 Parameter)は Linear を通らず
        # autocast の半精度化を受けないため、kv(autocast下でbf16)と型が食い違う。半精度に揃える。
        attn_dtype = kv.dtype if kv.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        kv = kv.to(attn_dtype)
        q = q.to(attn_dtype)

        cu_seqlens_q = torch.arange(
            batch + 1, device=x.device, dtype=torch.int32
        )  # each query length = 1

        out = flash_attn_varlen_kvpacked_func(
            q, kv,
            cu_seqlens_q,
            cu_seqlens.to(torch.int32),
            1,               # max_seqlen_q
            max_seqlen_k,    # max_seqlen_k
        )  # (total_q=batch, num_heads, head_dim)

        out = out.view(batch, self.out_dim)
        return self.w_o(out)

class MILPool(nn.Module):
    """LogSumExp による Multiple-Instance Learning 型プーリング(PMA の代替)。

    なぜ PMA では駄目だったか
    ------------------------------------------------------------------
    PMA は学習済み seed query 1 本による **加重平均**、つまりトークン特徴分布の
    1 次モーメントしか計算できない。ところが CARL の判別課題では:

      1. 効いている特徴が **ヒストグラムと標準偏差**(音価分布/発音位置分布/跳躍幅/
         グリッド整合率)であり、加重平均では表現できない。実測でも表層統計 + 小 MLP
         が 0.799 に対し PMA 判別器は 0.78 で同着、それ以上を一切取れなかった。
      2. 入力が **人間PAST ++ AI CONST ++ 人間FUTURE** を畳んだ最大 24 小節で、
         AI が書いたのは一部だけ。加重平均だとクラス差が AI トークン比率に比例して
         希釈される(自前ベースラインで gen-only 0.888 -> folded 0.799 と実測)。
      3. デコーダが因果マスクなので、PAST 内の位置は AI 区間の情報を構造的に
         持てないのに、ほぼ等重みで平均に入る(R17 時点でも attention logit の std は
         推定 0.18 しかなく、実質ほぼ一様平均だった)。

    LogSumExp/MIL が効く理由
    ------------------------------------------------------------------
    折り畳み系列は「bag に 1 つでも陽性インスタンスがあれば陽性」という MIL の
    問題設定そのもの。各トークンを instance として個別にスコアリングし、
    **soft-max 的に集約**する:

        s_t  = scorer(h_t)                          # トークンごとの AI らしさ
        bag  = (1/r) * log( (1/N) * sum_t exp(r * s_t) )

      r -> 0   : 平均プーリング(従来の PMA と同じ挙動)
      r -> inf : 最大プーリング(「どこか 1 箇所でも AI 的なら AI」)

    r は学習可能にしてあるので、データから平均寄り/最大寄りを選べる。1/N で割って
    いるので **系列長に不変**(入力は 1 小節から 1552 トークンまで幅がある)。

    副産物として s_t が「どの位置を AI 的と見たか」の可視化になる。PMA の
    attention 重みと違い、s_t は直接 bag スコアへの寄与を表すので解釈が素直。
    """

    def __init__(self, d_model: int, hidden: Optional[int] = None, r_init: float = 2.0):
        super().__init__()
        h = hidden or d_model
        self.scorer = nn.Sequential(
            nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, 1))
        # r は正である必要があるので log 空間で持つ
        self.log_r = nn.Parameter(torch.tensor(float(math.log(r_init))))

    def forward(self, x: Tensor, cu_seqlens: Tensor, return_scores: bool = False):
        """x: (total_tokens, d_model) packed varlen / cu_seqlens: (B+1,)

        Returns: (B,) の bag スコア。return_scores=True なら (bag, per-token スコア)。
        """
        s = self.scorer(x).squeeze(-1).float()          # (total_tokens,)
        r = self.log_r.exp().clamp(0.05, 20.0)
        cu = cu_seqlens.detach().to("cpu").tolist()      # 同期は 1 回だけにまとめる
        out = []
        for i in range(len(cu) - 1):
            lo, hi = int(cu[i]), int(cu[i + 1])
            if hi <= lo:
                out.append(s.new_zeros(()))
                continue
            seg = s[lo:hi]
            # logsumexp は内部で最大値を引くので、r*seg が大きくても安定
            out.append((torch.logsumexp(r * seg, dim=0)
                        - math.log(float(hi - lo))) / r)
        bag = torch.stack(out)
        return (bag, s) if return_scores else bag


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory



class MORTMDecoder(nn.Module):
    def __init__(self, args: MORTMArgs, progress):
        super(MORTMDecoder, self).__init__()
        self.num_layer = args.d_layer
        self.layers = nn.ModuleList([MORTMDecoderLayer(args, i, progress) for i in range(self.num_layer)])
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

    def __init__(self, args: MORTMArgs, layer_id, progress):
        super(MORTMDecoderLayer, self).__init__()
        self.n_head = args.num_heads
        self.args = args
        self.d_model = args.d_model
        self.layer_id = layer_id
        self.self_attention: FlashSelfAttentionM =FlashSelfAttentionM(args, progress=progress)

        if args.use_cross_attention:
            self.cross_attention: FlashCrossAttentionM =FlashCrossAttentionM(args, progress=progress)

        if args.use_moe_decoder:
            print("FFN TYPE: Gate Network")
            self.ffn = MoE(args, layer_id)
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


class Gate(nn.Module):
    """
    Sigmoid-gated MoE router with auxiliary-loss-free load balancing.

    制御則
    ------------------------------------------------------------------
        delta      = clamp((ema_counts - mu) / mu, -1, 1)      # 平均ゼロのベクトル
        gamma_eff  = clamp(cv / bias_gain_ref, max=1.0)        # スカラー
        bias       = -gamma_eff * delta

    ここで cv = std(ema_counts)/mu であり、delta の平均はゼロなので

        cv = ||delta||_2 / sqrt(N)

    が恒等的に成り立つ。したがって

        ||bias|| = ||delta||^2 / (bias_gain_ref * sqrt(N))

    つまりこれは線形P制御ではなく **二次制御** である。この指数が 1 であることが
    設計の中核で、3つの性質を同時に生む:

      1. 均衡近傍で介入が線形より速く消える（誤差が半分 -> bias は 1/4）
      2. 激しい不均衡には superlinear に立ち上がる
      3. 振動に対する負のフィードバック。境界ギャップが小さい層（僅差領域）は
         プラントゲインが極端に高く、固定ゲインだと過剰応答して振動し
         エキスパートを殺す。cv スケーリングは振動が始まると空間CVが下がるため
         ループゲインを自動的に下げ、振動を止める。

    指数を 0（固定gamma）にすると 3 が失われ、僅差領域で時間CVが 1.3 を超える
    リミットサイクルに入る（空間CVは低いままなので気付きにくい）。
    指数を 2 にすると介入が弱すぎて負荷が偏る。

    設計上の約束（変更する前に読むこと）
    ------------------------------------------------------------------
    1. routing_bias は top-k の「選択」にのみ加算する。出力の重み付けには
       素のスコアを使う。bias が weights に漏れると、負荷の都合で選ばれた
       エキスパートが負荷由来の信号で訓練されてしまう。

    2. routing_bias は self.training で切り替えない。学習中ずっと bias 込みで
       選択・学習してきたので、bias 込みの選択こそが学習済みの挙動である。
       止めるべきは「更新」であって「適用」ではない。

    3. 更新は copy_（上書き）で行う。ema_counts は alpha=1-decay で正規化された
       DCゲイン=1 の平均推定量であり積分器ではないため、これは P制御である。
       積分項は意図的に入れていない。定常誤差が残るが、それは
       「均衡 / ルーティング歪み」のトレードオフ上で自動的に止まる機構として働く。
       CV=0 は目的ではない（ランダムルーティングで達成できてしまう）。

    4. 出力値に影響する状態は buffer にする。更新タイミングだけを決める
       カウンタは Python int にする。buffer にすると毎 forward で
       GPU->CPU 同期が発生し、MoE層数 x forward 回数だけ遅くなる。

    checkpoint 移行
    ------------------------------------------------------------------
    buffer を新規追加しているので旧 checkpoint からは
    load_state_dict(..., strict=False) でロードする。
    routing_bias / ema_counts は名前が同じなので引き継がれる。
    """

    # model.to(torch.bfloat16) は buffer も含めてキャストする。bf16 は仮数8bit
    # なので counts が数千のオーダーになると EMA が数%狂い（実測 7.7%）、
    # bias に 0.015 程度の誤差が乗る。健全な定常状態の |bias| は 0.002 程度なので
    # これは信号がノイズに埋もれる水準。以下の buffer は fp32 を維持する。
    _FP32_BUFFERS = ("routing_bias", "ema_counts", "ema_counts_sq", "ema_counts_long",
                     "pending_counts", "diag_gap", "diag_distortion", "diag_aff_loss")

    def _apply(self, *args, **kwargs):
        mod = super()._apply(*args, **kwargs)
        for name in Gate._FP32_BUFFERS:
            buf = mod._buffers.get(name)
            if buf is not None and buf.is_floating_point() and buf.dtype is not torch.float32:
                mod._buffers[name] = buf.float()
        return mod

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # 旧 checkpoint が bf16 で保存されていた場合に備えて明示的に戻す
        for name in Gate._FP32_BUFFERS:
            k = prefix + name
            if k in state_dict and state_dict[k].is_floating_point():
                state_dict[k] = state_dict[k].float()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, args, layer_id: int, route_scale: float = 1.0):
        super().__init__()
        n_exp = args.num_experts

        self.topk = args.topk_experts
        self.n_exp = n_exp
        self.score_func = args.score_type
        self.layer_id = layer_id
        self.route_scale = route_scale

        # --- 制御パラメータ ---
        self.ema_decay = getattr(args, "ema_decay", 0.97)
        self.long_decay = getattr(args, "ema_decay_long", 0.999)   # tau ~ 1000
        self.manual_step = getattr(args, "gate_manual_step", False)
        self.bias_gain_ref = getattr(args, "bias_gain_ref", 1.2)
        self.update_every = getattr(args, "bias_update_every", 1)   # = grad_accum_steps

        # --- 監視・診断 ---
        self.cv_alert_threshold = getattr(args, "bias_cv_threshold", 0.70)
        self.dead_steps_threshold = getattr(args, "dead_steps_threshold", 16)
        self.dead_clear_patience = getattr(args, "dead_clear_patience", 16)
        self.log_every = getattr(args, "gate_log_every", 10)     # flush 何回ごとにログ判定
        self.diag_every = getattr(args, "gate_diag_every", 50)   # 0 で診断オフ

        if getattr(args, "use_gate_lora", False):
            import loralib as lora
            self.gate_proj = lora.Linear(
                args.d_model, n_exp,
                r=args.lora_r, lora_alpha=args.lora_alpha, bias=False,
            )
        else:
            self.gate_proj = nn.Linear(args.d_model, n_exp, bias=False)

        # ============ 出力値に影響する状態（buffer 必須）============
        self.register_buffer("routing_bias", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("ema_counts", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("ema_init", torch.tensor(False))
        self.register_buffer("bias_enabled", torch.tensor(bool(args.use_gate_bias)))

        # ============ 監視状態（学習再開で失わないため buffer）============
        # 部分累積窓は保存しない。_pending_steps（Python int）と永続性を揃えるため
        # persistent=False にする。片方だけ復元すると次の flush が約2倍の量になる。
        self.register_buffer("pending_counts", torch.zeros(n_exp, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("ema_counts_sq", torch.zeros(n_exp, dtype=torch.float32))
        # 長期EMA。データのドメイン相関を検出するための第2時定数
        self.register_buffer("ema_counts_long", torch.zeros(n_exp, dtype=torch.float32))
        self.register_buffer("zero_streak", torch.zeros(n_exp, dtype=torch.long))
        self.register_buffer("cv_alert", torch.tensor(False))
        self.register_buffer("dead_alert", torch.tensor(False))
        self.register_buffer("dead_clear_streak", torch.zeros((), dtype=torch.long))
        for k in ("diag_gap", "diag_distortion", "diag_aff_loss"):
            self.register_buffer(k, torch.tensor(float("nan")))

        # ============ 出力に影響しないカウンタ（同期回避のため int）============
        self._pending_steps = 0
        self._flush_count = 0
        self._last_diag_flush = -1

        if bool(self.bias_enabled):
            print(f"[MOE_{layer_id}] gate bias: on (quadratic, gain_ref={self.bias_gain_ref})")
        print(f"[MOE_{layer_id}] score func: {self.score_func}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # スコアは fp32 で計算・保持する。routing_bias も fp32。
        logits = self.gate_proj(x).float()
        scores = logits.softmax(dim=-1) if self.score_func == "softmax" else logits.sigmoid()

        # --- 選択: bias 込み。self.training で切り替えない ---
        routing_scores = scores + self.routing_bias if self.bias_enabled else scores
        _, indices = torch.topk(routing_scores, self.topk, dim=-1)

        # --- 重み: bias を含まない素のスコアから取る ---
        weights = scores.gather(dim=-1, index=indices)

        # sigmoid は各次元独立で top-k 和が [0, topk] の任意値をとるため正規化が必須。
        # これがないと「全スコアを一律に上げる」縮退方向が開き、ゲートが
        # エキスパート間の相対比較（競合）を学習しなくなる:
        #   正規化なし : dL/ds_i = <g, E_i>            ... 中心化されない
        #   正規化あり : dL/ds_i = (1/S)<g, E_i - y>   ... 混合出力 y を基準に中心化
        # softmax でも top-k 部分和は 1 未満なので同様に正規化する。
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = weights * self.route_scale

        if self.training:
            # 診断は flush 境界の直前 micro-batch で1回だけ走らせる。
            # manual_step=True のときは _pending_steps が増えないので、
            # 境界を予測できない。その場合は毎 micro-batch 走らせず
            # diag_every 回の flush ごとに最初の micro-batch で走らせる。
            if self.diag_every > 0 and self._flush_count % self.diag_every == 0:
                at_boundary = (self._pending_steps == 0 if self.manual_step
                               else self._pending_steps + 1 >= self.update_every)
                if at_boundary and self._flush_count != self._last_diag_flush:
                    self._last_diag_flush = self._flush_count
                    self._diagnose(scores, indices)
            self._observe(indices)

        return weights.to(dtype=x.dtype), indices

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _diagnose(self, scores: torch.Tensor, indices: torch.Tensor) -> None:
        """bias が「境界上の僅差の反転」に留まっているかを測る。

        bias_absmax / boundary_gap の比で判定する（routing_stats 参照）:
            < 0.1  親和性に介入していない
            ~ 1    境界上の僅差のみ反転（理想的な動作領域）
            > 3    親和性を上書きしている。bias 設定ではなくゲート自体を見るべき

        affinity_loss は「捨てた品質」を単位付きで直接測る唯一の指標。
        distortion（歪み率）だけではモード崩壊と僅差を区別できないが、
        1件あたりの損失 = affinity_loss / distortion を見れば区別できる。
        """
        k = self.topk
        if k >= self.n_exp:
            return

        top_val, top_idx = torch.topk(scores, k + 1, dim=-1)

        # 境界ギャップ: 第k位 - 第(k+1)位。bias なしの順位で測る
        gap = (top_val[..., k - 1] - top_val[..., k]).flatten().median()

        # 歪み率: bias によって top-k 集合が変わったトークンの割合
        ref = top_idx[..., :k]
        changed = (ref.sort(dim=-1).values != indices.sort(dim=-1).values).any(dim=-1)
        distortion = changed.to(torch.float32).mean()

        # 親和性損失: bias なしなら得られたスコア和との差
        aff_loss = (top_val[..., :k].sum(-1)
                    - scores.gather(-1, indices).sum(-1)).flatten().mean()

        self.diag_gap.copy_(gap)
        self.diag_distortion.copy_(distortion)
        self.diag_aff_loss.copy_(aff_loss)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _observe(self, indices: torch.Tensor) -> None:
        """負荷を集計し、update_every 回ごとに bias を更新する。

        bincount / EMA / bias 更新は全て GPU 上で完結し同期を起こさない。
        同期が入るのは _log の中だけで、そこは log_every で間引いている。
        """
        # bincount は同期しないので毎 micro-batch 実行して良い
        self.pending_counts += torch.bincount(
            indices.flatten(), minlength=self.n_exp
        ).to(self.pending_counts.dtype)

        # manual_step=True のとき flush は trainer が gate_step() で明示的に起こす。
        # activation checkpointing 下では forward が backward 中に再実行され
        # _observe が2回呼ばれるため、自動 flush では実効 tau が半減する。
        # counts の一律スケールは bias に影響しない（delta も cv もスケール不変）が
        # flush 頻度は影響を受けるので、確実にしたいなら manual_step を使う。
        if self.manual_step:
            return
        self._pending_steps += 1
        if self._pending_steps < self.update_every:
            return
        self._pending_steps = 0
        self.gate_step()

    @torch.no_grad()
    def gate_step(self) -> None:
        """累積した負荷を flush して bias を1回更新する。

        manual_step=True のとき、trainer が optimizer.step() の直後に
        1回だけ呼ぶ。activation checkpointing / 任意の grad_accum に対して
        「optimizer step 1回 = bias 更新1回」が厳密に保証される。

            for m in model.modules():
                if isinstance(m, Gate):
                    m.gate_step()
        """
        self._flush_count += 1
        counts = self.pending_counts.clone()
        self.pending_counts.zero_()

        # all_reduce は flush 時のみ。しないと rank ごとに違う bias ができモデルが分岐する
        if dist.is_initialized():
            dist.all_reduce(counts)

        # 測定の平滑化。alpha=1-decay により DCゲイン=1 の平均推定量になる（積分器ではない）。
        # 初回だけ counts をそのまま入れる。`if ema.sum() == 0` は同期を起こすので
        # torch.where で分岐する。
        d = self.ema_decay
        self.ema_counts.copy_(torch.where(
            self.ema_init, d * self.ema_counts + (1.0 - d) * counts, counts))
        self.ema_counts_sq.copy_(torch.where(
            self.ema_init,
            d * self.ema_counts_sq + (1.0 - d) * counts * counts,
            counts * counts))
        dl = self.long_decay
        self.ema_counts_long.copy_(torch.where(
            self.ema_init, dl * self.ema_counts_long + (1.0 - dl) * counts, counts))
        self.ema_init.fill_(True)

        mu = self.ema_counts.mean()
        cv = self.ema_counts.std(unbiased=False) / (mu + 1e-6)

        # 死亡連続回数（同期しない。bias は触らない）
        zero = counts == 0
        self.zero_streak[zero] += 1
        self.zero_streak[~zero] = 0

        if self._flush_count % self.log_every == 0:
            self._log(cv)

        if not self.bias_enabled:
            return

        # cv = ||delta||/sqrt(N) なのでこれは二次制御になる。
        # sigmoid / softmax で式は同一（sum/n_exp == mean）なので分岐しない。
        gamma_eff = torch.clamp(cv / self.bias_gain_ref, max=1.0)
        delta = ((self.ema_counts - mu) / (mu + 1e-6)).clamp(-1.0, 1.0)

        new_bias = -gamma_eff * delta
        new_bias -= new_bias.mean()   # clamp で崩れたゼロ和を戻す
        self.routing_bias.copy_(new_bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _log(self, cv: torch.Tensor) -> None:
        """アラートの立ち上がり／立ち下がりのみ出力。ここだけ同期が入る。

        死亡エキスパートは検出のみ。clip 範囲内の bias は「境界上のゆらぎの補正」
        しかできないため、スコアが崩壊しきったエキスパートは bias では救済できない。
        その場合は gate_proj.weight の該当行を健全な行からコピー＋ノイズで
        再初期化する別経路で扱う。
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        high = bool(cv > self.cv_alert_threshold)
        if high != bool(self.cv_alert):
            self.cv_alert.fill_(high)
            if rank == 0:
                tag = "HIGH" if high else "CLEAR"
                extra = f" | dist={self.ema_counts.long().tolist()}" if high else ""
                print(f"[MOE_MONITOR] L{self.layer_id} CV={float(cv):.4f} {tag}{extra}")

        dead = torch.nonzero(self.zero_streak >= self.dead_steps_threshold).flatten()
        if dead.numel() > 0:
            self.dead_clear_streak.zero_()
            if not bool(self.dead_alert):
                self.dead_alert.fill_(True)
                if rank == 0:
                    print(f"[MOE_DEAD] L{self.layer_id} {dead.numel()}/{self.n_exp} dead "
                          f"ids={dead.tolist()} CV={float(cv):.4f}")
        elif bool(self.dead_alert):
            self.dead_clear_streak += 1
            if int(self.dead_clear_streak) >= self.dead_clear_patience:
                self.dead_alert.fill_(False)
                self.dead_clear_streak.zero_()
                if rank == 0:
                    print(f"[MOE_DEAD] L{self.layer_id} CLEAR")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def finalize_bias(self) -> None:
        """学習終了時に呼ぶ。長期EMA から routing_bias を作り直す。

        通常の routing_bias は直近 tau=33 ステップの負荷から作られるため、
        学習が偏ったドメインのフェーズで終わると、そのフェーズ用の bias が
        固定されて出荷される。長期EMA（tau=1000）から作り直すことで
        学習全体を通した平均的な負荷補正になる。
        """
        if not bool(self.ema_init) or not bool(self.bias_enabled):
            return
        src = self.ema_counts_long
        mu = src.mean()
        cv = src.std(unbiased=False) / (mu + 1e-6)
        gamma_eff = torch.clamp(cv / self.bias_gain_ref, max=1.0)
        delta = ((src - mu) / (mu + 1e-6)).clamp(-1.0, 1.0)
        new_bias = -gamma_eff * delta
        new_bias -= new_bias.mean()
        self.routing_bias.copy_(new_bias)

    @torch.no_grad()
    def routing_stats(self) -> dict:
        """学習ループから任意に呼ぶ。この層が3領域のどこにいるかを判定する。

          (1) ばらばら   : cv 低 / gap 大  -> 健全。gamma_eff が自動で絞っている
          (2) モード崩壊 : cv 高 / gap 大  -> bias では直らない。ゲート自体を見る
          (3) 僅差       : cv 高 / gap 小  -> bias が最も安く効く領域

        bias_per_gap:
            < 0.1  介入していない（1）
            ~ 1    境界上の僅差のみ反転（3・理想）
            > 3    親和性を上書き（2・警告）

        temporal_cv はリミットサイクル検出用。cv が低いのに temporal_cv が
        高い（> 0.5 程度）場合、エキスパートが順番に飢餓状態になる振動が
        起きている。二次制御では通常起きないが、bias_gain_ref を
        下げすぎた場合（= ゲインを上げた場合）に現れる。

        loss_per_flip = affinity_loss / distortion は 1トークン反転あたりの
        品質コスト。(2) では (3) の約2倍になる。
        """
        mu = self.ema_counts.mean()
        var = (self.ema_counts_sq - self.ema_counts * self.ema_counts).clamp_min(0.0)
        babs = float(self.routing_bias.abs().max())
        gap = float(self.diag_gap)
        dist_ = float(self.diag_distortion)
        aff = float(self.diag_aff_loss)
        ok = lambda v: v == v  # not nan

        return {
            "layer": self.layer_id,
            "cv": float(self.ema_counts.std(unbiased=False) / (mu + 1e-6)),
            "temporal_cv": float((var.sqrt() / (self.ema_counts + 1e-6)).mean()),
            "bias_absmax": babs,
            "boundary_gap": gap,
            "bias_per_gap": babs / gap if ok(gap) and gap > 0 else float("nan"),
            "distortion": dist_,
            "affinity_loss": aff,
            "loss_per_flip": aff / dist_ if ok(dist_) and dist_ > 0 else float("nan"),
            "timescale_divergence": float((
                self.ema_counts / (self.ema_counts.mean() + 1e-6)
                - self.ema_counts_long / (self.ema_counts_long.mean() + 1e-6)
            ).abs().max()),
            "dead_now": int((self.ema_counts == 0).sum()),
            "load_min": float(self.ema_counts.min()),
            "load_max": float(self.ema_counts.max()),
        }

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

class SharedExpert(nn.Module):

    def __init__(self, args: MORTMArgs):
        super().__init__()
        if not args.use_ffn_lora:
            self.w1 = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
            self.w2 = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
            self.w3 = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
        else:
            self.w1 = lora.Linear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w2 = lora.Linear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)
            self.w3 = lora.Linear(args.d_model, args.d_model, r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class _DispatchGather(torch.autograd.Function):
    """x[token_ids] の gather。backward を決定的にするためだけに存在する。

    token_ids には同じトークンが k 回現れるので、既定の backward（index_add）は
    同じ行への atomic 加算が衝突し、実行ごとに加算順が変わって勾配がビット単位で
    変わる。旧実装（エキスパートごとに index_select）は1回の呼び出し内に重複が
    無いため決定的だったので、その性質を保つ。

    inv は order の逆順列（inv[order[i]] = i）。g を inv で並べ替えると
    「トークン順」に戻るので、あとは k 個ずつ通常の reduction で足せば順序が固定
    される。逆順列による index_select は重複が無いので backward も決定的。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, token_ids: torch.Tensor, inv: torch.Tensor, k: int):
        ctx.save_for_backward(inv)
        ctx.k = k
        ctx.n_tok = x.size(0)
        return x.index_select(0, token_ids)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (inv,) = ctx.saved_tensors
        g = grad_out.index_select(0, inv).view(ctx.n_tok, ctx.k, -1).sum(dim=1)
        return g, None, None, None


class MoE(nn.Module):
    """Routed MoE。エキスパート重みは 1 本のテンソルにスタックして保持する。

    重みレイアウト
    ------------------------------------------------------------------
        w13 : [E, D, 2F]   w1 が [..., :F]、w3 が [..., F:]（nn.Linear の転置）
        w2  : [E, F, D]

    ModuleList をやめた理由（速度）
    ------------------------------------------------------------------
    旧実装は Python の for でエキスパートを1個ずつ回し、境界を得るために
    counts.cumsum(0).cpu() で毎層 GPU->CPU 同期していた。E=64/L=11 では

        MoE 1層あたり 554 カーネル / 学習1ステップ 26,430 カーネル

    となり、カーネル発行数がトークン数ではなく「E x 層数」で決まる。この結果
    16k トークン以下では GPU 稼働率が 28-46% まで落ち、GPU が命令待ちで遊ぶ。
    スタック + _grouped_mm にすると発行数は 10,414 に減り、同期も無くなる。

    実測（A80M_E64 = D768/F192/E64/k7/L11、fwd+bwd の1 micro-batch）:

        1,024 tok : 162.0 ms ->  42.8 ms (3.78x)
        4,096 tok : 179.5 ms ->  47.4 ms (3.79x)
        8,192 tok : 180.3 ms ->  75.2 ms (2.40x)
       16,384 tok : 204.4 ms -> 130.4 ms (1.57x)
       32,768 tok : 302.4 ms -> 252.1 ms (1.20x)   <- 現行の学習設定
        生成 256  :  41.0 ms ->  15.0 ms (2.73x)

    32,768 tok で 1.2x に留まるのは、そこでは既に GPU 稼働率が 85% あり
    CPU 律速が解けているため。小さい micro-batch ほど効く。

    E が小さいと逆効果になる（同じ計算量で E=4 なら 0.59x = 1.7倍遅い）。
    ループ1回あたりの GPU 仕事量が大きくなり、発行コストが相対的に消えるため。
    損益分岐は E=8 付近なので grouped_min_experts で切り替える。

    _grouped_mm を使わない条件（自動でループ経路に落ちる）
    ------------------------------------------------------------------
      - use_ffn_lora=True   : lora.Linear はスタックできない
      - E < grouped_min_experts
      - CUDA 以外 / bf16・fp16 以外（_grouped_mm の要件）

    ループ経路もスタック済みの重みを使うので、どちらを通っても state_dict は同一。

    再現性（変更するときは必ずここを読むこと）
    ------------------------------------------------------------------
    forward・backward とも run-to-run でビット単位に再現する。旧実装が持っていた
    性質で、論文の再現性のために維持している。実測で確認済み。

    これは自明ではない。1トークンが k 個のエキスパートに送られるので、素直に
    書くと dispatch / combine の両方で「同じ行への重複 index」が生じ、
    index_add の atomic 加算が実行ごとに違う順序で走って結果が変わる
    （実際に一度そう書いて run-to-run で 1.4e-2 ずれた）。旧実装はエキスパート
    ごとにループしていたため1回の呼び出し内に重複が無く、たまたま決定的だった。

    そこで両方向とも「逆順列 inv で並べ替えてから k 個ずつ通常の reduction で
    足す」形にしてある（combine は _routed_grouped 内、dispatch は
    _DispatchGather.backward）。逆順列による index_select は重複が無いので
    その backward も決定的になる。

    例外: use_bias=True のとき b13 の勾配だけは非決定的になる。b13 を
    index_select で行ごとに展開するため、その backward が重複 index への
    atomic 加算になるため。実運用の A80M_E64 は use_bias=false なので影響しない。
    """

    _GROUPED_DTYPES = (torch.bfloat16, torch.float16)

    def __init__(self, args: MORTMArgs, layer_id, route_scale=1):
        super().__init__()
        self.dim = args.d_model
        self.d_ff = args.dim_feedforward
        self.n_routed_experts = args.num_experts
        self.n_activated_experts = args.topk_experts
        self.gate = Gate(args, layer_id, route_scale=route_scale)
        self.shared_experts = SharedExpert(args)

        self.use_lora = bool(args.use_ffn_lora)
        self.use_bias = bool(args.use_bias)
        self.grouped_min_experts = getattr(args, "moe_grouped_min_experts", 16)

        if self.use_lora:
            # LoRA は低ランク行列を各エキスパートに持つのでスタックできない
            self.experts = nn.ModuleList([Expert(args) for _ in range(self.n_routed_experts)])
            self.stacked = False
            return

        self.stacked = True
        E, D, FF = self.n_routed_experts, self.dim, self.d_ff
        self.w13 = nn.Parameter(torch.empty(E, D, 2 * FF))
        self.w2 = nn.Parameter(torch.empty(E, FF, D))
        if self.use_bias:
            self.b13 = nn.Parameter(torch.empty(E, 2 * FF))
            self.b2 = nn.Parameter(torch.empty(E, D))
        self._reset_expert_parameters()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _reset_expert_parameters(self) -> None:
        """nn.Linear の既定初期化と一致させる。

        nn.Linear は kaiming_uniform_(a=sqrt(5)) で、これは
        U(-1/sqrt(fan_in), 1/sqrt(fan_in)) と等価。w13 は転置で保持しているので
        fan_in は D、w2 は FF になる。ここを変えると学習の初期挙動が変わる。
        """
        bw = 1.0 / math.sqrt(self.dim)
        bo = 1.0 / math.sqrt(self.d_ff)
        self.w13.uniform_(-bw, bw)
        self.w2.uniform_(-bo, bo)
        if self.use_bias:
            self.b13.uniform_(-bw, bw)
            self.b2.uniform_(-bo, bo)

    def _use_grouped(self, x: torch.Tensor) -> bool:
        return (self.stacked
                and self.n_routed_experts >= self.grouped_min_experts
                and x.is_cuda
                and x.dtype in self._GROUPED_DTYPES
                and hasattr(torch, "_grouped_mm"))

    # ------------------------------------------------------------------
    @staticmethod
    def _route(indices: torch.Tensor, n_exp: int, k: int):
        """(expert 昇順の割当, 並べ替え順, 対応トークン, グループ境界) を返す。

        境界は int32 のまま GPU に置く。ここを .cpu() すると層ごとに
        GPU->CPU 同期が入り、CPU 側の先行実行が止まる（それが旧実装だった）。

        token_ids は旧実装の arange(T).repeat_interleave(k)[order] と同値。
        indices を flatten した位置 p は必ずトークン p//k のものなので、
        order をそのまま整数除算すれば同じ列が得られる（実測で完全一致）。
        """
        sorted_expert, order = torch.sort(indices.reshape(-1))
        token_ids = order // k
        offsets = torch.bincount(sorted_expert, minlength=n_exp).cumsum(0).to(torch.int32)
        return sorted_expert, order, token_ids, offsets

    def _routed_grouped(self, x_flat: torch.Tensor, weights, indices) -> torch.Tensor:
        k = self.n_activated_experts
        n_tok = x_flat.size(0)
        sorted_expert, order, token_ids, offsets = self._route(
            indices, self.n_routed_experts, k)

        # order の逆順列。dispatch の backward と combine の両方で使う
        inv = torch.empty_like(order)
        inv.scatter_(0, order, torch.arange(order.numel(), device=order.device))

        xs = _DispatchGather.apply(x_flat, token_ids, inv, k)
        h = torch._grouped_mm(xs, self.w13, offs=offsets)
        if self.use_bias:
            h = h + self.b13.index_select(0, sorted_expert)
        a, b = h.chunk(2, dim=-1)
        # chunk は非連続 view を返すので _grouped_mm に渡す前に詰める
        out = torch._grouped_mm((F.silu(a) * b).contiguous(), self.w2, offs=offsets)
        if self.use_bias:
            out = out + self.b2.index_select(0, sorted_expert)

        # 逆順列でトークン順に戻し、各トークンの k 個を通常の reduction で足す。
        #
        # ここを index_add_ にしてはいけない。1トークンが k 回現れるので同じ行への
        # atomic 加算が衝突し、実行ごとに加算順が変わって結果がビット単位で変わる
        # （実測 1.4e-2 の run-to-run 差）。旧実装はエキスパートごとに index_add_ を
        # 呼んでおり、1回の呼び出し内では同じトークンが2度出ないため決定的だった。
        # 論文用の再現性を保つため、その性質を維持する。
        out = out.index_select(0, inv).view(n_tok, k, -1)
        return (out * weights.reshape(n_tok, k, 1)).sum(dim=1)

    def _routed_loop(self, x_flat: torch.Tensor, weights, indices) -> torch.Tensor:
        """_grouped_mm を使えない場合の経路。ここだけ GPU->CPU 同期が入る。"""
        k = self.n_activated_experts
        assign_expert = indices.reshape(-1)
        order = torch.argsort(assign_expert)
        assign_expert = assign_expert[order]
        assign_weight = weights.reshape(-1)[order]
        token_ids = order // k

        counts = torch.bincount(assign_expert, minlength=self.n_routed_experts)
        boundaries = counts.cumsum(0).cpu().tolist()

        y_flat = torch.zeros_like(x_flat)
        start = 0
        for expert_id, end in enumerate(boundaries):
            if end == start:
                continue
            cur_token_ids = token_ids[start:end]
            cur_weights = assign_weight[start:end].unsqueeze(-1)
            expert_in = x_flat.index_select(0, cur_token_ids)

            if self.stacked:
                w13 = self.w13[expert_id]
                h = expert_in @ w13
                if self.use_bias:
                    h = h + self.b13[expert_id]
                a, b = h.chunk(2, dim=-1)
                expert_out = (F.silu(a) * b) @ self.w2[expert_id]
                if self.use_bias:
                    expert_out = expert_out + self.b2[expert_id]
            else:
                expert_out = self.experts[expert_id](expert_in)

            y_flat.index_add_(0, cur_token_ids, expert_out * cur_weights)
            start = end
        return y_flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.gate(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))  # [T, D]

        route = self._routed_grouped if self._use_grouped(x_flat) else self._routed_loop
        y = route(x_flat, weights, indices)

        z = self.shared_experts(x_flat)
        return (y + z).view(orig_shape)


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