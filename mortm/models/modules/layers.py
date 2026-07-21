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

class MoE(nn.Module):

    def __init__(self, args: MORTMArgs, layer_id, route_scale=1):
        super().__init__()
        self.dim = args.d_model
        self.n_routed_experts = args.num_experts
        self.n_activated_experts = args.topk_experts
        self.gate = Gate(args, layer_id, route_scale=route_scale)
        self.experts = nn.ModuleList([Expert(args) for i in range(self.n_routed_experts)])
        self.shared_experts = SharedExpert(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.gate(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))  # [T, D]
        k = self.n_activated_experts

        assign_expert = indices.reshape(-1)  # [T*K]
        assign_weight = weights.reshape(-1)  # [T*K]

        token_ids = torch.arange(x_flat.size(0), device=x.device)
        token_ids = token_ids.repeat_interleave(k)  # [T*K]

        order = torch.argsort(assign_expert)
        assign_expert = assign_expert[order]
        assign_weight = assign_weight[order]
        token_ids = token_ids[order]

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
            expert_out = self.experts[expert_id](expert_in)

            y_flat.index_add_(0, cur_token_ids, expert_out * cur_weights)
            start = end

        z = self.shared_experts(x_flat).view(orig_shape)
        y = y_flat.view(orig_shape)
        return y + z


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