from torch import nn
from .modules.layers import *
from vector_quantize_pytorch import GroupedResidualVQ

import torch.distributed as dist
from typing import Optional


class MidiVision(nn.Module):
    def __init__(self, args: MORTM5Args, sync_codebook: Optional[bool] = None):
        super().__init__()
        self.encoder = ResNetEncoder(args)

        if sync_codebook is None:
            sync_codebook = dist.is_available() and dist.is_initialized()

        self.vq_groups = getattr(args, "vq_groups", 4)
        assert args.encoder_wout % self.vq_groups == 0, (
            f"encoder_wout ({args.encoder_wout}) must be divisible by vq_groups ({self.vq_groups})"
        )
        self.codebook_dim = args.encoder_wout // self.vq_groups
        self.codebook_size = getattr(args, "codebook_size", 1024)

        # 各グループ(codebook_dim次元)ごとに独立してLayerNorm(アフィンなし)を適用する。
        # elementwise_affine=False により量子化入力の分散を厳密に 1.0 に固定し、スケール膨張とCommit Loss肥大化を根絶する。
        self.pre_vq_norm = nn.LayerNorm(self.codebook_dim, elementwise_affine=False)

        self.rvq = GroupedResidualVQ(
            dim=args.encoder_wout,
            groups=self.vq_groups,
            codebook_dim=self.codebook_dim,
            num_quantizers=args.encoder_layer,
            codebook_size=self.codebook_size,
            sync_codebook=sync_codebook,
            # 複数ビッグバッチを跨いでコードブックを安定保持するため長寿命EMA (decay=0.999) を使用
            threshold_ema_dead_code=0.05,
            # rotation_trick: 残差0時の勾配爆発を防ぐため Straight-Through を使用
            rotation_trick=False,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.999,
        )

        self.use_pre_vq_norm = getattr(args, "use_pre_vq_norm", True)
        self.decoder = ResNetDecoder(args)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if any(k.startswith("vq.") for k in state_dict.keys()):
            self.use_pre_vq_norm = False
            state_dict = {
                (k.replace("vq.", "rvq.", 1) if k.startswith("vq.") else k): v
                for k, v in state_dict.items()
            }
        elif "use_pre_vq_norm" not in self.__dict__:
            self.use_pre_vq_norm = True
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: Tensor):
        z = self.encoder(x)
        if getattr(self, "use_pre_vq_norm", True):
            orig_shape = z.shape
            # グループ単位で正規化: [..., encoder_wout] -> [..., vq_groups, codebook_dim] -> [..., encoder_wout]
            z_norm = self.pre_vq_norm(
                z.reshape(*orig_shape[:-1], self.vq_groups, self.codebook_dim)
            ).reshape(orig_shape)
        else:
            z_norm = z

        x_q, tokens, commit_loss = self.rvq(z_norm)
        # GroupedResidualVQ の tokens は [groups, B, (S,), num_quantizers] で返るため
        # [B, (S,), groups * num_quantizers] (例: [B, 8] または [B, 16]) に整形する
        tokens = tokens.movedim(0, -2).flatten(-2)
        commit_loss = commit_loss.mean()

        out = self.decoder(x_q)
        return out, tokens, commit_loss
