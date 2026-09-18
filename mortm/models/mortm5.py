from torch import nn
from .modules.layers import *
from vector_quantize_pytorch import ResidualVQ

import torch.distributed as dist
from typing import Optional

class MidiVision(nn.Module):
    def __init__(self, args: MORTM5Args, sync_codebook: Optional[bool] = None):
        super().__init__()
        self.encoder = ResNetEncoder(args)

        if sync_codebook is None:
            sync_codebook = dist.is_available() and dist.is_initialized()

        # 潜在ベクトルをRVQへ渡す前に正規化する。
        # 無しの場合 ||z|| が学習中に無制限に漂流し(実測: 初期26.5 -> 772step目で426)、
        # コードブックとスケールが乖離して全コードが「死んだコード」判定に落ちる。
        # LayerNormにより ||z|| は sqrt(encoder_wout) に固定される。
        self.pre_vq_norm = nn.LayerNorm(args.encoder_wout)

        self.rvq = ResidualVQ(
            dim=args.encoder_wout,
            codebook_dim=64,
            num_quantizers=args.encoder_layer,
            codebook_size=1024,
            sync_codebook=sync_codebook,
            # threshold_ema_dead_code: cluster_size は「1バッチあたり命中数」のEMA
            #   (cluster_size = cluster_size*decay + 命中数*(1-decay)) なので、
            #   定常値は1バッチ平均命中数そのもの。1チャンク=潜在ベクトル1本のため
            #   inner batch 512 で量子化器が見るのは512本しかなく、1024コードに配ると
            #   平均0.50命中/コード。既定の 2 では最大256コードしか生存できず
            #   (実測 perplexity が 254 で天井に張り付く)、残り半分以上が毎step
            #   置換され続ける回転ドア状態になる。バッチ/コード比0.5に合わせて下げる。
            #   実測: 2.0 -> 使用424/perplexity254/実効63.8bit
            #        0.25 -> 使用838/perplexity563/実効72.9bit
            threshold_ema_dead_code=0.25,

            # --- 以下はライブラリ既定値が本構成と致命的に噛み合わないため明示指定する ---

            # rotation_trick: 既定 True。dead code 置換で残差が厳密に0になった量子化器で
            #   rotated = ... * safe_div(||q||, ||x||) の分母が 1e-6 にクランプされ、
            #   勾配を最大 1.57e6 倍に増幅する(実測: grad norm 6.27e6)。
            #   通常の Straight-Through に戻すと同条件で 10.3 まで落ちる。
            rotation_trick=False,

            # kmeans_init: 既定 False (uniform乱数初期化) では潜在分布と全く重ならず、
            #   どのコードも選ばれないまま毎forwardで約1000/1024個が置換され続ける。
            kmeans_init=True,
            kmeans_iters=10,

            # decay: 既定 0.8 は速すぎてコードブックが各バッチに追従してしまう。
            decay=0.95,
        )

        self.decoder = ResNetDecoder(args)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre_vq_norm(self.encoder(x))
        x_q, tokens, commit_loss = self.rvq(x)
        x = self.decoder(x_q)
        return x, tokens, commit_loss
