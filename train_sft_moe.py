"""汎用 MoE SFT (Supervised Fine-Tuning) 学習スクリプト:
MORTM.4.5E-A80M-E64 などの MoE モデルをベースに、指定データセットで生成タスク SFT を実施。
- Attention (qkv, out) + Router (Gate) に LoRA (r=8, alpha=16)
- 64基の Routed Experts は Freeze（_grouped_mm 高速実行を維持）
- Router 負荷分散バイアスは OFF (use_gate_bias=False)
- Embedding + Wout はフル学習
- MaskedCrossEntropyLoss (<MGEN> 以降のみ損失計算)

使用例:
  # MAESTRO で学習
  python3 train_sft_moe.py --dataset maestro

  # JAZZ で学習
  python3 train_sft_moe.py --dataset jazz

  # マルチ GPU (例: 2 GPU)
  torchrun --nproc_per_node=2 train_sft_moe.py --dataset maestro
"""
import argparse
import os
import json
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from loralib import mark_only_lora_as_trainable

from torch import Tensor
from mortm.train.train import MORTMTrainSet, train_custom, collate_fn, _get_padding_mask
from mortm.train.config import AbstractTrainSet, TrainArgs
from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.train.utils.loss import MaskedCrossEntropyLoss
from mortm.utils.messager import _DefaultMessenger
from mortm.models.modules.progress import _DefaultLearningProgress


class MORTMMoESFTTrainSet(MORTMTrainSet):
    """MORTMTrainSet を MoE SFT 用に拡張。
    Attention + Router(Gate) LoRA + Embedding/Wout 解凍 + MaskedCrossEntropyLoss。
    """

    def __init__(self, args: MORTMArgs, t_args: TrainArgs, tokenizer: Tokenizer,
                 calc_val_loss_tokens, progress, lora_rank=8, lora_alpha=16,
                 log_scale=False, project_name="", config=None,
                 load_directory=None, model_name=None):
        # --- MoE LoRA 設定 ---
        args.use_attn_lora = True
        args.use_gate_lora = False
        args.use_ffn_lora = False       # Routed Experts は Stacked のまま固定
        args.use_shared_expert_lora = True
        args.use_gate_bias = False      # 特化型 SFT では均等化バイアスの介入を OFF
        args.lora_r = lora_rank
        args.lora_alpha = lora_alpha

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.tokenizer = tokenizer
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # モデル構築
        self.model = MORTM(progress=progress, args=args).to(device)

        # ベース重みロード
        if load_directory is not None:
            sd = torch.load(load_directory, map_location=device)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            non_lora_missing = [k for k in missing if "lora_" not in k]
            if self.local_rank == 0:
                print(f"[MoE-SFT] Base loaded from {load_directory}")
                print(f"          missing={len(missing)} (non-lora={len(non_lora_missing)}), unexpected={len(unexpected)}")
                if non_lora_missing:
                    print(f"          [WARN] non-lora missing: {non_lora_missing[:5]}")
                if unexpected:
                    print(f"          [WARN] unexpected: {unexpected[:5]}")

        # 特化型SFT: 事前学習バイアスの適用は維持し、学習中の更新のみフリーズ
        for m in self.model.modules():
            if hasattr(m, "manual_step"):
                m.manual_step = True

        # MoE スタック重みとの型一致 (_grouped_mm が BFloat16 を要求するため)
        if torch.cuda.is_available():
            self.model = self.model.to(dtype=torch.bfloat16)

        # ベース重みを凍結し、LoRA アダプタのみ学習可能に
        mark_only_lora_as_trainable(self.model)

        # Embedding と Wout のみフル学習 (語彙トークンの意味獲得)
        # ※ 64基の Routed Experts は完全凍結を維持 (事前学習の記憶を保護)
        n_unfrozen = 0
        for name, p in self.model.named_parameters():
            if "embedding" in name.lower() or name.endswith("Wout.weight") or ".Wout." in name:
                p.requires_grad = True
                n_unfrozen += p.numel()

        if torch.cuda.is_available() and torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
            total_param, self.active_params = self.model.module.get_param()
        else:
            total_param, self.active_params = self.model.get_param()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        if self.local_rank == 0:
            print(f"[MoE-SFT] LoRA rank={self.lora_rank}, alpha={self.lora_alpha}")
            print(f"          Trainable params: {n_train/1e6:.3f}M (Embedding+Wout: {n_unfrozen/1e6:.3f}M, LoRA: {(n_train - n_unfrozen)/1e6:.3f}M)")
            print(f"          Routed Experts are FROZEN. Active params: {self.active_params/1e6:.2f}M / Total params: {total_param/1e6:.2f}M")

        adam = torch.optim.AdamW(trainable, lr=t_args.lr_param)

        # 損失関数: MaskedCrossEntropyLoss (<MGEN> 以降のみ損失計算)
        AbstractTrainSet.__init__(
            self,
            criterion=MaskedCrossEntropyLoss(ignore_index=0).to(device),
            optimizer=adam,
            t_args=t_args,
            m_args=args,
            calc_val_loss_tokens=calc_val_loss_tokens,
        )

        self.all_tokens = torch.tensor(0, device=device, dtype=torch.long)
        self.optimizer_steps = 0

    def epoch_fc(self, model, pack, progress):
        """損失を <MGEN> 以降(生成対象CONST)のみに掛ける"""
        src = pack
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        target2d: Tensor = src[:, 1:].to(device)
        mask2d: Tensor = self.loss_mask(target2d)
        target = target2d.reshape(-1).long()
        mask = mask2d.reshape(-1)

        src = src[:, :-1].to(device)
        padding_mask_in: Tensor = _get_padding_mask(src, progress, device)
        if model.training:
            self.all_tokens += torch.sum(padding_mask_in == 1)

        out: Tensor = model(x=src, padding_mask=padding_mask_in, is_causal=True)
        out = out.view(-1, out.size(-1)).to(device)
        return out.to(device=device, dtype=torch.float32), target, mask


def build_parser():
    parser = argparse.ArgumentParser(description="MORTM MoE SFT (Supervised Fine-Tuning) Runner")
    parser.add_argument("--dataset", type=str, default="maestro",
                        help="データセット名 (例: maestro, jazz)。未指定のパス類を自動補完します。")
    parser.add_argument("--model_config", type=str, default="configs/models/mortm/foundation/A80M_E64.json",
                        help="モデル設定JSON")
    parser.add_argument("--train_config", type=str, default=None,
                        help="学習設定JSON (未指定時は configs/train/mortm/sft/{dataset}_moe.json)")
    parser.add_argument("--base_checkpoint", type=str, default="out/models/4_5/MORTM.4.5E-A80M-E64.pth",
                        help="ベースモデル重み (.pth)")
    parser.add_argument("--train_json", type=str, default=None,
                        help="学習データリストJSON (未指定時は data/sft/{dataset}/train.json)")
    parser.add_argument("--eval_json", type=str, default=None,
                        help="評価データリストJSON (未指定時は data/sft/{dataset}/eval.json)")
    parser.add_argument("--save_directory", type=str, default=None,
                        help="モデル保存先ディレクトリ (未指定時は out/models/mortm/sft/{dataset})")
    parser.add_argument("--version", type=str, default=None,
                        help="モデルバージョン名 (未指定時は MORTM.4.5E-A80M-E64-SFT-{dataset})")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ds = args.dataset
    train_config = args.train_config or f"configs/train/mortm/sft/{ds}_moe.json"
    train_json = args.train_json or f"data/sft/{ds}/train.json"
    eval_json = args.eval_json or f"data/sft/{ds}/eval.json"
    save_directory = args.save_directory or f"out/models/mortm/sft/{ds}"
    version = args.version or f"MORTM.4.5E-A80M-E64-SFT-{ds}"

    os.makedirs(save_directory, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("==================================================")
        print(f"MORTM MoE SFT Training Runner")
        print(f"Dataset         : {ds}")
        print(f"Version         : {version}")
        print(f"Model Config    : {args.model_config}")
        print(f"Train Config    : {train_config}")
        print(f"Base Checkpoint : {args.base_checkpoint}")
        print(f"Train JSON      : {train_json}")
        print(f"Eval JSON       : {eval_json}")
        print(f"Save Directory  : {save_directory}")
        print("==================================================")

    import torch.distributed as dist
    if not dist.is_initialized() and torch.cuda.is_available():
        dist.init_process_group(backend=os.environ.get("MORTM_DDP_BACKEND", "nccl"))

    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))
    progress = _DefaultLearningProgress()
    m_args = MORTMArgs(json_directory=args.model_config)
    t_args = TrainArgs(json_directory=train_config)

    trainer = MORTMMoESFTTrainSet(
        m_args, t_args, tokenizer, t_args.val_total_tokens, progress,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        log_scale=False,
        config=args.model_config,
        load_directory=args.base_checkpoint,
        model_name=version,
    )

    train_custom(
        trainer, t_args,
        root_directory=(train_json,),
        save_directory=save_directory,
        version=version,
        message=_DefaultMessenger(),
        eval_list_json=(eval_json,),
        progress=progress,
        coll_fn=collate_fn,
        resume=False,
    )


if __name__ == "__main__":
    main()
