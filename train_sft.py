"""SFT 学習 (デモ用): SOTA 80M (MORTM.4.5D-Lite) に LoRA を載せて生成タスクを微調整する。

MORTMTrainSet を継承した MORTMSFTTrainSet を定義し、以下を差し込む:
  1. 損失を MaskedCrossEntropyLoss に変更 (<MGEN> 以降のみ損失 = SFT)
  2. Attention と FFN に LoRA を追加
  3. LoRA rank = 4
ベース重みは凍結し LoRA アダプタのみ学習する。学習は train_custom で起動 (run_train.py と同型)。
"""
import os
import json

import torch
import torch.nn as nn
import wandb
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


class MORTMSFTTrainSet(MORTMTrainSet):
    """MORTMTrainSet を SFT 用に拡張。LoRA(attn+ffn, r=4) + MaskedCrossEntropyLoss。
    ベース重みを SOTA チェックポイントからロードして凍結し、LoRA のみ学習する。
    epoch_fc / pre_processing / view_logs / loss_mask などは親をそのまま継承。
    """

    LORA_RANK = 4

    def __init__(self, args: MORTMArgs, t_args: TrainArgs, tokenizer: Tokenizer,
                 calc_val_loss_tokens, progress, log_scale=False, project_name="",
                 config=None, load_directory=None, model_name=None):
        # --- 2 & 3: Attention と FFN に LoRA を追加 (rank=4)。モデル構築前に設定する ---
        args.use_attn_lora = True
        args.use_ffn_lora = True
        args.use_gate_lora = False
        args.lora_r = self.LORA_RANK

        self.tokenizer = tokenizer
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{self.local_rank}")

        # LoRA 付きモデルを構築
        self.model = MORTM(progress=progress, args=args).to(device)

        # --- ベース SOTA 重みをロード。LoRA(lora_A/lora_B)は ckpt に無いため strict=False ---
        if load_directory is not None:
            sd = torch.load(load_directory, map_location=device)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            non_lora_missing = [k for k in missing if "lora_" not in k]
            if self.local_rank == 0:
                print(f"[SFT] base loaded: missing={len(missing)}(lora以外={len(non_lora_missing)}), "
                      f"unexpected={len(unexpected)}")
                if non_lora_missing:
                    print(f"[SFT][警告] lora以外のmissingキー: {non_lora_missing[:5]}")
                if unexpected:
                    print(f"[SFT][警告] unexpectedキー: {unexpected[:5]}")

        # --- ベース凍結、LoRA アダプタのみ学習可能に ---
        mark_only_lora_as_trainable(self.model)

        # --- Embedding と Wout はフル学習 (ジャンル等の新規/語彙トークンの意味を獲得させる) ---
        n_unfrozen = 0
        for name, p in self.model.named_parameters():
            if "embedding" in name.lower() or name.endswith("Wout.weight") or ".Wout." in name:
                p.requires_grad = True
                n_unfrozen += p.numel()
        if self.local_rank == 0:
            print(f"[SFT] Embedding+Wout フル学習: 解凍 {n_unfrozen/1e6:.2f}M")

        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        total_param, self.active_params = self.model.module.get_param()

        # --- optimizer は学習可能(LoRA)パラメータのみ ---
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        if self.local_rank == 0:
            print(f"[SFT] LoRA rank={self.LORA_RANK}  学習可能パラメータ={n_train/1e6:.3f}M / 全体={total_param/1e6:.1f}M")
        adam = torch.optim.AdamW(trainable, lr=t_args.lr_param)

        if self.local_rank == 0 and log_scale and config is not None:
            with open(config, "r") as f:
                data: dict = json.load(f)
            data["model_params"] = self.active_params
            data["total_params"] = total_param
            data["sft_lora_rank"] = self.LORA_RANK
            wandb.init(project=project_name, name=model_name, config=data, reinit=True)

        # --- 1: 損失を MaskedCrossEntropyLoss に変更 ---
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
        """SFT用: 損失を <MGEN> 以降(生成対象CONST)のみに掛ける。
        Meta / PAST / FUTURE / SYSTEM などユーザー入力(条件)部分は損失計算しない。
        (親 MORTMTrainSet.epoch_fc は (input,target) のみでマスク無し → ここで mask を追加)
        """
        src = pack
        device = torch.device(f"cuda:{self.local_rank}")
        target2d: Tensor = src[:, 1:].to(device)          # [B, S-1]
        mask2d: Tensor = self.loss_mask(target2d)         # <MGEN>以降=1, それ以前=0
        target = target2d.reshape(-1).long()
        mask = mask2d.reshape(-1)

        src = src[:, :-1].to(device)
        padding_mask_in: Tensor = _get_padding_mask(src, progress, device)
        if model.training:
            self.all_tokens += torch.sum(padding_mask_in == 1)

        out: Tensor = model(x=src, padding_mask=padding_mask_in, is_causal=True)
        out = out.view(-1, out.size(-1)).to(device)
        # criterion=MaskedCrossEntropyLoss(inputs, targets, mask): maskで<MGEN>以前の損失を0に
        return out.to(device=device, dtype=torch.float32), target, mask


def run_sft(model_config, train_config, base_checkpoint, root_directory, save_directory,
            version, eval_list_json=None, project_name="MORTM_SFT_Demo", log_scale=True):
    """run_train.py と同型。SFT トレーナーを構築して train_custom で起動する。"""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))
    progress = _DefaultLearningProgress()
    args = MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)

    trainer = MORTMSFTTrainSet(
        args, t_args, tokenizer, t_args.val_total_tokens, progress,
        log_scale=log_scale, project_name=project_name,
        config=model_config, load_directory=base_checkpoint, model_name=version,
    )

    train_custom(
        trainer, t_args, root_directory, save_directory, version,
        message=_DefaultMessenger(), eval_list_json=eval_list_json, progress=progress,
        coll_fn=collate_fn,  # 可変長系列をパディング(train_mortm と同じ)。None だと default_collate が stack で落ちる
    )


if __name__ == "__main__":
    BASE_CKPT = "out/models/4_5/MORTM.4.5D-160M.pth"
    MODEL_CONFIG = "configs/models/mortm/foundation/160M.json"
    TRAIN_CONFIG = "configs/train/mortm/sft/generation.json"
    ROOT = ("/home/takaaki-nagoshi/data/sft/generation/train.json",)
    EVAL = ("/home/takaaki-nagoshi/data/sft/generation/eval.json",)
    SAVE_DIR = "out/models/mortm/sft/generation"
    VERSION = "4.5D-160M-SFT-gen"

    os.makedirs(SAVE_DIR, exist_ok=True)
    run_sft(MODEL_CONFIG, TRAIN_CONFIG, BASE_CKPT, ROOT, SAVE_DIR, VERSION, eval_list_json=EVAL)
