"""MORTM.4.5E-A80M-E64 SFT (Supervised Fine-Tuning) Runner for Full Generation Dataset.

データセット:
  - train_json: /home/takaaki-nagoshi/data/sft/generation/train.json
  - eval_json: /home/takaaki-nagoshi/data/sft/generation/eval.json
  - train_config: configs/train/mortm/sft/generation.json

モデル:
  - model_config: configs/models/mortm/foundation/A80M_E64.json
  - base_checkpoint: out/models/4_5/MORTM.4.5E-A80M-E64_0.9015218019485474.pth
  - save_directory: out/models/mortm/sft/generation
  - version: MORTM.4.5E-A80M-E64-SFT-gen
  - wandb_project: MORTM_SFT_Demo
"""
import os
import json
import argparse
import torch
import wandb
from train_sft_moe import MORTMMoESFTTrainSet
from mortm.train.train import train_custom, collate_fn
from mortm.train.config import TrainArgs
from mortm.models.modules.config import MORTMArgs
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.messager import _DefaultMessenger
from mortm.models.modules.progress import _DefaultLearningProgress


def main():
    parser = argparse.ArgumentParser(description="Run full SFT for MORTM.4.5E-A80M-E64")
    parser.add_argument("--model_config", type=str, default="configs/models/mortm/foundation/A80M_E64.json",
                        help="モデル設定JSON")
    parser.add_argument("--train_config", type=str, default="configs/train/mortm/sft/generation.json",
                        help="学習設定JSON")
    parser.add_argument("--base_checkpoint", type=str,
                        default="out/models/4_5/MORTM.4.5E-A80M-E64_0.9015218019485474.pth",
                        help="ベースモデル重み (.pth)")
    parser.add_argument("--train_json", type=str,
                        default="/home/takaaki-nagoshi/data/sft/generation/train.json",
                        help="学習データリストJSON")
    parser.add_argument("--eval_json", type=str,
                        default="/home/takaaki-nagoshi/data/sft/generation/eval.json",
                        help="評価データリストJSON")
    parser.add_argument("--save_directory", type=str, default="out/models/mortm/sft/generation",
                        help="モデル保存先ディレクトリ")
    parser.add_argument("--version", type=str, default="MORTM.4.5E-A80M-E64-SFT-gen",
                        help="モデルバージョン名")
    parser.add_argument("--project", type=str, default="MORTM_SFT_Demo",
                        help="WandB プロジェクト名")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="VRAM調整用バッチサイズ (未指定時は train_config の値を尊重)")
    parser.add_argument("--accumulation_steps", type=int, default=None,
                        help="VRAM調整用累積ステップ数")
    parser.add_argument("--no_wandb", action="store_true", help="WandBログを無効化")
    args = parser.parse_args()

    os.makedirs(args.save_directory, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("==================================================")
        print("MORTM.4.5E-A80M-E64 Full Generation SFT Runner")
        print(f"Base Checkpoint : {args.base_checkpoint}")
        print(f"Model Config    : {args.model_config}")
        print(f"Train Config    : {args.train_config}")
        print(f"Train Dataset   : {args.train_json}")
        print(f"Eval Dataset    : {args.eval_json}")
        print(f"Save Directory  : {args.save_directory}")
        print(f"Version         : {args.version}")
        print(f"WandB Project   : {args.project} (Enabled: {not args.no_wandb})")
        print("==================================================")

    import torch.distributed as dist
    if not dist.is_initialized() and torch.cuda.is_available():
        dist.init_process_group(backend=os.environ.get("MORTM_DDP_BACKEND", "nccl"))

    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))
    progress = _DefaultLearningProgress()
    m_args = MORTMArgs(json_directory=args.model_config)
    t_args = TrainArgs(json_directory=args.train_config)

    if args.batch_size is not None:
        t_args.batch_size = args.batch_size
    if args.accumulation_steps is not None:
        t_args.accumulation_steps = args.accumulation_steps

    trainer = MORTMMoESFTTrainSet(
        m_args, t_args, tokenizer, t_args.val_total_tokens, progress,
        lora_rank=8,
        lora_alpha=16,
        log_scale=False,
        config=args.model_config,
        load_directory=args.base_checkpoint,
        model_name=args.version,
    )

    if local_rank == 0 and not args.no_wandb and os.environ.get("WANDB_MODE") != "disabled":
        try:
            wandb_cfg = {}
            if os.path.exists(args.model_config):
                with open(args.model_config, "r") as f:
                    wandb_cfg.update(json.load(f))
            if os.path.exists(args.train_config):
                with open(args.train_config, "r") as f:
                    wandb_cfg.update(json.load(f))
            wandb_cfg["base_checkpoint"] = args.base_checkpoint
            wandb_cfg["train_json"] = args.train_json
            wandb_cfg["batch_size"] = t_args.batch_size
            wandb_cfg["accumulation_steps"] = t_args.accumulation_steps
            wandb_cfg["lora_rank"] = 8
            wandb_cfg["lora_alpha"] = 16
            wandb.init(project=args.project, name=args.version, config=wandb_cfg, reinit=True)
            print(f"[WandB] Successfully initialized project='{args.project}', run='{args.version}'")
        except Exception as e:
            print(f"[WandB Skip] {e}")

    train_custom(
        trainer, t_args,
        root_directory=(args.train_json,),
        save_directory=args.save_directory,
        version=args.version,
        message=_DefaultMessenger(),
        eval_list_json=(args.eval_json,),
        progress=progress,
        coll_fn=collate_fn,
        resume=False,
    )


if __name__ == "__main__":
    main()
