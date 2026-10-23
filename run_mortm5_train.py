import argparse
import os
from typing import Optional

import torch

from mortm.models.modules.config import MORTM5Args
from mortm.models.mortm5 import ABLATION_MODEL_CONFIGS
from mortm.train.train import train_mortm5
from mortm.utils.messager import _DefaultMessenger


def resolve_model_config(model_key_or_path: str) -> str:
    if model_key_or_path in ABLATION_MODEL_CONFIGS:
        return ABLATION_MODEL_CONFIGS[model_key_or_path]
    return model_key_or_path


def resolve_token_dir(model_config_path: str, explicit_token_dir: Optional[str]) -> str:
    if explicit_token_dir is not None:
        return explicit_token_dir
    m_args = MORTM5Args(json_directory=model_config_path)
    tok_key = getattr(m_args, "tokenizer_key", None) or (
        "m3_g8l2" if m_args.vq_groups == 8 else "m4_g4l4"
    )
    return os.path.join("out/mortm5_tokens", tok_key)


def main():
    parser = argparse.ArgumentParser(description="MORTM5 8-Way Ablation Autoregressive Training")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/mortm5/ablation/m4_grouped_shared.json",
        help="Model config JSON path or ablation key (default: #7 m4_grouped_shared)",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/mortm5/ablation_muon_cos.json",
        help="Training config JSON path",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default=None,
        help="Directory containing {train,val}_{tokens.bin,index.npy,slots.npy} (auto-resolved from tokenizer_key if omitted)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/models/mortm5/ablation/",
        help="Directory to save checkpoints and trained weights",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="m4_grouped_shared_v1",
        help="Run version / experiment tag",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="MORTM5_Ablation",
        help="W&B project name",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Optional pretrained model weights (.pth) to initialize from",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit .checkpoint.pt path for resuming",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed across all ranks",
    )
    parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        default=True,
        help="Disable automatic checkpoint resume",
    )

    cli_args = parser.parse_args()

    model_config_path = resolve_model_config(cli_args.model_config)
    token_dir = resolve_token_dir(model_config_path, cli_args.token_dir)
    message = _DefaultMessenger()

    train_mortm5(
        model_config=model_config_path,
        train_config=cli_args.train_config,
        token_directory=token_dir,
        save_directory=cli_args.save_dir,
        version=cli_args.version,
        message=message,
        load_model_directory=cli_args.load_model,
        log_scale=True,
        project_name=cli_args.project_name,
        seed=cli_args.seed,
        resume=cli_args.resume,
        resume_checkpoint_path=cli_args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
