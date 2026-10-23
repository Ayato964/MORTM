import copy
import json
import os
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR

from .noam import *


MORTM5_TRAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "muon_cos": {
        "optimizer": "muon",
        "lr_param": 0.001,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "ns_steps": 5,
        "adjust_lr_fn": "match_rms_adamw",
        "adamw_betas": [0.9, 0.95],
        "big_batch_size": 2048,
        "batch_size": 32,
        "accumulation_steps": 2,
        "num_epochs": 1,
        "is_save_training_progress": True,
        "val_total_tokens": 50000000,
        "train_dataset_split": 0.99,
        "shuffle": True,
        "checkpoint_percents": [25, 50, 75, 100],
        "scheduler": {
            "type": "cos",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.1,
            "total_steps": 17500,
            "auto_total_steps": True,
        },
    },
}
MORTM5_TRAIN_PRESETS["ablation_muon_cos"] = copy.deepcopy(MORTM5_TRAIN_PRESETS["muon_cos"])


def resolve_train_config_dict(source: Union[str, os.PathLike, Dict[str, Any]]) -> Dict[str, Any]:
    """辞書、内蔵学習プリセット名 ('muon_cos' / 'ablation_muon_cos')、または JSON パスから学習設定辞書を解決する。"""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    key = str(source).strip()
    lower_key = key.lower()
    if lower_key in MORTM5_TRAIN_PRESETS and not os.path.isfile(key):
        return copy.deepcopy(MORTM5_TRAIN_PRESETS[lower_key])
    with open(source, "r") as f:
        return json.load(f)


class TrainArgs:
    def __init__(self, json_directory: Union[str, os.PathLike, Dict[str, Any]] = "muon_cos"):
        data: dict = resolve_train_config_dict(json_directory)
        self._raw_config: dict = copy.deepcopy(data)
        self.batch_size = data['batch_size'] if data.get('batch_size') else 16
        self.is_save_training_progress = data['is_save_training_progress'] if data.get('is_save_training_progress') else False
        self.train_dataset_split: float = data['train_dataset_split'] if data.get('train_dataset_split') else 0.9
        self.accumulation_steps = data['accumulation_steps'] if data.get('accumulation_steps') else 4
        self.lr_param: Optional[float] = data['lr_param'] if data.get('lr_param') else None
        self.scheduler: dict = copy.deepcopy(data['scheduler']) if data.get('scheduler') else {"type": "noam", "warmup_steps": 4000}
        self.num_epochs = data['num_epochs'] if data.get('num_epochs') else 20
        self.big_batch_size = data['big_batch_size'] if data.get('big_batch_size') else 16
        self.val_total_tokens = data['val_total_tokens'] if data.get('val_total_tokens') else 50000000
        self.shuffle = data['shuffle'] if 'shuffle' in data else True
        self.optimizer_type: str = str(data.get('optimizer', 'adamw')).lower()
        self.weight_decay: float = float(data.get('weight_decay', 0.1))
        self.momentum: float = float(data.get('momentum', 0.95))
        self.ns_steps: int = int(data.get('ns_steps', 5))
        self.adjust_lr_fn: str = str(data.get('adjust_lr_fn', 'match_rms_adamw'))
        _betas = data.get('adamw_betas', [0.9, 0.95])
        self.adamw_betas: tuple[float, float] = (float(_betas[0]), float(_betas[1]))
        # §9.1/E2: フルFT の途中チェックポイントを total_steps の指定%時点で保存する。
        # None(既定)なら従来どおり保存しない(E1 等に影響なし・opt-in)。
        self.checkpoint_percents: Optional[list] = data.get('checkpoint_percents')
        dataset_batch_allocation = data.get('dataset_batch_allocation')
        self.dataset_batch_allocation: Optional[list[int]] = None

        if dataset_batch_allocation is not None:
            if not isinstance(dataset_batch_allocation, list) or len(dataset_batch_allocation) == 0:
                raise ValueError("dataset_batch_allocation must be a non-empty list of integers.")

            parsed_allocation = [int(v) for v in dataset_batch_allocation]
            if any(v < 0 for v in parsed_allocation):
                raise ValueError("dataset_batch_allocation must contain only non-negative integers.")
            if sum(parsed_allocation) <= 0:
                raise ValueError("dataset_batch_allocation must contain at least one positive integer.")

            self.dataset_batch_allocation = parsed_allocation

    def to_dict(self) -> Dict[str, Any]:
        """学習設定を JSON シリアライズ可能な辞書として返す。"""
        out = copy.deepcopy(getattr(self, "_raw_config", {}))
        out.update({
            "optimizer": self.optimizer_type,
            "lr_param": self.lr_param,
            "weight_decay": float(self.weight_decay),
            "momentum": float(self.momentum),
            "ns_steps": int(self.ns_steps),
            "adjust_lr_fn": str(self.adjust_lr_fn),
            "adamw_betas": [float(self.adamw_betas[0]), float(self.adamw_betas[1])],
            "big_batch_size": int(self.big_batch_size),
            "batch_size": int(self.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "num_epochs": int(self.num_epochs),
            "is_save_training_progress": bool(self.is_save_training_progress),
            "val_total_tokens": int(self.val_total_tokens),
            "train_dataset_split": float(self.train_dataset_split),
            "shuffle": bool(self.shuffle),
            "checkpoint_percents": copy.deepcopy(self.checkpoint_percents),
            "scheduler": copy.deepcopy(self.scheduler),
        })
        if self.dataset_batch_allocation is not None:
            out["dataset_batch_allocation"] = list(self.dataset_batch_allocation)
        return out

class AbstractTrainSet:
    model: nn.Module

    def __init__(self, criterion: nn.Module, optimizer: torch.optim.Optimizer, t_args: TrainArgs, m_args, calc_val_loss_tokens=50000000):
        self.criterion = criterion
        self.optimizer = optimizer
        self.t_args = t_args
        self.calc_val_loss_tokens = calc_val_loss_tokens
        _dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.all_tokens = torch.tensor(0, device=_dev, dtype=torch.long)
        self.last_val_calc_tokens = 0
        self.args = m_args
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self, override_total_steps: Optional[int] = None) -> Optional[LambdaLR]:
        sched_cfg = self.t_args.scheduler if isinstance(self.t_args.scheduler, dict) else {"type": "none"}
        sched_type = str(sched_cfg.get("type", "none")).lower()

        if sched_type == "noam":
            print("Using Noam Scheduler")
            _warmup = int(sched_cfg.get("warmup_steps", 4000))
            return LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=noam_lr(self.args.d_model, warmup_steps=_warmup),
            )

        if sched_type in ("cos", "cosine", "wsd"):
            raw_total = override_total_steps if override_total_steps is not None else sched_cfg.get("total_steps", 10000)
            _total = max(1, int(raw_total) if isinstance(raw_total, (int, float)) and int(raw_total) > 0 else 10000)
            sched_cfg["total_steps"] = _total

            if "warmup_steps" in sched_cfg and sched_cfg["warmup_steps"] is not None:
                _warmup = max(1, int(sched_cfg["warmup_steps"]))
            else:
                _warmup = max(1, int(float(sched_cfg.get("warmup_ratio", 0.05)) * _total))

            if "stable_steps" in sched_cfg and sched_cfg["stable_steps"] is not None:
                _stable = max(0, int(sched_cfg["stable_steps"]))
            else:
                _stable = max(0, int(float(sched_cfg.get("stable_ratio", 0.0)) * _total))

            _min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.05))
            label = "WSD-Cosine" if _stable > 0 or sched_type == "wsd" else "Cosine Annealing"
            print(
                f"Using {label} Scheduler "
                f"(total_steps={_total}, warmup_steps={_warmup}, stable_steps={_stable}, min_lr_ratio={_min_lr_ratio})"
            )
            return LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=get_cosine_schedule_with_warmup(
                    warmup_steps=_warmup,
                    total_steps=_total,
                    min_lr_ratio=_min_lr_ratio,
                    stable_steps=_stable,
                ),
            )

        return None

    def rebuild_scheduler(self, total_steps: int) -> None:
        """DataLoader 構築後に実ステップ数 total_steps でスケジューラを再構築する。"""
        if int(total_steps) > 0:
            self.scheduler = self._build_scheduler(override_total_steps=int(total_steps))

    def get_synced_tokens(self) -> int:
        if dist.is_initialized():
            sync_tokens = self.all_tokens.detach().clone()
            dist.all_reduce(sync_tokens, op=dist.ReduceOp.SUM)
            return int(sync_tokens.item())
        return int(self.all_tokens.item())

    @abstractmethod
    def epoch_fc(self, model, pack, progress):
        raise NotImplementedError("epoch_fc is not implemented.")

    @abstractmethod
    def pre_processing(self, pack, progress):
        raise NotImplementedError("pre_processing is not implemented.")

    @abstractmethod
    def view_logs(self, *args, **kwargs):
        raise NotImplementedError("view_logs is not implemented.")

    def optional_logging(self, *args, **kwargs):
        pass

    def is_need_calc_val(self):
        current_tokens = self.get_synced_tokens()

        if current_tokens == 0:
            return False

        if current_tokens >= self.last_val_calc_tokens + self.calc_val_loss_tokens:
            self.last_val_calc_tokens = current_tokens
            return True
        return False

    def backward(self, accumulation_steps, is_step, progress, lr_param, *args):
        loss: Tensor = self.criterion(*args)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\033[31m[Warning]\033[0m NaN or Inf loss detected: {loss.item() if loss.numel() == 1 else 'tensor'}. Skipping backward.")
            if is_step:
                self.optimizer.zero_grad()
            return loss.detach()

        return_loss = loss.clone()
        loss = loss / accumulation_steps
        loss.backward()

        if is_step:
            progress.step_optimizer(self.optimizer, self.model, accumulation_steps)
            if self.scheduler is not None:
                self.scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return return_loss

    def get_eval_loss(self, *eval):
        return self.criterion(*eval)
