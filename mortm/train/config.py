import json
from abc import abstractmethod
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR

from .noam import *

class TrainArgs:
    def __init__(self, json_directory: str):
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.batch_size = data['batch_size'] if data.get('batch_size') else 16
            self.is_save_training_progress = data['is_save_training_progress'] if data.get('is_save_training_progress') else False
            self.train_dataset_split:float = data['train_dataset_split'] if data.get('train_dataset_split') else 0.9
            self.accumulation_steps= data['accumulation_steps'] if data.get('accumulation_steps') else 4
            self.lr_param: Optional[float]= data['lr_param'] if data.get('lr_param') else None
            self.scheduler: dict = data['scheduler'] if data.get('scheduler') else {"type": "noam", "warmup_steps": 4000}
            self.num_epochs = data['num_epochs'] if data.get('num_epochs') else 20
            self.big_batch_size = data['big_batch_size'] if data.get('big_batch_size') else 16
            self.val_total_tokens = data['val_total_tokens'] if data.get('val_total_tokens') else 50000000
            self.shuffle = data['shuffle'] if data.get('shuffle') else True
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

class AbstractTrainSet:
    model: nn.Module

    def __init__(self, criterion: nn.Module, optimizer: torch.optim.Optimizer, t_args: TrainArgs, m_args, calc_val_loss_tokens=50000000):
        self.criterion = criterion
        self.optimizer = optimizer
        self.t_args = t_args
        self.calc_val_loss_tokens = calc_val_loss_tokens
        self.all_tokens = torch.tensor(0, device="cuda", dtype=torch.long)
        self.last_val_calc_tokens = 0
        self.args = m_args

        if t_args.scheduler['type'] == "noam":
            print("Using Noam Scheduler")
            _warmup = t_args.scheduler.get('warmup_steps', 4000)
            self.scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=noam_lr(m_args.d_model, warmup_steps=_warmup)
            )
        elif t_args.scheduler['type'] == "cos":
            print("Using Cosine Annealing Scheduler")
            _total = t_args.scheduler['total_steps']
            if 'warmup_steps' in t_args.scheduler:
                _warmup = t_args.scheduler['warmup_steps']
            else:
                _warmup = max(1, int(t_args.scheduler.get('warmup_ratio', 0.05) * _total))
            self.scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=get_cosine_schedule_with_warmup(
                    warmup_steps=_warmup,
                    total_steps=_total
                )
            )
        else:
            self.scheduler = None

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
        return_loss = loss.clone()
        loss = loss / accumulation_steps
        loss.backward()

        if is_step:
            progress.step_optimizer(self.optimizer, self.model, accumulation_steps)
            if self.scheduler is not None:
                self.scheduler.step()
            torch.cuda.empty_cache()

        return return_loss

    def get_eval_loss(self, *eval):
        return self.criterion(*eval)
