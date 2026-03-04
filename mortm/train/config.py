import json
from abc import abstractmethod
from typing import Optional

import torch
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


class AbstractTrainSet:
    model: nn.Module

    def __init__(self, criterion: nn.Module, optimizer: torch.optim.Optimizer,t_args: TrainArgs, m_args, calc_val_loss_tokens=50000000):
        self.criterion = criterion
        self.optimizer = optimizer
        self.t_args = t_args
        self.calc_val_loss_tokens = calc_val_loss_tokens
        self.all_tokens = torch.tensor(0, device="cuda", dtype=torch.long)
        self.last_val_calc_tokens = 0
        self.args=m_args

        if t_args.scheduler['type'] == "noam":
            print("Using Noam Scheduler")
            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam_lr(m_args.d_model, warmup_steps=t_args.scheduler['warmup_steps']))
        elif t_args.scheduler['type'] == "cos":
            print("Using Cosine Annealing Scheduler")
            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=get_cosine_schedule_with_warmup(warmup_ratio=t_args.scheduler['warmup_ratio'], total_steps=t_args.scheduler['total_steps']))


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
        # DDP環境では、各GPUが異なるデータ増強などにより処理トークン数に僅かな差が生じます。
        # ここで全GPUの合計トークン数を同期して確認しないと、検証に入るタイミングがズレてデッドロック（タイムアウト）の原因になります。
        if dist.is_initialized():
            sync_tokens = self.all_tokens.clone().detach()
            dist.all_reduce(sync_tokens, op=dist.ReduceOp.SUM)
            current_tokens = sync_tokens.item()
        else:
            current_tokens = self.all_tokens.item()

        if current_tokens == 0:
            return False

        if current_tokens >= self.last_val_calc_tokens + self.calc_val_loss_tokens:
            self.last_val_calc_tokens = current_tokens # 同期したトークン数で更新
            return True
        return False

    def backward(self, accumulation_steps, is_step, progress, lr_param, *args):
        loss: Tensor = self.criterion(*args)
        return_loss = loss.clone()
        loss = loss / accumulation_steps
        loss.backward()  # 逆伝播

        if is_step:
            progress.step_optimizer(self.optimizer, self.model, accumulation_steps)
            if self.scheduler is not None:
                self.scheduler.step()
            torch.cuda.empty_cache()

        return return_loss

    def get_eval_loss(self, *eval):
        return self.criterion(*eval)