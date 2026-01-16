import json
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor


class TrainArgs:
    def __init__(self, json_directory: str):
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.batch_size = data['batch_size'] if data.get('batch_size') else 16
            self.is_save_training_progress = data['is_save_training_progress'] if data.get('is_save_training_progress') else False
            self.train_dataset_split:float = data['train_dataset_split'] if data.get('train_dataset_split') else 0.9
            self.accumulation_steps= data['accumulation_steps'] if data.get('accumulation_steps') else 4
            self.warmup_steps= data['warmup_steps'] if data.get('warmup_steps') else 4000
            self.lr_param: Optional[float]= data['lr_param'] if data.get('lr_param') else None
            self.num_epochs = data['num_epochs'] if data.get('num_epochs') else 20
            self.big_batch_size = data['big_batch_size'] if data.get('big_batch_size') else 16
            self.val_total_tokens = data['val_total_tokens'] if data.get('val_total_tokens') else 50000000


class AbstractTrainSet:
    model: nn.Module

    def __init__(self, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler.LambdaLR], calc_val_loss_tokens=50000000):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.calc_val_loss_tokens = calc_val_loss_tokens
        self.all_tokens = torch.tensor(0, device="cuda", dtype=torch.long)
        self.last_val_calc_tokens = 0

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
        if self.all_tokens == 0:
            return False

        if self.all_tokens >= self.last_val_calc_tokens + self.calc_val_loss_tokens:
            self.last_val_calc_tokens = self.all_tokens.item() # 実行タイミングを更新
            return True
        return False

    def backward(self, accumulation_steps, is_step, progress, lr_param, *args):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss: Tensor = self.criterion(*args)
        return_loss = loss.clone()
        loss = loss / accumulation_steps
        loss.backward()  # 逆伝播

        if is_step:
            progress.step_optimizer(self.optimizer, self.model, accumulation_steps)
            if lr_param is None and self.scheduler is not None:
                self.scheduler.step()
            torch.cuda.empty_cache()

        return return_loss

    def get_eval_loss(self, *eval):
        return self.criterion(*eval)