from typing import Optional

import numpy as np
import torch

from mortm.gmail_messanger import GmailMessanger
from mortm.messager import Messenger
from mortm.models.modules.progress import _DefaultLearningProgress
import loralib as lora
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.train.train import _get_padding_mask
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from mortm.train.config import TrainArgs
from mortm.train.config import AbstractTrainSet
from mortm.models.mortm import MORTM, MORTMArgs
from mortm.train.datasets import MORTM_SEQDataset
from mortm.train.noam import noam_lr
from mortm.train.train import train_custom, collate_fn
from mortm.train.utils.loss import MaskedCrossEntropyLoss


class TTMORTM(AbstractTrainSet):
    def __init__(self, args_config: str, load_model_directory, tokenizer: Tokenizer, progress):
        self.args = MORTMArgs(args_config)
        self.args.use_lora = True
        self.tokenizer = tokenizer
        self.model = MORTM(self.args, progress).to(device=progress.get_device())
        self.model.load_state_dict(torch.load(load_model_directory), strict=False)

        lora.mark_only_lora_as_trainable(self.model)

        for name, p in self.model.named_parameters(): #EmbeddingとWout以外のパラメータを学習しない
            if "embedding" in name or "Wout" in name:
                p.requires_grad = True

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        adam = torch.optim.Adam(trainable_params, lr=2e-1)

        super().__init__(optimizer=adam,
                         scheduler=LambdaLR(optimizer=adam, lr_lambda=noam_lr(d_model=self.args.d_model, warmup_steps=4000)),
                         criterion=MaskedCrossEntropyLoss(ignore_index=0))


    def epoch_fc(self, model, pack, progress):
        src = pack
        target: Tensor = src[:, 1:].to(progress.get_device())
        mask = self.loss_mask(target)
        target = target.reshape(-1).long()
        mask = mask.reshape(-1).long()

        src = src[:, :-1]
        padding_mask_in: Tensor = _get_padding_mask(src, progress)

        input: Tensor = model(x=src, padding_mask=padding_mask_in, is_causal=True)
        input = input.view(-1, input.size(-1)).to(progress.get_device())
        return input.to(device=progress.get_device(), dtype=torch.float32), target, mask


    def pre_processing(self, pack, progress):
        dt: DataLoader = pack
        mini_dataset = MORTM_SEQDataset(progress, self.args.position_length, self.args.min_length)
        for d in dt:
            np_load_data = np.load(d, allow_pickle=True)
            mini_dataset.add_data(np_load_data)

        return mini_dataset

    def loss_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        バッチ処理に対応した損失マスクを作成する。

        Args:
            x (torch.Tensor): 形状が (batch_size, sequence_length) の入力テンソル。

        Returns:
            torch.Tensor: 形状が (batch_size, sequence_length) のマスクテンソル。
        """
        # x が (batch_size, sequence_length) の場合、
        # 以下の比較も要素ごとに行われ、結果は (batch_size, sequence_length) の
        # ブール型テンソルになる。
        mgen_id = self.tokenizer.get("<MGEN>")
        cgen_id = self.tokenizer.get("<CGEN>")
        is_start_token = ((x == mgen_id) | (x == cgen_id)).long()

        # `dim=1` を指定しているため、累積和はシーケンス長（次元1）に沿って
        # バッチ内の各サンプル（各行）ごとに独立して計算される。
        # バッチをまたいで計算されることはない。
        cumulative_mask = torch.cumsum(is_start_token, dim=1)

        # この比較も要素ごとに行われる。
        mask_x = (cumulative_mask > 0).long()
        # マスクを入力`x`と同じデバイスに転送する。
        return mask_x.to(x.device)


if __name__ == "__main__":
    MODEL_CONFIG =  "configs/models/mortm/A.json"
    TRAIN_CONFIG = "configs/train/task_training.json"
    LOOT_DIRECTORY = "out/np/Sax/task_train/"
    LOAD_MODEL_DIRECTORY = "out/model/mortm/MORTM.4.0EX5-SAX-Phase1_1.4054.pth"
    SAVE_DIRECTORY = "out/model/mortm"
    VERSION = "4.0-SAX-Phase2"

    message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)
    progress = _DefaultLearningProgress()
    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))
    trainer = TTMORTM(args_config=MODEL_CONFIG, load_model_directory=LOAD_MODEL_DIRECTORY, tokenizer=tokenizer, progress=progress)
    t_args = TrainArgs(TRAIN_CONFIG)

    train_custom(
        trainer=trainer,
        t_args=t_args,
        root_directory=LOOT_DIRECTORY,
        save_directory=SAVE_DIRECTORY,
        version=VERSION,
        message=message,
        progress=progress,
        coll_fn=collate_fn
    )
