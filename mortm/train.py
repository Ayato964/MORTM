'''
MORTMの学習を行う際にこのモジュールを使います。
train_mortmメソッドを呼び出し、引数の型に合ったオブジェクトを代入してください。
最低でも、「データセット(Tokenizerで変換したもの)のディレクトリ」、「モデルの出力先のディレクトリ」,
「モデルのバージョン」,「ボキャブラリーサイズ」,「エポック回数」、「各トークンの出現回数のリスト」が必要です。
'''

import datetime
import json
import math
import os
import time
from abc import abstractmethod

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


from .messager import Messenger, _DefaultMessenger
from .progress import LearningProgress, _DefaultLearningProgress
from .datasets import MORTM_DataSets
from .mortm import MORTM
from .noam import noam_lr
from .loss import MORTCrossEntropyLoss

IS_DEBUG = False


def _send_prediction_end_time(message, loader_len, begin_time, end_time,
                              vocab_size: int, num_epochs: int, trans_layer, num_heads, d_model,
                              dim_feedforward, dropout, position_length):
    t = end_time - begin_time
    end_time_progress = (t * loader_len * num_epochs) / 3600
    message.send_message("終了見込みについて",
                         f"現在学習が進行しています。\n"
                         f"今回設定したパラメータに基づいて終了時刻を計算しました。\n"
                         f"ボキャブラリーサイズ:{vocab_size}\n"
                         f"エポック回数:{num_epochs}\n"
                         f"Transformerのレイヤー層:{trans_layer}\n"
                         f"Modelの次元数:{d_model}\n"
                         f"シーケンスの長さ:{dim_feedforward}\n"
                         f"ドロップアウト:{dropout}\n"
                         f"\n\n シーケンスの1回目の処理が終了しました。かかった時間は{t:.1f}秒でした。\n"
                         f"終了見込み時間は{end_time_progress:.2f}時間です"
                         )


# デバイスを取得
def _set_train_data(directory, datasets, progress: LearningProgress):
    print("Starting load....")
    mortm_datasets = MORTM_DataSets(progress)
    loss_count = 0
    for dataset in datasets:
        print(f"Load [{directory + dataset}]")
        np_load_data = np.load(directory + dataset)
        if len(np_load_data) > 1:
            mortm_datasets.add_data(np_load_data)
            print(f"最初の5音:{mortm_datasets.musics_seq[-1][:5]}")
        else:
            loss_count += 1
    print("load Successful!!")
    print(f"データセットの規模（曲数）：{len(datasets) - loss_count}")
    print("---------------------------------------")

    return mortm_datasets

def _get_padding_mask(input_ids, progress: LearningProgress):
    # input_ids が Tensor であることを仮定
    pad_id = (input_ids != 0).to(torch.float)
    padding_mask = pad_id.to(progress.get_device())
    return padding_mask


def collate_fn(batch):
    # バッチ内のテンソルの長さを揃える（パディングする）
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return batch



def update_log(model, writer, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"Gradient Norm/{name}", param.grad.norm(), global_step)

        writer.add_scalar(f"Parameter Value/{name}", param.norm(), global_step)

def _train(save_directory, ayato_dataset, message: Messenger, vocab_size: int, num_epochs: int, weight: Tensor, progress: LearningProgress,
           trans_layer=6, load_model_directory:str = None,
           num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1, is_save_training_progress=False,
           position_length=2048, accumulation_steps=4, batch_size=16, num_workers=0, warmup_steps=4000, lr_param=1):

    loader = DataLoader(ayato_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=collate_fn)

    print("Creating Model....")
    model = MORTM(vocab_size=vocab_size, progress=progress, trans_layer=trans_layer, num_heads=num_heads,
                  d_model=d_model, dim_feedforward=dim_feedforward,
                  dropout=dropout, position_length=position_length).to(progress.get_device())
    if load_model_directory is not None:
        model.load_state_dict(torch.load(load_model_directory))

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight.to(progress.get_device())).to(progress.get_device())  # 損失関数を定義


    optimizer = torch.optim.Adam(model.parameters(), lr=lr_param, betas=(0.9, 0.98), weight_decay=1e-6)  # オプティマイザを定義
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam_lr(d_model=d_model, warmup_steps=warmup_steps))

    print("Start training...")
    writer = SummaryWriter(save_directory + "/runs/")

    loss_val = None
    mail_bool = True
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} start....")
        count = 1
        epoch_loss = 0.0
        verification_loss = 0.0

        model.train()
        optimizer.zero_grad()

        for inputs in loader:  # seqにはbatch_size分の楽曲が入っている
            begin_time = time.time()
            input_ids: Tensor = inputs[:, :-1].to(progress.get_device())
            targets: Tensor = inputs[:, 1:].to(progress.get_device())

            padding_mask_in: Tensor = _get_padding_mask(input_ids, progress)

            output = model(input_ids, padding_mask_in)

            outputs = output.view(-1, output.size(-1)).to(progress.get_device())
            targets = targets.reshape(-1).long()

            loss = criterion(outputs, targets)  # 損失を計算

            loss.backward()  # 逆伝播

            update_log(model, writer, count)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            if count % accumulation_steps == 0:  #実質バッチサイズは64である
                progress.step_optimizer(optimizer, model, accumulation_steps)
                scheduler.step()

            epoch_loss += loss.item()
            count += 1
            end_time = time.time()

            if mail_bool and message is not None:
                _send_prediction_end_time(message, len(loader), begin_time, end_time, vocab_size, num_epochs,
                                          trans_layer, num_heads, d_model, dim_feedforward, dropout, position_length)
                mail_bool = False

            if (count + 1) % message.step_by_message_count == 0:
                message.send_message("機械学習の途中経過について", f"Epoch {epoch + 1}/{num_epochs}の"
                                                                   f"learning sequence {count}結果は、\n {epoch_loss / count:.4f}でした。")
            writer.flush()

            progress_bar(epoch, num_epochs, count, len(loader), epoch_loss / count, scheduler.get_last_lr(), verification_loss)


        message.send_message("機械学習の途中経過について",
                                 f"Epoch {epoch + 1}/{num_epochs}の結果は、{epoch_loss / count:.4f}でした。")
        loss_val = epoch_loss / count
        writer.add_scalar('Loss/train', epoch_loss / count, epoch)  # 損失値を記録

        if is_save_training_progress:
            torch.save(model.state_dict(), f"{save_directory}/MORTM.train.{epoch}.{epoch_loss / count:.4f}.pth") #エポック終了時に途中経過を保存
            print("途中経過を保存しました。")
    writer.close()

    return model, loss_val


def progress_bar(epoch, sum_epoch, sequence, batch_size, loss, lr, verif_loss):
    per = sequence / batch_size * 100
    block = int(per / 100 * 50)
    bar = f" \033[32m{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"
    print(f"\r learning Epoch {epoch + 1}/{sum_epoch} [{bar}] {per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}", end="")


def train_mortm(dataset_directory, save_directory, version: str, vocab_size: int, num_epochs: int, weight_directory,
                message: Messenger = _DefaultMessenger(), load_model_directory: str=None,
                trans_layer=9, num_heads=32, d_model=1024, is_save_training_progress=False, lr_param=1,
                dim_feedforward=4096, dropout=0.2, position_length=8000, num_workers=0, warmup_steps=4000,
                accumulation_steps=32, batch_size=1, progress: LearningProgress = _DefaultLearningProgress()):

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')

    print(f"ToDay is{datetime.date.today()}! start generating MORTEM_Model.{version}_{today_date}")

    datasets = os.listdir(dataset_directory)
    train_data = _set_train_data(dataset_directory, datasets, progress)

    try:
        with open(weight_directory, 'r') as file:
            freq_dict = json.load(file)
            # 逆数を取り、頻出度が0の場合は小さい値に設定
            epsilon = 1e-11  # 非ゼロの小さい値を設定しておく
            weights = []

            for i in range(len(freq_dict)):
                freq = freq_dict[str(i)]  # JSONのキーは文字列なのでstrに変換
                if freq == 0:
                    weights.append(epsilon)
                else:
                    weights.append(1.0 / (math.log(freq + 1.0) + epsilon))  # 対数スケーリングを適用
            # テンソルに変換
            weight_tensor = torch.tensor(weights)
            #weight_tensor = weight_tensor / weight_tensor.sum()

            print(weight_tensor[weight_tensor.argmax(dim=-1)], weight_tensor[weight_tensor.argmin(dim=-1)])

        model, loss = _train(save_directory, train_data, message, vocab_size, num_epochs, weight_tensor, progress=progress,
                             load_model_directory=load_model_directory,
                             d_model=d_model,
                             dim_feedforward=dim_feedforward,
                             trans_layer=trans_layer,
                             num_heads=num_heads,
                             position_length=position_length,
                             dropout=dropout,
                             accumulation_steps=accumulation_steps,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             warmup_steps=warmup_steps,
                             is_save_training_progress=is_save_training_progress,
                             lr_param=lr_param
                             )  # 20エポック分機械学習を行う。

        message.send_message("機械学習終了のお知らせ",
                                 f"MORTM.{version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {loss}")

        torch.save(model.state_dict(), f"{save_directory}/MORTM.{version}_{loss}.pth")  # できたモデルをセーブする

        return model

    except torch.cuda.OutOfMemoryError:
        message.send_message("エラーが発生し、処理を中断しました",
                                 "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")
        print("オーバーフローしました。")
    pass

class VerificationLoss:

    def __init__(self, datasets: MORTM_DataSets):
        self.dataloader = DataLoader(dataset=datasets, collate_fn=collate_fn, shuffle=False)

    def __call__(self, *args, **kwargs):
        pass

