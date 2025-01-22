import datetime
import json
import math
import os
import time
from abc import abstractmethod
from typing import  Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


from .messager import Messenger, _DefaultMessenger
from .progress import LearningProgress, _DefaultLearningProgress
from .datasets import MORTM_SEQDataset
from .mortm import MORTM
from .noam import noam_lr
from .loss import ReinforceCrossEntropy
from .tokenizer import Tokenizer



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



def find_npz_files(root_folder):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith('.npz'):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files

def _set_train_data(directory, datasets, positional_length, progress: LearningProgress):
    print("Starting load....")
    mortm_datasets = MORTM_SEQDataset(progress, positional_length)
    loss_count = 0
    count = 0
    dataset_length = 0
    loss_data = 4
    for i in range(len(datasets)):
        count += 1

        np_load_data = np.load(f"{directory[i]}/{datasets[i]}", allow_pickle=True)
        if len(np_load_data) > loss_data:
            dataset_length += mortm_datasets.add_data(np_load_data)
            print(f"\r {count}/{len(datasets)} | Dataset Length:{dataset_length} | Load[{directory[i]}/{datasets[i]}]", end="")
        else:
            loss_count += 1
    print("load Successful!!")
    print(f"データセットの規模（曲数）：{len(datasets) - loss_count}")
    print("---------------------------------------")

    return mortm_datasets


def progress_bar(epoch, sum_epoch, sequence, batch_size, loss, lr, verif_loss, criterion:ReinforceCrossEntropy):
    per = sequence / batch_size * 100
    block = int(per / 100 * 50)
    #color_bar = get_color(criterion)
    color_bar = "\033[32m"
    bar = f" {color_bar}{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"
    print(f"\r learning Epoch {epoch + 1}/{sum_epoch} [{bar}] {per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}", end="")


def _get_padding_mask(input_ids, progress: LearningProgress):
    # input_ids が Tensor であることを仮定
    pad_id = (input_ids != 0).to(torch.float)
    padding_mask = pad_id.to(progress.get_device())
    return padding_mask


def collate_fn(batch):
    # バッチ内のテンソルの長さを揃える（パディングする）
    src = pad_sequence(batch, batch_first=True, padding_value=0)
    return src


def get_verification_loss(model: MORTM, val_loader: DataLoader, criterion: nn.CrossEntropyLoss, progress: LearningProgress):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src in val_loader:
            correct, loss_mask = get_correct_src2(src, progress)
            src = src[:, :-1]
            padding_mask_in: Tensor = _get_padding_mask(src, progress)

            outputs: Tensor = model(src=src, input_padding_mask=padding_mask_in)

            outputs = outputs.view(-1, outputs.size(-1)).to(progress.get_device())
            correct = correct.reshape(-1).long()
            loss_mask = loss_mask.reshape(-1).long()
            loss = criterion(outputs, correct)  # 損失を計算
            loss = loss * loss_mask
            loss = loss.sum() / loss_mask.sum()

            val_loss += loss
        model.train()
    return val_loss / len(val_loader)




def get_correct_src2(src: Tensor, progress) -> [Tensor, Tensor]:
    if not isinstance(src, Tensor):
        print(src)
        src = torch.tensor(src, device=progress.get_device())
    batch, ind = torch.nonzero(src == 2, as_tuple=True)
    result = []
    masks = []
    for b, i in zip(batch.tolist(), ind.tolist()):
        new_sequence = torch.cat((src[b, :i], src[b, i+1:]))
        result.append(new_sequence)

        pad_first = torch.where(src[b] == 0)[0]
        mask = torch.zeros(len(src[b, :i])) #Lossに含まない部分を計算
        if pad_first.numel() != 0:
            pad_first = pad_first[0]
            loss = torch.ones(len(src[b, i+1:pad_first])) #Lossを産出する部分
            pad = torch.zeros(len(src[b, pad_first:])) #PaddingによるLossを算出しない部分
            masks.append(torch.cat((mask, loss, pad)))
        else:
            loss = torch.ones(len(src[b, i+1:])) #Lossを算出する部分
            masks.append(torch.cat((mask, loss)))

    return torch.stack([r for r in result]).to(progress.get_device()), torch.stack([m for m in masks]).to(progress.get_device())


def _train_self_tuning(tokenizer: Tokenizer, save_directory, mortm_dataset, message: Messenger, vocab_size: int, num_epochs: int, weight: Tensor, progress: LearningProgress,
                       e_layer, d_layer, load_model_directory:str = None, train_dataset_split:float = 0.9,
                       num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1, is_save_training_progress=False,
                       position_length=2048, accumulation_steps=4, batch_size=16, num_workers=0, warmup_steps=4000, lr_param=1):

    train_size = int(train_dataset_split * len(mortm_dataset))
    val_size = len(mortm_dataset) - train_size
    train_dataset, val_dataset = random_split(mortm_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn
                              )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate_fn)

    print("Creating Model....")
    model = MORTM(vocab_size=vocab_size, progress=progress, num_heads=num_heads, e_layer=e_layer, d_layer=d_layer,
                  d_model=d_model, dim_feedforward=dim_feedforward, decoder_only=True,
                  dropout=dropout, position_length=position_length).to(progress.get_device())

    if load_model_directory is not None:
        model.load_state_dict(torch.load(load_model_directory))

    #criterion = ReinforceCrossEntropy(tokenizer=tokenizer, ignore_index=0, k=1, warmup=10, weight=weight.to(progress.get_device()))
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none").to(progress.get_device())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_param, betas=(0.9, 0.98), weight_decay=1e-6)  # オプティマイザを定義
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam_lr(d_model=d_model, warmup_steps=warmup_steps))

    print("Start training...")
    writer = SummaryWriter(save_directory + f"/runs/{time.time()}/")

    loss_val = None
    mail_bool = True
    all_count = 1
    for epoch in range(num_epochs):
        #criterion.step()
        try:
            print(f"epoch {epoch + 1} start....")
            count = 1
            epoch_loss = 0.0
            verification_loss = 0.0

            model.train()
            optimizer.zero_grad()

            for src in train_loader:  # seqにはbatch_size分の楽曲が入っている
                #print(f"learning sequence {count}")
                begin_time = time.time()
                correct, loss_mask = get_correct_src2(src, progress)
                src = src[:, :-1]

                padding_mask_in: Tensor = _get_padding_mask(src, progress)
                outputs: Tensor = model(src=src, input_padding_mask=padding_mask_in)

                outputs = outputs.view(-1, outputs.size(-1)).to(progress.get_device())
                correct = correct.reshape(-1).long()
                loss_mask = loss_mask.reshape(-1).long()
                loss = criterion(outputs, correct)  # 損失を計算
                loss = loss * loss_mask
                loss = loss.sum() / loss_mask.sum()
                epoch_loss += loss
                loss = loss / accumulation_steps
                loss.backward()  # 逆伝播

                if count % accumulation_steps == 0:  #実質バッチサイズは64である
                    progress.step_optimizer(optimizer, model, accumulation_steps)
                    scheduler.step()
                    torch.cuda.empty_cache()

                count += 1
                end_time = time.time()

                if mail_bool and message is not None:
                    _send_prediction_end_time(message, len(train_loader), begin_time, end_time, vocab_size, num_epochs,
                                              e_layer, num_heads, d_model, dim_feedforward, dropout, position_length)
                    mail_bool = False

                if (count + 1) % message.step_by_message_count == 0:
                    message.send_message("機械学習の途中経過について", f"Epoch {epoch + 1}/{num_epochs}の"
                                                                       f"learning sequence {count}結果は、\n {epoch_loss / count:.4f}でした。\n"
                                                                       f"また、検証データの損失は{verification_loss:.4f}となっています。\n以上です。")
                    #f"損失関数スケジューラーは{criterion.cs}です。")
                writer.flush()

                progress_bar(epoch, num_epochs, count, len(train_loader), epoch_loss / count, scheduler.get_last_lr(), verification_loss, criterion)

                if (count + 1) % (100000 / batch_size) == 0:
                    torch.save(model.state_dict(), f"{save_directory}/MORTM.train.{epoch}.{epoch_loss / count:.4f}_{count}.pth")
                    print("途中経過を保存しました。")

                if (count + 1) % (10000 / batch_size) == 0:
                    verification_loss = get_verification_loss(model, val_loader, criterion, progress)
                    writer.add_scalars("Train/Verification Loss", {"Train": epoch_loss / count,
                                                                   "Verification": verification_loss}, all_count)

                all_count += 1

            message.send_message("機械学習の途中経過について",
                                 f"Epoch {epoch + 1}/{num_epochs}の結果は、{epoch_loss / count:.4f}でした。\n"
                                 f"また、検証データの損失は{verification_loss:.4f}となっています。\n以上です。")
            #f"現在の損失関数スケジューラーの重みは{criterion.cs}となっています。")
            loss_val = epoch_loss / count
            writer.add_scalar('EpochLoss', epoch_loss / count, epoch)  # 損失値を記録

            if is_save_training_progress:
                torch.save(model.state_dict(), f"{save_directory}/MORTM.train.{epoch}.{epoch_loss / count:.4f}.pth") #エポック終了時に途中経過を保存
                print("途中経過を保存しました。")
        except torch.cuda.OutOfMemoryError or RuntimeError:
            torch.save(model.state_dict(), f"{save_directory}/MORTM.error_end.{epoch}.pth")
            message.send_message("オーバーフローしました・。", f"{epoch}エポック中にオーバーフローが発生しました。\n"
                                                              f"次のエポックに移行します。\n")
            #f"現在の損失関数スケジューラーの重みは{criterion.cs}となっています。")
    writer.close()

    return model, loss_val


def train_mortm(tokenizer, root_directory, save_directory, version: str, vocab_size: int, num_epochs: int, weight_directory,
                message: Messenger = _DefaultMessenger(), load_model_directory: str=None, train_dataset_split = 0.9,
                e_layer=15, num_heads=12, d_model=768, is_save_training_progress=False, lr_param=2e-1,
                dim_feedforward=3072, dropout=0.2, position_length=320, num_workers=0, warmup_steps=4000, src_mask_method: Callable[[Tensor], Tensor]=None,
                accumulation_steps=32, batch_size=1, progress: LearningProgress = _DefaultLearningProgress(), ):

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')

    print(f"ToDay is{datetime.date.today()}! start generating MORTEM_Model.{version}_{today_date}")

    directory, filename = find_npz_files(root_directory)
    train_data = _set_train_data(directory, filename, position_length, progress)

    try:

        with open(weight_directory, 'r') as file:
            model, loss = _train_self_tuning(tokenizer,save_directory, train_data, message, vocab_size, num_epochs, None, progress=progress,
                                             load_model_directory=load_model_directory,
                                             d_model=d_model,
                                             dim_feedforward=dim_feedforward,
                                             e_layer=e_layer,
                                             d_layer=e_layer,
                                             num_heads=num_heads,
                                             position_length=position_length,
                                             dropout=dropout,
                                             accumulation_steps=accumulation_steps,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             warmup_steps=warmup_steps,
                                             is_save_training_progress=is_save_training_progress,
                                             lr_param=lr_param,
                                             train_dataset_split= train_dataset_split
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
