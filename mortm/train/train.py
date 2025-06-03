'''
MORTMの学習を行う際にこのモジュールを使います。
train_mortmメソッドを呼び出し、引数の型に合ったオブジェクトを代入してください。
最低でも、「データセット(Tokenizerで変換したもの)のディレクトリ」、「モデルの出力先のディレクトリ」,
「モデルのバージョン」,「ボキャブラリーサイズ」,「エポック回数」、「各トークンの出現回数のリスト」が必要です。
'''

import datetime
import json
import os
import time
from abc import abstractmethod
from typing import Optional, Callable, Any

import torchaudio
from einops import rearrange
import soundfile as sf


import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

from mortm.messager import Messenger, _DefaultMessenger
from mortm.models.modules.progress import LearningProgress, _DefaultLearningProgress
from .datasets import MORTM_SEQDataset, ClassDataSets, PreLoadingDatasets, TensorDataset
from mortm.models.mortm import MORTM, MORTMArgs
from mortm.models.bertm import BERTM
from mortm.models.v_mortm import V_MORTM, V_MORTMArgs
from .noam import noam_lr
from .epoch import EpochObserver
from .config import AbstractTrainSet, TrainArgs

from solo.adamw import AdamWQ

IS_DEBUG = False


class MORTMTrainSet(AbstractTrainSet):
    def __init__(self, args: MORTMArgs, progress: LearningProgress, load_directory=None):
        self.args = args
        self.model = MORTM(progress=progress, args=args).to(progress.get_device())
        if load_directory is not None:
            self.model.load_state_dict(torch.load(load_directory))

        adam = torch.optim.Adam(self.model.parameters(), lr=1e-1)

        super().__init__(criterion=nn.CrossEntropyLoss(ignore_index=0).to(progress.get_device()),
                        optimizer=adam,
                        scheduler=LambdaLR(optimizer=adam, lr_lambda=noam_lr(d_model=args.d_model, warmup_steps=4000)))

    def pre_processing(self, pack, progress):
        dt: DataLoader = pack
        mini_dataset = MORTM_SEQDataset(progress, self.args.position_length, self.args.min_length)
        for d in dt:
            np_load_data = np.load(d, allow_pickle=True)
            mini_dataset.add_data(np_load_data)

        return mini_dataset


    def epoch_fc(self, model, pack, progress):
        src = pack
        target: Tensor = src[:, 1:].to(progress.get_device())
        target = target.reshape(-1).long()

        src = src[:, :-1]
        padding_mask_in: Tensor = _get_padding_mask(src, progress)
        input: Tensor = model(src=src, input_padding_mask=padding_mask_in, src_is_causal=True)
        input = input.view(-1, input.size(-1)).to(progress.get_device())
        return input.to(device=progress.get_device(), dtype=torch.float32), target


class BERTMTrainSet(AbstractTrainSet):
    def __init__(self, args: MORTMArgs, progress: LearningProgress, load_directory=None):
        self.model = BERTM(progress=progress, args=args).to(progress.get_device())
        if load_directory is not None:
            self.model.load_state_dict(torch.load(load_directory))

        adam = torch.optim.Adam(self.model.parameters(), lr=5e-3, betas=(0.9, 0.98))

        super().__init__(criterion=nn.BCEWithLogitsLoss().to(progress.get_device()),
                         optimizer=adam,
                         scheduler=LambdaLR(optimizer=adam, lr_lambda=noam_lr(d_model=args.d_model, warmup_steps=4000)))


    def epoch_fc(self, model, pack, progress):
        src, tgt = pack
        target: Tensor = tgt.to(progress.get_device())
        src = src.to(progress.get_device())
        src_pad = _get_padding_mask(src, progress)

        out = model(src, input_padding_mask=src_pad)
        inputs = out.view(-1, out.size(-1)).to(progress.get_device())
        return inputs.squeeze(-1).to(dtype=torch.float32), target.to(dtype=torch.float32)


class V_MORTMTrainSet(AbstractTrainSet):
    def __init__(self, args: V_MORTMArgs, progress: LearningProgress, load_directory=None, split_time=10,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
                 ):
        self.split_time = split_time
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.model = V_MORTM(progress=progress, args=args).to(progress.get_device())
        if load_directory is not None:
            self.model.load_state_dict(torch.load(load_directory))

        adam = torch.optim.Adam(self.model.parameters(), lr=1e-1, betas=(0.9, 0.98))

        super().__init__(criterion=nn.MSELoss().to(progress.get_device()),
                         optimizer=adam,
                         scheduler=LambdaLR(optimizer=adam, lr_lambda=noam_lr(d_model=args.d_model, warmup_steps=4000)))

    def pre_processing(self, pack, progress):
        wav_set = pack
        comp = []
        for ws in wav_set:
            wav_np, sr = sf.read(ws, always_2d=True)
            wav_np = wav_np.T.astype("float32")       # shape: (ch, time)
            waveform = torch.from_numpy(wav_np)

            # 3) モノラル化
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # → [1, time]

            # 4) 分割長サンプル数の決定（必ず split_time 秒ごと）
            if self.split_time:
                seg_len = int(self.split_time * sr)
                total = waveform.shape[1]
                num_segments = (total + seg_len - 1) // seg_len  # ceil
            else:
                seg_len = waveform.shape[1]
                num_segments = 1

            # 5) メル変換器を一度だけ生成
            mel_tf = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )

            # 6) 各セグメントを切り出し、最後は無音でパディング
            for i in range(num_segments):
                start = i * seg_len
                end = start + seg_len
                if end <= waveform.shape[1]:
                    seg = waveform[:, start:end]
                else:
                    # 残り部分 + 無音パディング
                    rest = waveform[:, start:]
                    pad_len = end - waveform.shape[1]
                    pad = torch.zeros((waveform.shape[0], pad_len), dtype=waveform.dtype)
                    seg = torch.cat([rest, pad], dim=1)

                # mel: [1, n_mels, T]
                mel = mel_tf(seg)
                # squeeze → [n_mels, T]
                mel = mel.squeeze(0)
                # log1p
                logmel = torch.log1p(mel)
                comp.append(logmel)

        # 7) バッチ化
        datasets = TensorDataset(progress)
        datasets.add_data(comp)
        return datasets


    def epoch_fc(self, model, pack, progress):
        src: Tensor = pack

        src = rearrange(src, 'b d s -> b s d')

        target: Tensor = src[:, 1:, :]
        src = src[:, :-1, :]

        input: Tensor = model(src=src)
        input = rearrange(input, 'b s d -> b d s')
        target = rearrange(target, 'b s d -> b d s')
        return input.to(dtype=torch.float32), target


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


def _get_padding_mask(input_ids, progress: LearningProgress):
    # input_ids が Tensor であることを仮定
    pad_id = (input_ids != 0).to(torch.float)
    padding_mask = pad_id.to(progress.get_device())
    return padding_mask


def find_files(root_folder, extension: str):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(extension):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


# デバイスを取得
def _set_train_data(directory, datasets, mortm_datasets, *args):
    print("Starting load....")
    loss_count = 0
    count = 0
    dataset_length = 0
    loss_data = 0
    print(len(datasets))
    for i in range(len(datasets)):
        count += 1
        np_load_data = np.load(f"{directory[i]}/{datasets[i]}", allow_pickle=True)

        if len(np_load_data) > loss_data:
            dataset_length += mortm_datasets.add_data(np_load_data, *args)
            print(f"\r {count}/{len(datasets)} | Dataset Length:{dataset_length} | Load[{directory[i]}/{datasets[i]}]", end="")
        else:
            loss_count += 1
    print("load Successful!!")
    print(f"データセットの規模（曲数）：{len(datasets) - loss_count}")
    print("---------------------------------------")

    return mortm_datasets

def _set_train_data_preloading(directory, datasets, mortm_datasets, *args):
    print("Starting load....")
    datasets_length = 0
    mortm_datasets.add_data(directory, datasets)
    print("load Successful!!")
    return mortm_datasets


def collate_fn(batch):
    # バッチ内のテンソルの長さを揃える（パディングする）
    src = pad_sequence(batch, batch_first=True, padding_value=0)
    return src


def update_log(model, writer, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"params_mean/{name}", param.grad.mean(), global_step)
            writer.add_scalar(f"params_std/{name}", param.grad.std(), global_step)

            writer.add_scalar(f"Parameter Norm/{name}", param.grad.norm(), global_step)


def progress_bar(epoch, sum_epoch, sequence, batch_size, loss, lr, verif_loss):
    per = sequence / batch_size * 100
    block = int(per / 100 * 50)
    #color_bar = get_color(criterion)
    color_bar = "\033[32m"
    bar = f" {color_bar}{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"
    print(f"\r learning Epoch {epoch + 1}/{sum_epoch} [{bar}] {per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}", end="")


def progress_bar_with_minibatch(epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss):
    big_per = seq_count / all_pac * 100
    block = int(big_per / 100 * 50)
    color_bar = "\033[32m"
    big_bar = f" {color_bar}{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"

    mini_per = mini_seq_count / mini_seq_pac * 100
    mini_block = int(mini_per / 100 * 20)
    mini_bar = f"{color_bar}{'#' * mini_block}\033[31m{'-' * (20 - mini_block)} \033[0m"

    print(f"\r learning Epoch {epoch + 1}/{sum_epoch} Package [{big_bar}] {big_per:.2f}%  Mini Package [{mini_bar}]  {mini_per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}", end="")


def get_data_loader(t_args: TrainArgs, mortm_dataset, shuffle=True, collate_fn=None):
    train_size = int(t_args.train_dataset_split * len(mortm_dataset))
    val_size = len(mortm_dataset) - train_size
    train_dataset, val_dataset = random_split(mortm_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=t_args.big_batch_size, shuffle=shuffle,
                              num_workers=0, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=t_args.big_batch_size, shuffle=shuffle,
                            num_workers=0, collate_fn=collate_fn)
    return train_loader, val_loader


def get_verification_loss(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, progress: LearningProgress,
                          trainer, train_args: TrainArgs,
                          coll_fn=None):
    model.eval()
    val_loss = 0.0
    all_count = 0
    with torch.no_grad():
        for pack in val_loader:
            pre_processing: Dataset = trainer.pre_processing(pack, progress)
            loader = DataLoader(pre_processing, batch_size=train_args.batch_size, shuffle=True, collate_fn=coll_fn)
            for pack2 in loader:
                r_pack = trainer.epoch_fc(model, pack2, progress)
                loss = criterion(*r_pack)  # 損失を計算
                val_loss += loss.item()
                all_count += 1
    model.train()
    return val_loss / all_count


def self_turing(args, train_args: TrainArgs, save_directory, trainer:AbstractTrainSet,
                train_loader: DataLoader, val_loader: DataLoader,
                message: Messenger, progress: LearningProgress,
                writer,  coll_fn=None):
    print("Creating Trainer...")
    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler
    print("Start training...")

    mail_bool = True
    all_count = 1
    verification_loss = 0.0
    for epoch in range(train_args.num_epochs):
        #criterion.step()
        try:
            print(f"epoch {epoch + 1} start....")
            count = 1
            epoch_loss = EpochObserver(1000)
            verification_loss = 0.0

            model.train()
            optimizer.zero_grad()

            for pack in train_loader:
                pre_processing: Dataset = trainer.pre_processing(pack, progress)
                loader = DataLoader(pre_processing, batch_size=train_args.batch_size, shuffle=True, collate_fn=coll_fn)
                mini_c = 0
                count += 1
                for pack2 in loader:
                    mini_c += 1
                    all_count += 1
                    if mini_c % train_args.accumulation_steps == 0:  #実質バッチサイズは64である
                        progress.step_optimizer(optimizer, model, train_args.accumulation_steps)
                        if train_args.lr_param is None:
                            scheduler.step()
                        torch.cuda.empty_cache()

                    begin_time = time.time()

                    r_pack = trainer.epoch_fc(model, pack2, progress)

                    loss = criterion(*r_pack)  # 損失を計算
                    epoch_loss.add(loss.item())

                    loss = loss / train_args.accumulation_steps
                    loss.backward()  # 逆伝播

                    end_time = time.time()
                    progress_bar_with_minibatch(epoch, train_args.num_epochs, count, len(train_loader), mini_c, len(loader),  epoch_loss.get(), scheduler.get_last_lr() if train_args.lr_param is None else train_args.lr_param, verification_loss)

                    if mail_bool and message is not None:
                        _send_prediction_end_time(message, len(train_loader), begin_time, end_time, args.vocab_size, train_args.num_epochs,
                                                  args.e_layer, args.num_heads, args.d_model, args.dim_feedforward, args.dropout,
                                                  args.position_length)
                        mail_bool = False

                if (count + 1) % message.step_by_message_count == 0:
                    message.send_message("機械学習の途中経過について", f"Epoch {epoch + 1}/{train_args.num_epochs}の"
                                                                       f"learning sequence {count}結果は、\n {epoch_loss.get():.4f}でした。\n"
                                                                       f"また、検証データの損失は{verification_loss:.4f}となっています。\n以上です。")
                    #f"損失関数スケジューラーは{criterion.cs}です。")
                writer.flush()


                if (count + 1) % int(10000 / train_args.batch_size) == 0:
                    torch.save(model.state_dict(), f"{save_directory}/MORTM.train.{epoch}.{verification_loss:.4f}_{count}.pth")
                    print("途中経過を保存しました。")

                if (count + 1) % int(500 / train_args.batch_size) == 0:
                    print("検証損失を求めています")
                    torch.cuda.empty_cache()
                    verification_loss = get_verification_loss(model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn)
                    writer.add_scalars("Train/Verification Loss", {"Train": epoch_loss.get(),
                                                                   "Verification": verification_loss}, all_count)
                    update_log(model, writer, all_count)

            message.send_message("機械学習の途中経過について",
                                 f"Epoch {epoch + 1}/{train_args.num_epochs}の結果は、{epoch_loss.get():.4f}でした。\n"
                                 f"また、検証データの損失は{verification_loss:.4f}となっています。\n以上です。")
                #f"現在の損失関数スケジューラーの重みは{criterion.cs}となっています。")
            loss_val = verification_loss
            writer.add_scalar('EpochLoss', epoch_loss.get(), epoch)  # 損失値を記録

            if train_args.is_save_training_progress:
                torch.save(model.state_dict(), f"{save_directory}/{args.name}.train.{epoch}.{verification_loss:.4f}.pth") #エポック終了時に途中経過を保存
                print("途中経過を保存しました。")


        except  torch.cuda.OutOfMemoryError:
            message.send_message("エラーが発生し、処理を中断しました",
                                 "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")
        print("オーバーフローしました。")
    return model, verification_loss




def _train(args, t_args, save_directory, trainer, version, today_date,
           message, train_loader, val_loader,
           progress, coll_fn=None):
    try:
        writer = SummaryWriter(save_directory + f"/runs/{version}_{today_date}/")

        model, loss = self_turing(args, t_args, save_directory, trainer,
                                         message=message,
                                         train_loader=train_loader, val_loader=val_loader,
                                         progress=progress,
                                         writer=writer,
                                        coll_fn=coll_fn
                                         )  # 20エポック分機械学習を行う。

        message.send_message("機械学習終了のお知らせ",
                             f"{args.name}.{version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {loss}")

        torch.save(model.state_dict(), f"{save_directory}/{args.name}.{version}_{loss}.pth")  # できたモデルをセーブする

        return model

    except torch.cuda.OutOfMemoryError:
        message.send_message("エラーが発生し、処理を中断しました",
                             "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")
        print("オーバーフローしました。")


def train_mortm(model_config: str, train_config: str, root_directory, save_directory, version: str,
                message: Messenger = _DefaultMessenger(), load_model_directory: str=None,
                progress: LearningProgress = _DefaultLearningProgress(), ):
    args = MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)
    trainer = MORTMTrainSet(args, progress, load_directory=load_model_directory)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')

    print(f"ToDay is{datetime.date.today()}! start learning. {args.name}.Ver.{version}_{today_date}")

    directory, filename = find_files(root_directory, '.npz')
    mortm_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
    train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=True)

    _train(args, t_args, save_directory, trainer,message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader,coll_fn=collate_fn,
           progress=progress)

def train_bertm(model_config: str, train_config: str, human_dir, ai_dir, save_directory, version: str,
                message: Messenger = _DefaultMessenger(), load_model_directory: str=None,
                progress: LearningProgress = _DefaultLearningProgress()):
    def collate_fn(batch):

        src_list = [item[0] for item in batch]  # 各タプルのsrcを抽出
        tgt_list = [item[1] for item in batch]  # 各タプルのtgtを抽出

        tgt_list = torch.tensor(tgt_list)
        src = pad_sequence(src_list, batch_first=True, padding_value=0)
        return src, tgt_list

    args = MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)
    trainer = BERTMTrainSet(args, progress, load_directory=load_model_directory)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')

    print(f"ToDay is{datetime.date.today()}! start learning. {args.name}.Ver.{version}_{today_date}")

    directory, filename = find_files(human_dir, '.npz')
    mortm_dataset = _set_train_data(directory, filename, ClassDataSets(progress, args.position_length), 0)

    directory, filename = find_files(ai_dir, '.npz')
    mortm_dataset = _set_train_data(directory, filename, mortm_dataset, 1)
    train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=True, collate_fn=collate_fn)

    _train(args, t_args, save_directory, trainer,message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader,
           progress=progress)


def train_v_mortm(model_config: str, train_config: str, root_directory, save_directory, version: str,
                  message: Messenger = _DefaultMessenger(), load_model_directory: str=None,
                  progress: LearningProgress = _DefaultLearningProgress()):

    args = V_MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)
    trainer = V_MORTMTrainSet(args, progress, load_directory=load_model_directory)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')
    print(f"ToDay is{datetime.date.today()}! start learning. {args.name}.Ver.{version}_{today_date}")

    directory, filename = find_files(root_directory, '.wav')
    mortm_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
    train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=True)

    _train(args, t_args, save_directory, trainer, message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader,
           progress=progress)