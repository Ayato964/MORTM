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

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from .messager import Messenger, _DefaultMessenger
from .progress import LearningProgress, _DefaultLearningProgress
from .datasets import MORTM_DataSets
from .mortm import MORTM
from .noam import noam_lr

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
    if not IS_DEBUG:
        print("Generating TrainData.....")
        t_data = MORTM_DataSets(progress)
        for dataset in datasets:
            print(f"Load [{directory + dataset}]")
            np_load_data = np.load(directory + dataset)
            train_data = None
            for i in range(len(np_load_data)):

                np_data = np.expand_dims(np_load_data[f'arr_{i}'], axis=0)[0]
                print(np_data.shape)

                if train_data is None:
                    train_data = np_data
                else:
                    train_data = np.concatenate((train_data, np_data), axis=0)
            print(f"最初の５音:{train_data[-1][0:25]}")
            t_data.add_data(train_data)
        print(f"Token size: {sum(len(sub) for sub in t_data.musics_seq)} ")
        print("----------------------------------")
        #t_data.split_seq_data()
        #t_data.set_padding()
        t_data.set_train_data()

        return t_data
    else:
        np_load_data = np.load(directory + datasets[0])
        print(np_load_data[f'arr_{3}'])
    return None

def _get_padding_mask(input_ids, progress: LearningProgress):
    # input_ids が Tensor であることを仮定
    pad_id = (input_ids != 0).to(torch.float)
    padding_mask = pad_id.to(progress.get_device())
    return padding_mask


def _train(save_directory, ayato_dataset, message: Messenger, vocab_size: int, num_epochs: int, weight: Tensor, progress: LearningProgress, trans_layer=6,
           num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1, is_save_training_progress=False,
           position_length=2048, accumulation_steps=4, batch_size=16, num_workers=0, warmup_steps=4000):

    loader = DataLoader(ayato_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    print("Creating Model....")
    model = MORTM(vocab_size=vocab_size, progress=progress, trans_layer=trans_layer, num_heads=num_heads,
                  d_model=d_model, dim_feedforward=dim_feedforward,
                  dropout=dropout, position_length=position_length).to(progress.get_device())

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight.to(progress.get_device())).to(progress.get_device())  # 損失関数を定義
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), weight_decay=0.01)  # オプティマイザを定義
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam_lr(d_model=d_model, warmup_steps=warmup_steps))

    print("Start training...")

    loss_val = None
    mail_bool = True
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} start....")
        print(f"batch size :{len(loader)}")
        count = 1
        epoch_loss = 0.0

        model.train()
        optimizer.zero_grad()

        for input_ids, targets in loader:  # seqにはbatch_size分の楽曲が入っている
            print(f"learning sequence {count}")
            begin_time = time.time()
            input_ids = input_ids.to(progress.get_device())
            targets = targets.to(progress.get_device())
            #inputs_mask = model.mortm_X.generate_square_subsequent_mask(input_ids.shape[1]).to(device)
            #targets_mask = model.transformer.generate_square_subsequent_mask(targets.shape[1]).to(progress.get_device())
            padding_mask_in: Tensor = _get_padding_mask(input_ids, progress)
            padding_mask_tgt: Tensor = _get_padding_mask(targets, progress)

            output = model(input_ids, targets, None, None, padding_mask_in, padding_mask_tgt)

            outputs = output.view(-1, output.size(-1)).to(progress.get_device())
            targets = targets.view(-1).to(progress.get_device()).long()
            loss = criterion(outputs, targets)  # 損失を計算
            loss.backward()  # 逆伝播
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if count % accumulation_steps == 0:  #実質バッチサイズは64である
                progress.step_optimizer(optimizer)
                scheduler.step()
                print("Optimizerを更新しました。")
                print(f"学習率：{scheduler.get_last_lr()}")

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
            print(loss.item())

        message.send_message("機械学習の途中経過について",
                                 f"Epoch {epoch + 1}/{num_epochs}の結果は、{epoch_loss / count:.4f}でした。")
        loss_val = epoch_loss / count

        if is_save_training_progress:
            torch.save(model.state_dict(), f"{save_directory}/MORTM.train.{epoch}.{epoch_loss / count:.4f}.pth") #エポック終了時に途中経過を保存
            print("途中経過を保存しました。")

    return model, loss_val


def train_mortm(dataset_directory, save_directory, version: str, vocab_size: int, num_epochs: int, weight_directory,
                message: Messenger = _DefaultMessenger(),
                trans_layer=12, num_heads=8, d_model=1024, is_save_training_progress=False,
                dim_feedforward=2048, dropout=0.2, position_length=2048, num_workers=0, warmup_steps=4000,
                accumulation_steps=4, batch_size=16, progress: LearningProgress = _DefaultLearningProgress()):
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
                    weights.append(1.0 / freq)
            # テンソルに変換
            weight_tensor = torch.tensor(weights)
            weight_tensor = weight_tensor / weight_tensor.sum()

            print(weight_tensor[weight_tensor.argmax(dim=-1)], weight_tensor[weight_tensor.argmin(dim=-1)])

        model, loss = _train(save_directory, train_data, message, vocab_size, num_epochs, weight_tensor, progress=progress,
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
                             is_save_training_progress=is_save_training_progress
                             )  # 20エポック分機械学習を行う。

        message.send_message("機械学習終了のお知らせ",
                                 f"MORTM.{version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {loss}")

        torch.save(model.state_dict(), f"{save_directory}/MORTM.{version}_{loss}.pth")  # できたモデルをセーブする

        return model

    except torch.cuda.OutOfMemoryError:
        message.send_message("エラーが発生し、処理を中断しました",
                                 "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")
    pass

