import datetime
import json
import time

from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from .train import find_npz_files
from .datasets import ClassDataSets
from .train import  _get_padding_mask, _send_prediction_end_time, progress_bar, update_log
from mortm.models.modules.progress import LearningProgress, _DefaultLearningProgress
from mortm.models.bertm import BERTM
from torch.utils.data.dataset import random_split
from torch.nn import CrossEntropyLoss
from mortm.models.mortm import MORTMArgs
from .noam import noam_lr
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from .epoch import EpochObserver
import numpy as np

# デバイスを取得
def _set_train_data(directory, datasets, mortm_datasets, v):
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
            dataset_length += mortm_datasets.add_data(np_load_data, v)
            print(f"\r {count}/{len(datasets)} | Dataset Length:{dataset_length} | Load[{directory[i]}/{datasets[i]}]", end="")
        else:
            loss_count += 1
    print("load Successful!!")
    print(f"データセットの規模（曲数）：{len(datasets) - loss_count}")
    print("---------------------------------------")

    return mortm_datasets


def collate_fn(batch):

    src_list = [item[0] for item in batch]  # 各タプルのsrcを抽出
    tgt_list = [item[1] for item in batch]  # 各タプルのtgtを抽出

    tgt_list = torch.tensor(tgt_list)
    src = pad_sequence(src_list, batch_first=True, padding_value=0)
    return src, tgt_list

def get_verification_loss(model: BERTM, val_loader: DataLoader, criterion: CrossEntropyLoss, progress: LearningProgress):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src, tgt in val_loader:

            tgt: Tensor = tgt.to(progress.get_device())
            #src = torch.cat([torch.full((src.size(0), 1), 399, dtype=src.dtype, device=src.device), src], dim=1)

            padding_mask_in: Tensor = _get_padding_mask(src, progress)

            out = model(src, input_padding_mask=padding_mask_in)

            outputs = out.view(-1, out.size(-1)).to(progress.get_device())

            loss = criterion(outputs, tgt)  # 損失を計算
            val_loss += loss.item()
    model.train()
    return val_loss / len(val_loader)

def train_epoch(epoch, batch, args: MORTMArgs, train_dataset, val_dataset, message, save_directory, writer,
                progress: LearningProgress, lr_param, warmup_steps,  accumulation_steps, is_save_training_progress):

    model = BERTM(args=args, progress=progress).to(progress.get_device())
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1 if lr_param is None else lr_param, betas=(0.9, 0.98))  # オプティマイザを定義

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn)

    if lr_param is None:
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam_lr(d_model=args.d_model, warmup_steps=warmup_steps))
    else:
        scheduler = None

    loss_val = None
    mail_bool = True
    all_count = 1
    for ep in range(epoch):
        try:
            print(f"epoch {ep + 1} start....")
            count = 1
            epoch_loss = EpochObserver(1000)
            verification_loss = 0.0

            model.train()
            optimizer.zero_grad()
            for src, tgt in train_loader:
                begin_time = time.time()
                tgt: Tensor = tgt.to(progress.get_device())
                src = src.to(progress.get_device())
    #            src = torch.cat([torch.full((src.size(0), 1), 399, dtype=src.dtype, device=src.device), src], dim=1)
                src_pad = _get_padding_mask(src, progress)

                out = model(src, input_padding_mask=src_pad)


                outputs = out.view(-1, out.size(-1)).to(progress.get_device())
                loss = criterion(outputs, tgt)  # 損失を計算
                epoch_loss.add(loss.item())

                loss = loss / accumulation_steps
                loss.backward()  # 逆伝播


                if count % accumulation_steps == 0:  #実質バッチサイズは64である
                    progress.step_optimizer(optimizer, model, accumulation_steps)
                    if lr_param is None:
                        scheduler.step()
                    torch.cuda.empty_cache()

                count += 1
                end_time = time.time()

                if mail_bool and message is not None:
                    _send_prediction_end_time(message, len(train_loader), begin_time, end_time, args.vocab_size, epoch,
                                              args.e_layer, args.num_heads, args.d_model, args.dim_feedforward, args.dropout,
                                              args.position_length)
                    mail_bool = False

                if (count + 1) % message.step_by_message_count == 0:
                    message.send_message("機械学習の途中経過について", f"Epoch {ep + 1}/{epoch}の"
                                                                       f"learning sequence {count}結果は、\n {epoch_loss.get():.4f}でした。\n"
                                                                       f"また、検証データの損失は{verification_loss:.4f}となっています。\n以上です。")
                    #f"損失関数スケジューラーは{criterion.cs}です。")
                writer.flush()

                progress_bar(ep, epoch, count, len(train_loader), epoch_loss.get(), scheduler.get_last_lr() if lr_param is None else lr_param, verification_loss)
                if (count + 1) % int(100000 / batch) == 0:
                    torch.save(model.state_dict(), f"{save_directory}/MORTM.train.{ep}.{verification_loss:.4f}_{count}.pth")
                    print("途中経過を保存しました。")

                if (count + 1) % int(10000 / batch) == 0:
                    print("検証損失を求めています")
                    torch.cuda.empty_cache()
                    verification_loss = get_verification_loss(model, val_loader, criterion, progress)
                    writer.add_scalars("Train/Verification Loss", {"Train": epoch_loss.get(),
                                                                   "Verification": verification_loss}, all_count)
                update_log(model, writer, all_count)
                all_count += 1

            message.send_message("機械学習の途中経過について",
                                 f"Epoch {ep + 1}/{epoch}の結果は、{epoch_loss.get():.4f}でした。\n"
                                 f"また、検証データの損失は{verification_loss:.4f}となっています。\n以上です。")
            writer.add_scalar('EpochLoss', epoch_loss.get(), epoch)  # 損失値を記録

            if is_save_training_progress:
                torch.save(model.state_dict(), f"{save_directory}/BERTM.train.{ep}.{verification_loss:.4f}.pth") #エポック終了時に途中経過を保存
                print("途中経過を保存しました。")
        except torch.cuda.OutOfMemoryError or RuntimeError:
            torch.save(model.state_dict(), f"{save_directory}/MORTM.error_end.{epoch}.pth")
            message.send_message("オーバーフローしました・。", f"{epoch}エポック中にオーバーフローが発生しました。\n"
                                                              f"次のエポックに移行します。\n")
        writer.close()
    return model, loss_val






def train_bertm(human_dir, ai_dir, args_dir, save_directory, epoch, batch_size, warmup_steps, message,
                accumlation_steps, version,
                train_split, lr_param=None, is_save_training_progress=False,  progress = _DefaultLearningProgress()):

    args = MORTMArgs(args_dir)
    today_date = datetime.date.today().strftime('%Y%m%d')
    datasets = ClassDataSets(progress=progress, positional_length=args.position_length)

    directory_human, file_name_human = find_npz_files(human_dir)
    directory_ai, file_name_ai = find_npz_files(ai_dir)

    datasets = _set_train_data(directory_human, file_name_human, datasets, 0)
    datasets = _set_train_data(directory_ai, file_name_ai, datasets, 1)

    print(f"検証データを抽出しています。")
    train_size = int(train_split * len(datasets))
    val_size = len(datasets) - train_size
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    writer = SummaryWriter(save_directory + f"/runs/{version}_{today_date}/")
    print(len(train_dataset))
    model, val_loss = train_epoch(epoch=epoch, batch=batch_size, args=args,
                                  train_dataset=train_dataset, val_dataset=val_dataset,
                                  message=message, save_directory=save_directory, writer=writer,
                                  progress=progress, lr_param=lr_param, warmup_steps=warmup_steps,
                                  accumulation_steps=accumlation_steps,
                                  is_save_training_progress=is_save_training_progress)
    message.send_message("機械学習終了のお知らせ",
                         f"BERTM.{version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {val_loss}")


    torch.save(model.state_dict(), f"{save_directory}/BERTM.{version}_{val_loss}.pth")  # できたモデルをセーブする

    return model



