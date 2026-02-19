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

import torchaudio
import wandb
from einops import rearrange
import soundfile as sf


import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

from mortm.utils.messager import Messenger, _DefaultMessenger
from mortm.models.modules.progress import LearningProgress, _DefaultLearningProgress
from .datasets import MORTM_SEQDataset, ClassDataSets, PreLoadingDatasets, TensorDataset, PianoRollDataset
from mortm.models.mortm import MORTM, MORTMArgs
from mortm.utils.pianoroll_convert import *
from .epoch import EpochObserver
from .config import AbstractTrainSet, TrainArgs
from .tokenizer import Tokenizer
from .utils.loss import MusicEntropyLoss, MaskedCrossEntropyLoss

IS_DEBUG = False



class MORTMTrainSet(AbstractTrainSet):
    def __init__(self, args: MORTMArgs, t_args, tokenizer, calc_val_loss_tokens, progress: LearningProgress, log_scale=False, project_name="", config=None,  load_directory=None, model_name=None):
        self.tokenizer: Tokenizer = tokenizer
        self.model = MORTM(progress=progress, args=args).to(progress.get_device())
        if load_directory is not None:
            self.model.load_state_dict(torch.load(load_directory))

        total_param, self.active_params, = self.model.get_param()
        adam = torch.optim.Adam(self.model.parameters(), lr=t_args.lr_param)

        with open(config, 'r') as f:
            data: dict = json.load(f)
            if log_scale:
                data['model_params'] = self.active_params
                data['total_params'] = total_param
                wandb.init(
                    project=project_name,
                    name=model_name,
                    config=data,
                    reinit=True
                )

        super().__init__(criterion=MaskedCrossEntropyLoss(ignore_index=0).to(progress.get_device()),
                         optimizer=adam,
                         t_args=t_args,
                         m_args=args,
                         calc_val_loss_tokens=calc_val_loss_tokens)

    def pre_processing(self, pack, progress):
        dt: DataLoader = pack
        mini_dataset = MORTM_SEQDataset(progress, self.args.position_length, self.args.min_length,
                                        is_random_delete_key=False, mask_sample_task=[self.tokenizer.get("<MGEN>")],
                                        program_token_id=[self.tokenizer.get("<INST_PIANO>"), self.tokenizer.get("<INST_SAX>")],
                                        system_tag=(self.tokenizer.get("<SYSTEM>"), self.tokenizer.get("<TAG_END>")), sampling_inst_max=2)
        for d in dt:
            np_load_data = np.load(d, allow_pickle=True)
            mini_dataset.add_data(np_load_data)

        return mini_dataset


    def epoch_fc(self, model, pack, progress):
        model: MORTM
        src = pack
        target: Tensor = src[:, 1:].to(progress.get_device())
        mask = self.loss_mask(target)
        target = target.reshape(-1).long()
        mask = mask.reshape(-1).long()

        src = src[:, :-1]
        padding_mask_in: Tensor = _get_padding_mask(src, progress)
        if model.training:
            self.all_tokens += torch.sum(padding_mask_in == 1)
        input: Tensor = model(x=src, padding_mask=padding_mask_in, is_causal=True)
        input = input.view(-1, input.size(-1)).to(progress.get_device())
        return input.to(device=progress.get_device(), dtype=torch.float32), target, mask

    def view_logs(self, epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens):
        progress_bar_with_minibatch(epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens)

    def optional_logging(self, val_loss, step):
        all_tokens = self.all_tokens.item()
        wandb.log({
            "axis/val_loss": val_loss,
            "axis/tokens": all_tokens,  # 横軸に使う重要な指標
            "axis/flops": 6 * self.active_params * all_tokens,  # 横軸に使う重要な指標
            "trainer/global_step": step
        })

    def loss_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        バッチ処理に対応した損失マスクを作成する。

        Args:
            x (torch.Tensor): 形状が (batch_size, sequence_length) の入力テンソル。

        Returns:
            torch.Tensor: 形状が (batch_size, sequence_length) のマスクテンソル。
        """
        mgen_id = self.tokenizer.get("<MGEN>")
        cgen_id = self.tokenizer.get("<CGEN>")
        meta_id = self.tokenizer.get("<META>")
        is_start_token = ((x == mgen_id) | (x == cgen_id) | (x == meta_id)).long()

        cumulative_mask = torch.cumsum(is_start_token, dim=1)

        # この比較も要素ごとに行われる。
        mask_x = (cumulative_mask > 0).long()
        # マスクを入力`x`と同じデバイスに転送する。
        return mask_x.to(x.device)



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
    """
    root_folder 以下を再帰的に探索し、
    拡張子が extension のファイルの
    ・ディレクトリ（末尾にパス区切り文字付き）
    ・ファイル名
    を別々のリストで返す
    """
    direc = []
    midi_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(extension):
                # ディレクトリには末尾に os.sep を付与しておく
                direc.append(dirpath + os.sep)
                midi_files.append(fname)
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
    mortm_datasets.add_data(directory, datasets)
    print("load Successful!!")
    return mortm_datasets

def find_files_with_json(path: str, min_idx=0):
    with open(path, "r") as f:
        json_data = json.load(f)

    path_list = []
    selected_data = json_data[min_idx:]

    for item in selected_data:
        if isinstance(item, list):
            path_list.extend(item)
        else:
            path_list.append(item)

    directories = [os.path.dirname(p) for p in path_list]
    file_names = [os.path.basename(p) for p in path_list]

    return directories, file_names

def collate_fn(batch):
    # バッチ内のテンソルの長さを揃える（パディングする）
    src = pad_sequence(batch, batch_first=True, padding_value=0)
    return src

def collate_fn_with_tgt(batch):
    # バッチ内のテンソルの長さを揃える（パディングする）
    tgt_list = [item[1] for item in batch]
    src = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    tgt = torch.tensor(tgt_list, device=src.device)
    return src, tgt

def save_path_json(name, val_loader: DataLoader, save_directory, version):
    """
    検証データのパスをJSONファイルに保存する。
    :param val_loader: DataLoaderオブジェクト
    :param save_directory: 保存先ディレクトリ
    :param version: バージョン名
    """
    val_paths = []
    counter = 0
    for v in val_loader:
        val_paths.append(v)
        counter += len(v)
    with open(f"{save_directory}/{name}_paths_{version}.json", 'w') as f:
        json.dump(val_paths, f, indent=4)
    print(f"Validation paths saved to {save_directory}/{name}_{version}.json   All Count = {counter}")


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


def progress_bar_with_minibatch(epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens):
    big_per = seq_count / all_pac * 100
    block = int(big_per / 100 * 50)
    color_bar = "\033[32m"
    big_bar = f" {color_bar}{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"

    mini_per = mini_seq_count / mini_seq_pac * 100
    mini_block = int(mini_per / 100 * 20)
    mini_bar = f"{color_bar}{'#' * mini_block}\033[31m{'-' * (20 - mini_block)} \033[0m"

    print(f"\r learning Epoch {epoch + 1}/{sum_epoch} Package [{big_bar}] {big_per:.2f}%  Mini Package [{mini_bar}]  {mini_per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}  Learning tokens:{tokens}", end="")


def get_data_loader(t_args: TrainArgs, mortm_dataset: tuple | Dataset, shuffle=True, collate_fn=None):
    if isinstance(mortm_dataset, Dataset):
        train_size = int(t_args.train_dataset_split * len(mortm_dataset))
        val_size = len(mortm_dataset) - train_size
        train_dataset, val_dataset = random_split(mortm_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=t_args.big_batch_size, shuffle=shuffle,
                                  num_workers=0, collate_fn=collate_fn)

        val_loader = DataLoader(val_dataset, batch_size=t_args.big_batch_size, shuffle=shuffle,
                                num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(mortm_dataset[0], batch_size=t_args.big_batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(mortm_dataset[1], batch_size=t_args.big_batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)

    print(f"All Size:{len(mortm_dataset)} Train Size:{len(train_loader)} Val Size:{len(val_loader)}")
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r_pack = trainer.epoch_fc(model, pack2, progress)
                    loss = trainer.get_eval_loss(*r_pack)  # 損失を計算
                val_loss += loss.item()
                all_count += 1
    model.train()
    return val_loss / all_count


def self_turing(model_name, train_args: TrainArgs, save_directory, trainer:AbstractTrainSet,
                train_loader: DataLoader, val_loader: DataLoader,
                message: Messenger, progress: LearningProgress,
                writer,  coll_fn=None):
    print("Creating Trainer...")

    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler
    print(f"Start training...")
    print(f"検証損失計算回数:{len(train_loader) // 20}")

    mail_bool = True
    epoch1_end = False
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
                begin_time = time.time()
                pre_processing: Dataset = trainer.pre_processing(pack, progress)
                loader = DataLoader(pre_processing, batch_size=train_args.batch_size, shuffle=train_args.shuffle, collate_fn=coll_fn)
                mini_c = 0
                count += 1
                for pack2 in loader:
                    mini_c += 1
                    all_count += 1

                    is_step_optimizer = mini_c % train_args.accumulation_steps == 0
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        r_pack = trainer.epoch_fc(model, pack2, progress)

                    loss = trainer.backward(train_args.accumulation_steps, is_step_optimizer, progress, train_args.lr_param, *r_pack)

                    epoch_loss.add(loss.item())


                    trainer.view_logs(epoch, train_args.num_epochs, count, len(train_loader), mini_c, len(loader),  epoch_loss.get(), scheduler.get_last_lr() if scheduler is not None else train_args.lr_param, verification_loss, trainer.all_tokens)

                end_time = time.time()
                if mail_bool:
                    t = end_time - begin_time
                    end_time_progress = (t * len(train_loader) * train_args.num_epochs) / 3600
                    message.send_message(
                        "学習開始のお知らせ",
                        f"{model_name}の学習が開始されました。"
                            f"\n\n シーケンスの1回目の処理が終了しました。かかった時間は{t:.1f}秒でした。\n"
                            f"終了見込み時間は{end_time_progress:.2f}時間です"
                    )
                    mail_bool = False
                writer.flush()

                if trainer.is_need_calc_val():
                    update_log(model, writer, all_count)
                    print("検証損失を求めています")
                    torch.cuda.empty_cache()
                    verification_loss = get_verification_loss(model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn)

                    writer.add_scalars("Train/Verification Loss", {"Train": epoch_loss.get(),
                                                                   "Verification": verification_loss}, all_count)
                    trainer.optional_logging(verification_loss, all_count)
            if not epoch1_end:
                epoch1_end = True
                print("１エポック当たりのトークン数：", trainer.all_tokens)
                verification_loss = get_verification_loss(model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn)
                trainer.optional_logging(verification_loss, all_count)

            message.send_message(f"{model_name}の途中経過について",
                                 f"Epoch {epoch + 1}/{train_args.num_epochs}の結果は、{epoch_loss.get():.4f}でした。\n"
                                 f"また、検証データの損失は{verification_loss:.4f}となっています。\n　"
                                 f"また現在学習中のトークン数は{trainer.all_tokens}です。\n 以上です。")
                #f"現在の損失関数スケジューラーの重みは{criterion.cs}となっています。")

            if train_args.is_save_training_progress:
                torch.save(model.state_dict(), f"{save_directory}/{model_name}.train.{epoch}.{verification_loss:.4f}.pth") #エポック終了時に途中経過を保存
                print("途中経過を保存しました。")


        except  torch.cuda.OutOfMemoryError:
            message.send_message("エラーが発生し、処理を中断しました",
                                 "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")
            print("オーバーフローしました。")
    return model, verification_loss


def _train(args, t_args, save_directory, trainer, version, today_date,
           message, train_loader, val_loader,
           progress, coll_fn=None):
    save_path_json("eval", val_loader, save_directory, version)
    save_path_json("train", train_loader, save_directory, version)
    try:
        writer = SummaryWriter(save_directory + f"/runs/{version}_{today_date}/")

        model, loss = self_turing(f"{args.name}.{version}", t_args, save_directory, trainer,
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



def train_mortm(tokenizer, model_config: str, train_config: str, root_directory, save_directory, version: str,
                message: Messenger = _DefaultMessenger(), load_model_directory: str=None, eval_list_json: str = None,
                progress: LearningProgress = _DefaultLearningProgress(), log_scale=False,project_name=None):
    args = MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)
    trainer = MORTMTrainSet(args, t_args, tokenizer, t_args.val_total_tokens, progress, load_directory=load_model_directory, project_name=project_name, config=model_config, log_scale=log_scale, model_name=version)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')

    print(f"ToDay is{datetime.date.today()}! start learning. {args.name}.Ver.{version}_{today_date}")
    print(f"Need to calc val loss tokens:{t_args.val_total_tokens}")

    if isinstance(root_directory, str):
        if root_directory.endswith(".json"):
            directory, filename = find_files_with_json(root_directory)
        else:
            directory, filename = find_files(root_directory, '.npz')
    elif isinstance(root_directory, tuple):
        directory = []
        filename = []
        for r in root_directory:
            if r.endswith(".json"):
                d, f = find_files_with_json(r)
            else:
                d, f = find_files(r, '.npz')
            directory.extend(d)
            filename.extend(f)
    else:
        directory, filename = root_directory
    print("データセットの規模：", len(filename))
    mortm_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
    if eval_list_json is None:
        train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=t_args.shuffle)
    else:
        print("検証データセットが指定されました。")
        directory, filename = find_files_with_json(eval_list_json)
        val_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
        train_loader, val_loader = get_data_loader(t_args, (mortm_dataset, val_dataset), shuffle=t_args.shuffle)


    _train(args, t_args, save_directory, trainer,message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader,coll_fn=collate_fn,
           progress=progress)


def train_custom(trainer: AbstractTrainSet, t_args, root_directory, save_directory, version: str,
                 message: Messenger = _DefaultMessenger(), eval_list_json: str = None,
                 progress: LearningProgress = _DefaultLearningProgress(), coll_fn=None):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')
    print(f"ToDay is{datetime.date.today()}! start learning. {trainer.args.name}.Ver.{version}_{today_date}")

    if isinstance(root_directory, str):
        if root_directory.endswith(".json"):
            directory, filename = find_files_with_json(root_directory)
        else:
            directory, filename = find_files(root_directory, '.npz')
    elif isinstance(root_directory, tuple):
        directory = []
        filename = []
        for r in root_directory:
            if r.endswith(".json"):
                d, f = find_files_with_json(r)
            else:
                d, f = find_files(r, '.npz')
            directory.extend(d)
            filename.extend(f)
    else:
        directory, filename = root_directory
    print("データセットの規模：", len(filename))
    mortm_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
    if eval_list_json is None:
        train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=True)
    else:
        print("検証データセットが指定されました。")
        directory, filename = find_files_with_json(eval_list_json)
        val_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
        train_loader, val_loader = get_data_loader(t_args, (mortm_dataset, val_dataset), shuffle=True)

    _train(trainer.args, t_args, save_directory, trainer, message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader, coll_fn=coll_fn,
           progress=progress)