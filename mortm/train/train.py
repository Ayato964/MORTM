import datetime
import json
import os
import random
import time

import torchaudio
import wandb
from einops import rearrange
import numpy as np

import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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


def reduce_tensor(tensor: Tensor, op=dist.ReduceOp.SUM) -> Tensor:
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=op)
    return rt


class MORTMTrainSet(AbstractTrainSet):
    def __init__(self, args: MORTMArgs, t_args: TrainArgs, tokenizer: Tokenizer, calc_val_loss_tokens, progress: LearningProgress, log_scale=False, project_name="", config=None,  load_directory=None, model_name=None):
        self.tokenizer = tokenizer
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        device = torch.device(f"cuda:{self.local_rank}")
        self.model = MORTM(progress=progress, args=args).to(device)

        if load_directory is not None:
            self.model.load_state_dict(torch.load(load_directory, map_location=device))

        # torch.compile は DDP でラップするより「前」に適用するのがベストプラクティスです。
        # これにより、通信部分を除いた純粋な計算グラフを最適化できます。
        self.model = torch.compile(
             self.model,
             fullgraph=False,
             dynamic=True 
        )

        # MoEのようにバッチによって特定のExpert（パラメータ）が全く使われないことがある構造では、
        # find_unused_parameters=True が必須です。
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        total_param, self.active_params = self.model.module.get_param()
        adam = torch.optim.Adam(self.model.parameters(), lr=t_args.lr_param)

        if self.local_rank == 0:
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

        super().__init__(criterion=MaskedCrossEntropyLoss(ignore_index=0).to(device),
                         optimizer=adam,
                         t_args=t_args,
                         m_args=args,
                         calc_val_loss_tokens=calc_val_loss_tokens)

        self.all_tokens = torch.tensor(0, device=device, dtype=torch.long)

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
        src = pack
        device = torch.device(f"cuda:{self.local_rank}")
        target: Tensor = src[:, 1:].to(device)
        mask = self.loss_mask(target)
        target = target.reshape(-1).long()
        mask = mask.reshape(-1).long()

        src = src[:, :-1].to(device)
        padding_mask_in: Tensor = _get_padding_mask(src, progress, device)

        if model.training:
            self.all_tokens += torch.sum(padding_mask_in == 1)

        input: Tensor = model(x=src, padding_mask=padding_mask_in, is_causal=True)
        input = input.view(-1, input.size(-1)).to(device)
        return input.to(device=device, dtype=torch.float32), target, mask

    def view_logs(self, epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens):
        # 全プロセスで進捗を確認できるように、Rank 0 以外のログも抑制を外します (デバッグ用)
        progress_bar_with_minibatch(self.local_rank, epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens)

    def optional_logging(self, val_loss, step):
        # 頻繁な通信を避けるため、ログ出力が必要な Rank 0 でのみ動作させるか、
        # あるいはここで明示的な同期が必要な場合のみ全員で入るようにします。
        # (学習を止めないよう、ここでは同期なしで Rank 0 のローカル値を暫定表示する形にします)
        if self.local_rank == 0:
            wandb.log({
                "axis/val_loss": val_loss,
                "axis/tokens": self.all_tokens.item(), # 同期なしのローカル値
                "axis/flops": 6 * self.active_params * self.all_tokens.item(),
                "trainer/global_step": step
            })

    def loss_mask(self, x: torch.Tensor) -> torch.Tensor:
        mgen_id = self.tokenizer.get("<MGEN>")
        cgen_id = self.tokenizer.get("<CGEN>")
        meta_id = self.tokenizer.get("<META>")
        is_start_token = ((x == mgen_id) | (x == cgen_id) | (x == meta_id)).long()

        cumulative_mask = torch.cumsum(is_start_token, dim=1)
        mask_x = (cumulative_mask > 0).long()
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

def _get_padding_mask(input_ids, progress: LearningProgress, device: torch.device):
    pad_id = (input_ids != 0).to(torch.float)
    padding_mask = pad_id.to(device)
    return padding_mask

def find_files(root_folder, extension: str):
    direc = []
    midi_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(extension):
                direc.append(dirpath + os.sep)
                midi_files.append(fname)
    
    # DDP環境での同一順序を保証するためソートする
    combined = sorted(zip(direc, midi_files))
    if not combined:
        return [], []
    direc, midi_files = zip(*combined)
    return list(direc), list(midi_files)

def _set_train_data(directory, datasets, mortm_datasets, *args):
    loss_count = 0
    count = 0
    dataset_length = 0
    loss_data = 0
    for i in range(len(datasets)):
        count += 1
        np_load_data = np.load(f"{directory[i]}/{datasets[i]}", allow_pickle=True)

        if len(np_load_data) > loss_data:
            dataset_length += mortm_datasets.add_data(np_load_data, *args)
        else:
            loss_count += 1
    return mortm_datasets

def _set_train_data_preloading(directory, datasets, mortm_datasets, *args):
    mortm_datasets.add_data(directory, datasets)
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

    # DDP環境での同一順序を保証するためソートする
    combined = sorted(zip(directories, file_names))
    if not combined:
        return [], []
    directories, file_names = zip(*combined)
    return list(directories), list(file_names)

def collate_fn(batch):
    src = pad_sequence(batch, batch_first=True, padding_value=0)
    return src

def collate_fn_with_tgt(batch):
    tgt_list = [item[1] for item in batch]
    src = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    tgt = torch.tensor(tgt_list, device=src.device)
    return src, tgt

def save_path_json(name, val_loader: DataLoader, save_directory, version):
    val_paths = []
    for v in val_loader:
        val_paths.append(v)
    with open(f"{save_directory}/{name}_paths_{version}.json", 'w') as f:
        json.dump(val_paths, f, indent=4)

def update_log(model, writer, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"params_mean/{name}", param.grad.mean(), global_step)
            writer.add_scalar(f"params_std/{name}", param.grad.std(), global_step)
            writer.add_scalar(f"Parameter Norm/{name}", param.grad.norm(), global_step)

def progress_bar(epoch, sum_epoch, sequence, batch_size, loss, lr, verif_loss):
    per = sequence / batch_size * 100
    block = int(per / 100 * 50)
    color_bar = "\033[32m"
    bar = f" {color_bar}{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"
    print(f"\r learning Epoch {epoch + 1}/{sum_epoch} [{bar}] {per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}", end="")

def progress_bar_with_minibatch(rank, epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens):
    big_per = seq_count / all_pac * 100
    block = int(big_per / 100 * 50)
    color_bar = "\033[32m"
    big_bar = f" {color_bar}{'#' * block}\033[31m{'-' * (50 - block)}\033[0m"

    mini_per = mini_seq_count / mini_seq_pac * 100
    mini_block = int(mini_per / 100 * 20)
    mini_bar = f"{color_bar}{'#' * mini_block}\033[31m{'-' * (20 - mini_block)} \033[0m"

    print(f"\r [Rank{rank}] learning Epoch {epoch + 1}/{sum_epoch} Package [{big_bar}] {big_per:.2f}%  Mini Package [{mini_bar}]  {mini_per:.2f}%  loss:{loss:.4f} Lr:{lr}  verification loss:{verif_loss: .4f}  Learning tokens:{tokens}", end="")


def get_inner_loader(dataset: Dataset, batch_size: int, collate_fn=None):
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True
    )
    return loader, sampler

def get_data_loader(t_args: TrainArgs, mortm_dataset: tuple | Dataset, shuffle=True, collate_fn=None):
    if isinstance(mortm_dataset, Dataset):
        train_size = int(t_args.train_dataset_split * len(mortm_dataset))
        val_size = len(mortm_dataset) - train_size
        # DDP環境では全プロセスで同じ分割になるように乱数シードを固定
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(mortm_dataset, [train_size, val_size], generator=generator)

        train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=shuffle)
        val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=0, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=t_args.big_batch_size, sampler=train_sampler,
                                  num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=t_args.big_batch_size, sampler=val_sampler,
                                num_workers=0, collate_fn=collate_fn)
    else:
        train_sampler = DistributedSampler(mortm_dataset[0], num_replicas=1, rank=0, shuffle=shuffle)
        val_sampler = DistributedSampler(mortm_dataset[1], num_replicas=1, rank=0, shuffle=False)

        train_loader = DataLoader(mortm_dataset[0], batch_size=t_args.big_batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(mortm_dataset[1], batch_size=t_args.big_batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader

def get_verification_loss(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, progress: LearningProgress,
                          trainer, train_args: TrainArgs,
                          coll_fn=None):
    model.eval()
    val_loss = torch.tensor(0.0, device=torch.device(f"cuda:{trainer.local_rank}"))
    all_count = torch.tensor(0.0, device=torch.device(f"cuda:{trainer.local_rank}"))

    with torch.no_grad():
        for pack in val_loader:
            pre_processing: Dataset = trainer.pre_processing(pack, progress)
            if len(pre_processing) == 0:
                continue
            inner_sampler = DistributedSampler(pre_processing, shuffle=False)
            loader = DataLoader(pre_processing, batch_size=train_args.batch_size, sampler=inner_sampler, shuffle=False, collate_fn=coll_fn)
            for pack2 in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r_pack = trainer.epoch_fc(model, pack2, progress)
                    loss = trainer.get_eval_loss(*r_pack)
                val_loss += loss
                all_count += 1

    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_count, op=dist.ReduceOp.SUM)

    model.train()
    if all_count.item() == 0:
        return 0.0
    return (val_loss / all_count).item()

def self_turing(model_name, train_args: TrainArgs, save_directory, trainer:AbstractTrainSet,
                train_loader: DataLoader, val_loader: DataLoader,
                message: Messenger, progress: LearningProgress,
                writer,  coll_fn=None):

    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler

    local_rank = trainer.local_rank
    world_size = trainer.world_size

    mail_bool = True
    epoch1_end = False
    all_count = 1
    verification_loss = 0.0

    for epoch in range(train_args.num_epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        try:
            count = 1
            epoch_loss = EpochObserver(1000)
            verification_loss = 0.0

            model.train()
            optimizer.zero_grad()

            for pack in train_loader:
                begin_time = time.time()
                pre_processing_dataset = trainer.pre_processing(pack, progress)
                inner_loader, inner_sampler = get_inner_loader(
                    pre_processing_dataset,
                    train_args.batch_size,
                    coll_fn
                )
                inner_sampler.set_epoch(epoch)
                
                # DDP環境では、データの偏りや型番違いによる速度差でバッチ数がズレてデッドロックすることがあります。
                # 全Rankのバッチ数を同期し、最小数に合わせて打ち切ることでこれを完全に防ぎます。
                local_len = len(inner_loader)
                if dist.is_initialized():
                    device = torch.device(f"cuda:{local_rank}")
                    len_tensor = torch.tensor([local_len], device=device)
                    dist.all_reduce(len_tensor, op=dist.ReduceOp.MIN)
                    synced_len = len_tensor.item()
                else:
                    synced_len = local_len

                mini_c = 0
                count += 1
                for pack2 in inner_loader:
                    # 同期されたバッチ数を超えたら終了（デッドロック防止）
                    if mini_c >= synced_len:
                        break
                    
                    mini_c += 1
                    all_count += 1

                    is_step_optimizer = mini_c % train_args.accumulation_steps == 0
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        r_pack = trainer.epoch_fc(model, pack2, progress)

                    loss = trainer.backward(train_args.accumulation_steps, is_step_optimizer, progress, train_args.lr_param, *r_pack)

                    reduced_loss = reduce_tensor(loss, op=dist.ReduceOp.SUM)
                    avg_loss = (reduced_loss / world_size).item()

                    epoch_loss.add(avg_loss)

                    # 毎バッチのトークン数同期 (reduce_tensor) はフリーズの原因になるため、
                    # 表示用にはローカルの値を使い、同期は検証前などにまとめて行います。
                    current_tokens = trainer.all_tokens.item()
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else train_args.lr_param

                    trainer.view_logs(epoch, train_args.num_epochs, count, len(train_loader), mini_c, len(inner_loader), epoch_loss.get(), current_lr, verification_loss, current_tokens)

                end_time = time.time()

                if local_rank == 0:
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
                    if writer is not None:
                        writer.flush()

                if trainer.is_need_calc_val():
                    if local_rank == 0 and writer is not None:
                        update_log(model, writer, all_count)

                    torch.cuda.empty_cache()
                    verification_loss = get_verification_loss(model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn)

                    if local_rank == 0 and writer is not None:
                        writer.add_scalars("Train/Verification Loss", {"Train": epoch_loss.get(), "Verification": verification_loss}, all_count)

                    trainer.optional_logging(verification_loss, all_count)

            if not epoch1_end:
                epoch1_end = True
                verification_loss = get_verification_loss(model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn)
                trainer.optional_logging(verification_loss, all_count)

            sync_tokens = reduce_tensor(trainer.all_tokens, op=dist.ReduceOp.SUM).item()
            if local_rank == 0:
                message.send_message(f"{model_name}の途中経過について",
                                     f"Epoch {epoch + 1}/{train_args.num_epochs}の結果は、{epoch_loss.get():.4f}でした。\n"
                                     f"また、検証データの損失は{verification_loss:.4f}となっています。\n　"
                                     f"また現在学習中のトークン数は{sync_tokens}です。\n 以上です。")

                if train_args.is_save_training_progress:
                    torch.save(model.module.state_dict(), f"{save_directory}/{model_name}.train.{epoch}.{verification_loss:.4f}.pth")

        except torch.cuda.OutOfMemoryError:
            if local_rank == 0:
                message.send_message("エラーが発生し、処理を中断しました",
                                     "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")

    return model, verification_loss


def _train(args, t_args, save_directory, trainer, version, today_date,
           message, train_loader, val_loader,
           progress, coll_fn=None):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    writer = None

    if local_rank == 0:
        save_path_json("eval", val_loader, save_directory, version)
        save_path_json("train", train_loader, save_directory, version)
        writer = SummaryWriter(save_directory + f"/runs/{version}_{today_date}/")

    try:
        model, loss = self_turing(f"{args.name}.{version}", t_args, save_directory, trainer,
                                         message=message,
                                         train_loader=train_loader, val_loader=val_loader,
                                         progress=progress,
                                         writer=writer,
                                         coll_fn=coll_fn
                                         )

        if local_rank == 0:
            message.send_message("機械学習終了のお知らせ",
                                 f"{args.name}.{version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {loss}")
            torch.save(model.module.state_dict(), f"{save_directory}/{args.name}.{version}_{loss}.pth")

        return model

    except torch.cuda.OutOfMemoryError:
        if local_rank == 0:
            message.send_message("エラーが発生し、処理を中断しました",
                                 "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")

def train_mortm(tokenizer, model_config: str, train_config: str, root_directory, save_directory, version: str,
                message: Messenger = _DefaultMessenger(), load_model_directory: str=None, eval_list_json: str = None,
                progress: LearningProgress = _DefaultLearningProgress(), log_scale=False,project_name=None):

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # DDP環境でのプロセス間の再現性と同期を確保するため、すべての乱数シードを固定
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定論的な動作を優先（速度は落ちる可能性があるが、デッドロック回避のため）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    args = MORTMArgs(json_directory=model_config)
    t_args = TrainArgs(json_directory=train_config)
    trainer = MORTMTrainSet(args, t_args, tokenizer, t_args.val_total_tokens, progress, load_directory=load_model_directory, project_name=project_name, config=model_config, log_scale=log_scale, model_name=version)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # デバッグ終了につき無効化
    today_date = datetime.date.today().strftime('%Y%m%d')

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

    mortm_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
    if eval_list_json is None:
        train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=t_args.shuffle)
    else:
        directory, filename = find_files_with_json(eval_list_json)
        val_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
        train_loader, val_loader = get_data_loader(t_args, (mortm_dataset, val_dataset), shuffle=t_args.shuffle)

    _train(args, t_args, save_directory, trainer,message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader,coll_fn=collate_fn,
           progress=progress)

    if dist.is_initialized():
        dist.destroy_process_group()

def train_custom(trainer: AbstractTrainSet, t_args, root_directory, save_directory, version: str,
                 message: Messenger = _DefaultMessenger(), eval_list_json: str = None,
                 progress: LearningProgress = _DefaultLearningProgress(), coll_fn=None):

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    today_date = datetime.date.today().strftime('%Y%m%d')

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

    mortm_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
    if eval_list_json is None:
        train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=True)
    else:
        directory, filename = find_files_with_json(eval_list_json)
        val_dataset = _set_train_data_preloading(directory, filename, PreLoadingDatasets(progress))
        train_loader, val_loader = get_data_loader(t_args, (mortm_dataset, val_dataset), shuffle=True)

    _train(trainer.args, t_args, save_directory, trainer, message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader, coll_fn=coll_fn,
           progress=progress)

    if dist.is_initialized():
        dist.destroy_process_group()