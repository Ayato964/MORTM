import datetime
import json
import math
import os
import random
import time
from typing import Optional

import torchaudio
import wandb
from einops import rearrange
import numpy as np
import gc

import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from torch.utils.data import DataLoader, random_split, Subset
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

def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _build_distributed_sampler(dataset: Dataset, shuffle: bool):
    if _is_dist_ready():
        return DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
    return None


def _sync_min_loader_length(local_len: int, device: torch.device) -> int:
    if not _is_dist_ready():
        return local_len

    len_tensor = torch.tensor([local_len], device=device, dtype=torch.long)
    dist.all_reduce(len_tensor, op=dist.ReduceOp.MIN)
    return int(len_tensor.item())


def _allocate_counts(total_size: int, weights: list[int]) -> list[int]:
    if total_size <= 0:
        raise ValueError("total_size must be positive.")

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must contain at least one positive value.")

    raw_counts = [(total_size * weight / total_weight) if weight > 0 else 0.0 for weight in weights]
    counts = [math.floor(value) for value in raw_counts]
    remain = total_size - sum(counts)

    if remain > 0:
        order = sorted(
            range(len(raw_counts)),
            key=lambda idx: (raw_counts[idx] - counts[idx], weights[idx]),
            reverse=True,
        )
        for idx in order[:remain]:
            counts[idx] += 1

    return counts


def _set_loader_epoch(loader: DataLoader, epoch: int):
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)


def _extract_paths_and_dataset_ids_from_pack(pack) -> tuple[list[str], list[int]]:
    if (
        isinstance(pack, (list, tuple))
        and len(pack) == 2
        and isinstance(pack[0], (list, tuple))
    ):
        paths = [str(path) for path in pack[0]]
        dataset_ids_raw = pack[1]

        if isinstance(dataset_ids_raw, torch.Tensor):
            dataset_ids = [int(v) for v in dataset_ids_raw.tolist()]
        else:
            dataset_ids = [int(v) for v in dataset_ids_raw]

        return paths, dataset_ids

    if isinstance(pack, str):
        return [pack], [0]

    raise TypeError(f"Unsupported pack type: {type(pack)}")


def _extract_all_paths_from_dataset(dataset) -> list[str]:
    # PreLoadingDatasets
    if hasattr(dataset, "src_list"):
        return list(dataset.src_list)

    # random_split() が返す Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_paths = _extract_all_paths_from_dataset(dataset.dataset)
        return [base_paths[i] for i in dataset.indices]

    raise TypeError(f"Unsupported dataset type for path export: {type(dataset)}")


def _extract_all_dataset_ids_from_dataset(dataset) -> list[int]:
    if hasattr(dataset, "dataset_ids"):
        return list(dataset.dataset_ids)

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_ids = _extract_all_dataset_ids_from_dataset(dataset.dataset)
        return [base_ids[i] for i in dataset.indices]

    raise TypeError(f"Unsupported dataset type for dataset id export: {type(dataset)}")


def _extract_all_dataset_names_from_dataset(dataset) -> list[str]:
    if hasattr(dataset, "dataset_names"):
        return list(dataset.dataset_names)

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_names = _extract_all_dataset_names_from_dataset(dataset.dataset)
        return [base_names[i] for i in dataset.indices]

    raise TypeError(f"Unsupported dataset type for dataset name export: {type(dataset)}")


def _supports_dataset_ids(dataset) -> bool:
    try:
        _extract_all_dataset_ids_from_dataset(dataset)
        return True
    except TypeError:
        return False


def _group_local_indices_by_dataset_id(dataset) -> dict[int, list[int]]:
    grouped_indices: dict[int, list[int]] = {}
    for local_idx, dataset_id in enumerate(_extract_all_dataset_ids_from_dataset(dataset)):
        grouped_indices.setdefault(int(dataset_id), []).append(local_idx)
    return grouped_indices


def _compute_split_sizes(total_size: int, train_ratio: float) -> tuple[int, int]:
    train_size = int(train_ratio * total_size)

    if total_size > 0 and train_ratio > 0.0 and train_size == 0:
        train_size = 1
    if total_size > 1 and train_ratio < 1.0 and train_size >= total_size:
        train_size = total_size - 1

    val_size = total_size - train_size
    return train_size, val_size


def _split_dataset_by_dataset_id(dataset: Dataset, train_ratio: float) -> tuple[Subset, Subset]:
    grouped_indices = _group_local_indices_by_dataset_id(dataset)
    generator = torch.Generator().manual_seed(42)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for dataset_id in sorted(grouped_indices):
        indices = grouped_indices[dataset_id]
        shuffled_positions = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[pos] for pos in shuffled_positions]
        train_size, _ = _compute_split_sizes(len(shuffled_indices), train_ratio)
        train_indices.extend(shuffled_indices[:train_size])
        val_indices.extend(shuffled_indices[train_size:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class DatasetBatchAllocationDistributedBatchSampler:
    def __init__(self, dataset: Dataset, batch_size: int, dataset_batch_allocation: list[int], shuffle: bool):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_batch_allocation = [int(v) for v in dataset_batch_allocation]
        self.shuffle = shuffle
        self.epoch = 0
        self.num_replicas = dist.get_world_size() if _is_dist_ready() else 1
        self.rank = dist.get_rank() if _is_dist_ready() else 0
        self.grouped_indices = _group_local_indices_by_dataset_id(dataset)
        self.per_rank_batch_counts = _allocate_counts(self.batch_size, self.dataset_batch_allocation)
        self.global_batch_counts = [count * self.num_replicas for count in self.per_rank_batch_counts]

        max_dataset_id = max(self.grouped_indices.keys(), default=-1)
        if len(self.dataset_batch_allocation) <= max_dataset_id:
            raise ValueError(
                f"dataset_batch_allocation length ({len(self.dataset_batch_allocation)}) "
                f"does not match available datasets ({max_dataset_id + 1})."
            )

        for dataset_id, batch_count in enumerate(self.dataset_batch_allocation):
            if batch_count > 0 and len(self.grouped_indices.get(dataset_id, [])) == 0:
                raise ValueError(
                    f"dataset_batch_allocation[{dataset_id}]={batch_count} was requested, "
                    f"but the corresponding train split is empty."
                )

        for dataset_id, weight in enumerate(self.dataset_batch_allocation):
            if weight > 0 and self.per_rank_batch_counts[dataset_id] == 0:
                raise ValueError(
                    f"dataset_batch_allocation[{dataset_id}]={weight} is too small for batch_size={self.batch_size}. "
                    f"Increase batch_size or reduce the number of active datasets."
                )

        self.required_batches_per_dataset = [
            math.ceil(len(self.grouped_indices.get(dataset_id, [])) / self.global_batch_counts[dataset_id])
            if self.global_batch_counts[dataset_id] > 0 else 0
            for dataset_id in range(len(self.dataset_batch_allocation))
        ]
        self.total_batches = max(self.required_batches_per_dataset, default=0)

    def __len__(self) -> int:
        return self.total_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _shuffle_indices(self, indices: list[int], generator: torch.Generator) -> list[int]:
        if not self.shuffle or len(indices) <= 1:
            return list(indices)

        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in perm]

    def _draw_indices(
        self,
        dataset_id: int,
        count: int,
        generator: torch.Generator,
        dataset_state: dict[int, dict[str, object]],
    ) -> list[int]:
        state = dataset_state[dataset_id]
        base_indices = state["base_indices"]
        order = state["order"]
        offset = state["offset"]
        drawn: list[int] = []

        while len(drawn) < count:
            if len(base_indices) == 0:
                raise ValueError(f"Dataset {dataset_id} has no samples to draw from.")

            remaining = len(order) - offset
            take = min(count - len(drawn), remaining)
            drawn.extend(order[offset:offset + take])
            offset += take

            if offset >= len(order):
                order = self._shuffle_indices(base_indices, generator)
                offset = 0

        state["order"] = order
        state["offset"] = offset
        return drawn

    def __iter__(self):
        generator = torch.Generator().manual_seed(42 + self.epoch)
        dataset_state: dict[int, dict[str, object]] = {}
        for dataset_id, indices in self.grouped_indices.items():
            dataset_state[dataset_id] = {
                "base_indices": list(indices),
                "order": self._shuffle_indices(indices, generator),
                "offset": 0,
            }

        for _ in range(self.total_batches):
            rank_batches = [[] for _ in range(self.num_replicas)]

            for dataset_id, per_rank_count in enumerate(self.per_rank_batch_counts):
                if per_rank_count <= 0:
                    continue

                global_indices = self._draw_indices(
                    dataset_id,
                    per_rank_count * self.num_replicas,
                    generator,
                    dataset_state,
                )

                for replica_rank in range(self.num_replicas):
                    begin = replica_rank * per_rank_count
                    end = begin + per_rank_count
                    rank_batches[replica_rank].extend(global_indices[begin:end])

            for replica_rank in range(self.num_replicas):
                if self.shuffle and len(rank_batches[replica_rank]) > 1:
                    perm = torch.randperm(len(rank_batches[replica_rank]), generator=generator).tolist()
                    rank_batches[replica_rank] = [rank_batches[replica_rank][idx] for idx in perm]

            yield rank_batches[self.rank]


class MixedSequenceBatchSampler:
    def __init__(self, dataset: Dataset, batch_size: int, dataset_batch_allocation: list[int], shuffle: bool):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_batch_allocation = [int(v) for v in dataset_batch_allocation]
        self.shuffle = shuffle
        self.epoch = 0
        self.grouped_indices: dict[int, list[int]] = {}

        if not hasattr(dataset, "seq_dataset_ids"):
            raise TypeError(f"MixedSequenceBatchSampler requires seq_dataset_ids, got {type(dataset)}")

        for local_idx, dataset_id in enumerate(dataset.seq_dataset_ids):
            self.grouped_indices.setdefault(int(dataset_id), []).append(local_idx)

        max_dataset_id = max(self.grouped_indices.keys(), default=-1)
        if len(self.dataset_batch_allocation) <= max_dataset_id:
            raise ValueError(
                f"dataset_batch_allocation length ({len(self.dataset_batch_allocation)}) "
                f"does not match available sequence datasets ({max_dataset_id + 1})."
            )

        active_weights = [
            weight if len(self.grouped_indices.get(dataset_id, [])) > 0 else 0
            for dataset_id, weight in enumerate(self.dataset_batch_allocation)
        ]
        self.per_batch_counts = _allocate_counts(self.batch_size, active_weights)

        for dataset_id, weight in enumerate(active_weights):
            if weight > 0 and self.per_batch_counts[dataset_id] == 0:
                raise ValueError(
                    f"dataset_batch_allocation[{dataset_id}]={weight} is too small for inner batch_size={self.batch_size}."
                )

        self.required_batches_per_dataset = [
            math.ceil(len(self.grouped_indices.get(dataset_id, [])) / self.per_batch_counts[dataset_id])
            if self.per_batch_counts[dataset_id] > 0 else 0
            for dataset_id in range(len(self.dataset_batch_allocation))
        ]
        self.total_batches = max(self.required_batches_per_dataset, default=0)

    def __len__(self) -> int:
        return self.total_batches

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _shuffle_indices(self, indices: list[int], generator: torch.Generator) -> list[int]:
        if not self.shuffle or len(indices) <= 1:
            return list(indices)

        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in perm]

    def _draw_indices(self, dataset_id: int, count: int, generator: torch.Generator, state: dict[int, dict[str, object]]) -> list[int]:
        dataset_state = state[dataset_id]
        base_indices = dataset_state["base_indices"]
        order = dataset_state["order"]
        offset = dataset_state["offset"]
        drawn: list[int] = []

        while len(drawn) < count:
            remaining = len(order) - offset
            take = min(count - len(drawn), remaining)
            drawn.extend(order[offset:offset + take])
            offset += take

            if offset >= len(order):
                order = self._shuffle_indices(base_indices, generator)
                offset = 0

        dataset_state["order"] = order
        dataset_state["offset"] = offset
        return drawn

    def __iter__(self):
        generator = torch.Generator().manual_seed(42 + self.epoch)
        state: dict[int, dict[str, object]] = {}
        for dataset_id, indices in self.grouped_indices.items():
            state[dataset_id] = {
                "base_indices": list(indices),
                "order": self._shuffle_indices(indices, generator),
                "offset": 0,
            }

        for _ in range(self.total_batches):
            batch_indices: list[int] = []
            for dataset_id, count in enumerate(self.per_batch_counts):
                if count <= 0:
                    continue
                batch_indices.extend(self._draw_indices(dataset_id, count, generator, state))

            if self.shuffle and len(batch_indices) > 1:
                perm = torch.randperm(len(batch_indices), generator=generator).tolist()
                batch_indices = [batch_indices[idx] for idx in perm]

            yield batch_indices

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
        #self.model = torch.compile(
        #     self.model,
        #     fullgraph=False,
        #     dynamic=True
        #)

        # MoEのようにバッチによって特定のExpert（パラメータ）が全く使われないことがある構造では、
        # find_unused_parameters=True が必須です。
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        total_param, self.active_params = self.model.module.get_param()
        adam = torch.optim.AdamW(self.model.parameters(), lr=t_args.lr_param)

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

        super().__init__(criterion=nn.CrossEntropyLoss(ignore_index=0).to(device),
                         optimizer=adam,
                         t_args=t_args,
                         m_args=args,
                         calc_val_loss_tokens=calc_val_loss_tokens)

        self.all_tokens = torch.tensor(0, device=device, dtype=torch.long)
        self.optimizer_steps = 0

    def pre_processing(self, pack, progress):
        mini_dataset = MORTM_SEQDataset(
            progress,
            self.args.position_length,
            self.args.min_length,
            is_random_delete_key=False,
            mask_sample_task=[self.tokenizer.get("<MGEN>")],
            program_token_id=[self.tokenizer.get("<INST_PIANO>"), self.tokenizer.get("<INST_SAX>")],
            system_tag=(self.tokenizer.get("<SYSTEM>"), self.tokenizer.get("<TAG_END>")),
            sampling_inst_max=2,
        )

        paths, dataset_ids = _extract_paths_and_dataset_ids_from_pack(pack)
        for path, dataset_id in zip(paths, dataset_ids):
            with np.load(path, allow_pickle=True) as np_load_data:
                mini_dataset.add_data(np_load_data, dataset_id=dataset_id)

        return mini_dataset

    def epoch_fc(self, model, pack, progress):
        src = pack
        device = torch.device(f"cuda:{self.local_rank}")
        target: Tensor = src[:, 1:].to(device)
        target = target.reshape(-1).long()

        src = src[:, :-1].to(device)
        padding_mask_in: Tensor = _get_padding_mask(src, progress, device)

        if model.training:
            self.all_tokens += torch.sum(padding_mask_in == 1)

        input: Tensor = model(x=src, padding_mask=padding_mask_in, is_causal=True)
        input = input.view(-1, input.size(-1)).to(device)
        return input.to(device=device, dtype=torch.float32), target

    def view_logs(self, epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens):
        # 全プロセスで進捗を確認できるように、Rank 0 以外のログも抑制を外します (デバッグ用)
        progress_bar_with_minibatch(self.local_rank, epoch, sum_epoch, seq_count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens)

    def optional_logging(self, val_loss, step):
        global_tokens = self.get_synced_tokens()

        if self.local_rank == 0:
            wandb.log({
                "axis/val_loss": val_loss,
                "axis/tokens": global_tokens,
                "axis/flops": 6 * self.active_params * global_tokens,
                "trainer/global_step": step
            })

    def loss_mask(self, x: torch.Tensor) -> torch.Tensor:
        # 生成: <MGEN>/<CGEN>, メタ分析: <META>, 属性別分析: <KEY>/<DENCE>/<GENRE>/<LENGTH>
        # これら(トリガー)以降=予測対象(答え)のみ損失。入力(旋律/条件)は損失しない。
        start_ids = [self.tokenizer.get(t) for t in
                     ("<MGEN>", "<CGEN>", "<META>", "<KEY>", "<DENCE>", "<GENRE>", "<LENGTH>")]
        is_start_token = torch.zeros_like(x, dtype=torch.long)
        for sid in start_ids:
            is_start_token = is_start_token | (x == sid).long()

        cumulative_mask = torch.cumsum(is_start_token, dim=1)
        mask_x = (cumulative_mask > 0).long()
        # foundation事前学習: <MGEN>/<CGEN>/<META>マーカーを含まない系列は
        # 全トークンで損失を計算する(マスク全0→loss0→未学習 を回避)。
        has_marker = is_start_token.any(dim=1, keepdim=True)
        mask_x = torch.where(has_marker.bool(), mask_x, torch.ones_like(mask_x))
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


def _paths_from_directory_and_filename(directory: list[str], filename: list[str]) -> list[str]:
    return [os.path.join(directory[i], filename[i]) for i in range(len(directory))]


def _collect_grouped_paths(root_directory) -> tuple[list[list[str]], list[str]]:
    if isinstance(root_directory, str):
        if root_directory.endswith(".json"):
            directory, filename = find_files_with_json(root_directory)
        else:
            directory, filename = find_files(root_directory, '.npz')
        return [_paths_from_directory_and_filename(directory, filename)], [root_directory]

    if isinstance(root_directory, (tuple, list)):
        grouped_paths: list[list[str]] = []
        dataset_names: list[str] = []
        for root in root_directory:
            if root.endswith(".json"):
                directory, filename = find_files_with_json(root)
            else:
                directory, filename = find_files(root, '.npz')
            grouped_paths.append(_paths_from_directory_and_filename(directory, filename))
            dataset_names.append(root)
        return grouped_paths, dataset_names

    directory, filename = root_directory
    return [_paths_from_directory_and_filename(directory, filename)], ["provided_paths"]


def _build_preloading_dataset(grouped_paths: list[list[str]], progress: LearningProgress, dataset_names: Optional[list[str]] = None):
    mortm_dataset = PreLoadingDatasets(progress)

    for dataset_id, paths in enumerate(grouped_paths):
        dataset_name = dataset_names[dataset_id] if dataset_names is not None and dataset_id < len(dataset_names) else None
        mortm_dataset.add_paths(paths, dataset_id=dataset_id, dataset_name=dataset_name)

    return mortm_dataset


def _validate_grouped_paths_exist(grouped_paths: list[list[str]], dataset_names: Optional[list[str]] = None, preview_limit: int = 3):
    missing_reports: list[str] = []

    for dataset_id, paths in enumerate(grouped_paths):
        dataset_name = dataset_names[dataset_id] if dataset_names is not None and dataset_id < len(dataset_names) else f"dataset_{dataset_id}"
        missing_paths = [path for path in paths if not os.path.exists(path)]

        if not missing_paths:
            continue

        preview = "\n".join(f"  - {path}" for path in missing_paths[:preview_limit])
        remain = len(missing_paths) - min(len(missing_paths), preview_limit)
        remain_text = f"\n  ... and {remain} more" if remain > 0 else ""
        missing_reports.append(
            f"[{dataset_name}] missing {len(missing_paths)} / {len(paths)} files:\n{preview}{remain_text}"
        )

    if missing_reports:
        raise FileNotFoundError(
            "Dataset JSON contains paths that do not exist on this machine.\n"
            + "\n".join(missing_reports)
        )

def collate_fn(batch):
    src = pad_sequence(batch, batch_first=True, padding_value=0)
    return src

def collate_fn_with_tgt(batch):
    tgt_list = [item[1] for item in batch]
    src = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    tgt = torch.tensor(tgt_list, device=src.device)
    return src, tgt

def save_path_json(name, val_loader: DataLoader, save_directory, version):
    paths = _extract_all_paths_from_dataset(val_loader.dataset)
    with open(f"{save_directory}/{name}_paths_{version}.json", "w") as f:
        json.dump(paths, f, indent=4)


def _format_large_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _dataset_label(name: str) -> str:
    return os.path.basename(name.rstrip("/")) or name


def _print_loader_summary(train_loader: DataLoader, val_loader: DataLoader, train_args: TrainArgs, world_size: int):
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    global_outer_batch = train_args.big_batch_size * world_size

    print(
        f"[Train Setup] world_size={world_size} outer_batch/rank={train_args.big_batch_size} "
        f"global_outer_batch={global_outer_batch} inner_batch={train_args.batch_size} "
        f"accumulation={train_args.accumulation_steps} epochs={train_args.num_epochs}"
    )
    print(
        f"[Train Setup] train_outer_batches={len(train_loader)} val_outer_batches={len(val_loader)} "
        f"train_samples={_format_large_count(len(train_dataset))} val_samples={_format_large_count(len(val_dataset))}"
    )

    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if not isinstance(batch_sampler, DatasetBatchAllocationDistributedBatchSampler):
        return

    dataset_names = _extract_all_dataset_names_from_dataset(train_dataset)
    inner_counts = _allocate_counts(train_args.batch_size, train_args.dataset_batch_allocation)
    print(
        f"[Train Setup] inner_mix="
        + ", ".join(
            f"{dataset_id}:{count}" for dataset_id, count in enumerate(inner_counts) if count > 0
        )
    )
    for dataset_id, train_indices in sorted(batch_sampler.grouped_indices.items()):
        name = _dataset_label(dataset_names[train_indices[0]]) if train_indices else f"dataset_{dataset_id}"
        sample_count = len(train_indices)
        pack_count_per_rank = batch_sampler.per_rank_batch_counts[dataset_id]
        global_pack_count = batch_sampler.global_batch_counts[dataset_id]
        required = batch_sampler.required_batches_per_dataset[dataset_id]
        weight = batch_sampler.dataset_batch_allocation[dataset_id]
        seen_global_samples = batch_sampler.total_batches * global_pack_count
        repeat_factor = seen_global_samples / sample_count if sample_count > 0 else 0.0
        print(
            f"[Train Setup] dataset[{dataset_id}] {name}: weight={weight} "
            f"train_npz={_format_large_count(sample_count)} npz/rank/big_batch={pack_count_per_rank} "
            f"big_batches/epoch={batch_sampler.total_batches} min_big_batches_to_cover={required} "
            f"repeat_factor={repeat_factor:.2f}x"
        )


def _print_epoch_summary(epoch: int, train_args: TrainArgs, train_loader: DataLoader, optimizer_step: int):
    print(
        f"\n[Epoch {epoch + 1}/{train_args.num_epochs}] "
        f"outer_batches={len(train_loader)} optimizer_steps_so_far={optimizer_step}"
    )


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
    lr_text = f"{lr:.3e}" if isinstance(lr, (int, float)) else str(lr)

    print(
        f"\r [Rank{rank}] Epoch {epoch + 1}/{sum_epoch} "
        f"Outer {seq_count}/{all_pac} [{big_bar}] {big_per:.2f}%  "
        f"Inner {mini_seq_count}/{mini_seq_pac} [{mini_bar}] {mini_per:.2f}%  "
        f"loss:{loss:.4f} lr:{lr_text} val:{verif_loss:.4f} tokens:{_format_large_count(tokens)}",
        end=""
    )


def path_pack_collate_fn(batch):
    paths = [item[0] for item in batch]
    dataset_ids = [int(item[1]) for item in batch]
    return paths, dataset_ids


def get_inner_loader(dataset: Dataset, batch_size: int, collate_fn=None, dataset_batch_allocation: Optional[list[int]] = None):
    if dataset_batch_allocation is not None:
        batch_sampler = MixedSequenceBatchSampler(
            dataset,
            batch_size=batch_size,
            dataset_batch_allocation=dataset_batch_allocation,
            shuffle=True,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )
        return loader, batch_sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,
    )
    return loader, None

def get_data_loader(t_args: TrainArgs, mortm_dataset: tuple | Dataset, shuffle=True, collate_fn=None):
    if isinstance(mortm_dataset, Dataset):
        if t_args.dataset_batch_allocation is not None:
            train_dataset, val_dataset = _split_dataset_by_dataset_id(mortm_dataset, t_args.train_dataset_split)
        else:
            train_size, val_size = _compute_split_sizes(len(mortm_dataset), t_args.train_dataset_split)

            # 全 rank で同じ split になるよう固定シード
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                mortm_dataset,
                [train_size, val_size],
                generator=generator
            )
    else:
        train_dataset, val_dataset = mortm_dataset

    train_sampler = None
    train_batch_sampler = None
    if t_args.dataset_batch_allocation is not None:
        train_batch_sampler = DatasetBatchAllocationDistributedBatchSampler(
            train_dataset,
            batch_size=t_args.big_batch_size,
            dataset_batch_allocation=t_args.dataset_batch_allocation,
            shuffle=shuffle,
        )
    else:
        train_sampler = _build_distributed_sampler(train_dataset, shuffle=shuffle)

    val_sampler = _build_distributed_sampler(val_dataset, shuffle=False)

    outer_collate_fn = path_pack_collate_fn if _supports_dataset_ids(train_dataset) else collate_fn

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=0,
            collate_fn=outer_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=t_args.big_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None and shuffle),
            num_workers=0,
            collate_fn=outer_collate_fn,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=t_args.big_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=outer_collate_fn,
    )

    return train_loader, val_loader

def get_verification_loss(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, progress: LearningProgress,
                          trainer, train_args: TrainArgs, coll_fn=None):
    model.eval()

    device = torch.device(f"cuda:{trainer.local_rank}")
    val_loss = torch.tensor(0.0, device=device)
    all_count = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for pack in val_loader:
            pre_processing_dataset: Dataset = trainer.pre_processing(pack, progress)

            if len(pre_processing_dataset) == 0:
                del pre_processing_dataset, pack
                gc.collect()
                continue

            inner_loader, inner_sampler = get_inner_loader(
                pre_processing_dataset,
                train_args.batch_size,
                coll_fn,
                dataset_batch_allocation=train_args.dataset_batch_allocation,
            )

            if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
                inner_sampler.set_epoch(0)

            for pack2 in inner_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r_pack = trainer.epoch_fc(model, pack2, progress)
                    loss = trainer.get_eval_loss(*r_pack)

                val_loss += loss.detach()
                all_count += 1

                del pack2, r_pack, loss

            del inner_loader, inner_sampler, pre_processing_dataset, pack
            gc.collect()

    if _is_dist_ready():
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_count, op=dist.ReduceOp.SUM)

    model.train()

    if all_count.item() == 0:
        return 0.0

    return (val_loss / all_count).item()
def self_turing(model_name, train_args: TrainArgs, save_directory, trainer: AbstractTrainSet,
                train_loader: DataLoader, val_loader: DataLoader,
                message: Messenger, progress: LearningProgress,
                writer, coll_fn=None):

    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler

    local_rank = trainer.local_rank
    world_size = trainer.world_size

    mail_bool = True
    epoch1_end = False
    all_count = 1
    optimizer_step = 0
    verification_loss = 0.0

    device = torch.device(f"cuda:{local_rank}")

    for epoch in range(train_args.num_epochs):
        _set_loader_epoch(train_loader, epoch)
        _set_loader_epoch(val_loader, epoch)

        try:
            count = 1
            epoch_loss = EpochObserver(1000)
            verification_loss = 0.0

            model.train()
            optimizer.zero_grad()

            if local_rank == 0:
                _print_epoch_summary(epoch, train_args, train_loader, optimizer_step)

            for pack in train_loader:
                begin_time = time.time()

                pre_processing_dataset = trainer.pre_processing(pack, progress)

                # 空でも continue せず _sync_min_loader_length に必ず参加する。
                # 一方のランクだけが continue すると dist.all_reduce がズレて NCCL デッドロックになる。
                if len(pre_processing_dataset) > 0:
                    inner_loader, inner_sampler = get_inner_loader(
                        pre_processing_dataset,
                        train_args.batch_size,
                        coll_fn,
                        dataset_batch_allocation=train_args.dataset_batch_allocation,
                    )
                    local_len = len(inner_loader)
                else:
                    inner_loader = iter([])
                    inner_sampler = None
                    local_len = 0

                if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
                    inner_sampler.set_epoch(epoch)

                synced_len = _sync_min_loader_length(local_len, device)

                mini_c = 0
                count += 1

                for pack2 in inner_loader:
                    if mini_c >= synced_len:
                        break

                    mini_c += 1
                    all_count += 1

                    is_step_optimizer = (mini_c % train_args.accumulation_steps == 0)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        r_pack = trainer.epoch_fc(model, pack2, progress)
                        loss = trainer.backward(
                            train_args.accumulation_steps,
                            is_step_optimizer,
                            progress,
                            train_args.lr_param,
                            *r_pack
                        )

                    if is_step_optimizer:
                        optimizer_step += 1
                        trainer.optimizer_steps = optimizer_step

                        # §9.1/E2: total_steps の指定%時点でフルFT途中チェックポイントを保存(opt-in)。
                        _pcts = getattr(train_args, "checkpoint_percents", None)
                        _tot = train_args.scheduler.get("total_steps") if isinstance(train_args.scheduler, dict) else None
                        if _pcts and _tot and (trainer.local_rank == 0):
                            for _p in _pcts:
                                if optimizer_step == max(1, int(round(_p / 100.0 * _tot))):
                                    torch.save(
                                        model.module.state_dict(),
                                        f"{save_directory}/{model_name}.ckpt_p{_p}.pth",
                                    )
                                    print(f"[fullFT] checkpoint saved at {_p}% (step {optimizer_step}/{_tot})", flush=True)

                    reduced_loss = reduce_tensor(loss, op=dist.ReduceOp.SUM) if _is_dist_ready() else loss.detach()
                    avg_loss = (reduced_loss / world_size).item() if _is_dist_ready() else reduced_loss.item()

                    epoch_loss.add(avg_loss)

                    current_tokens = trainer.get_synced_tokens()
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else train_args.lr_param

                    trainer.view_logs(
                        epoch,
                        train_args.num_epochs,
                        count,
                        len(train_loader),
                        mini_c,
                        len(inner_loader),
                        epoch_loss.get(),
                        current_lr,
                        verification_loss,
                        current_tokens
                    )

                    del pack2, r_pack, loss, reduced_loss

                del inner_loader, inner_sampler, pre_processing_dataset, pack
                gc.collect()

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
                        update_log(model, writer, optimizer_step)

                    torch.cuda.empty_cache()
                    verification_loss = get_verification_loss(
                        model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn
                    )

                    if local_rank == 0 and writer is not None:
                        writer.add_scalars(
                            "Train/Verification Loss",
                            {"Train": epoch_loss.get(), "Verification": verification_loss},
                            optimizer_step
                        )

                    trainer.optional_logging(verification_loss, optimizer_step)

            if not epoch1_end:
                epoch1_end = True
                verification_loss = get_verification_loss(
                    model, val_loader, criterion, progress, trainer, train_args, coll_fn=coll_fn
                )
                trainer.optional_logging(verification_loss, optimizer_step)

            sync_tokens = reduce_tensor(trainer.all_tokens, op=dist.ReduceOp.SUM).item() if _is_dist_ready() else trainer.all_tokens.item()

            if local_rank == 0:
                message.send_message(
                    f"{model_name}の途中経過について",
                    f"Epoch {epoch + 1}/{train_args.num_epochs}の結果は、{epoch_loss.get():.4f}でした。\n"
                    f"また、検証データの損失は{verification_loss:.4f}となっています。\n "
                    f"また現在学習中のトークン数は{sync_tokens}です。\n 以上です。"
                )

                if train_args.is_save_training_progress:
                    torch.save(
                        model.module.state_dict(),
                        f"{save_directory}/{model_name}.train.{epoch}.{verification_loss:.4f}.pth"
                    )

        except torch.cuda.OutOfMemoryError:
            if local_rank == 0:
                print(
                    "エラーが発生し、処理を中断しました",
                    "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください"
                )

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
        _print_loader_summary(train_loader, val_loader, t_args, trainer.world_size)

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

    # rank 0 だけ NTFS をスキャンしてブロードキャスト。
    # ntfs-3g (FUSE) への並列 readdir は D クラスデッドロックを引き起こすため、
    # 全ランクが同時に os.walk するのを避ける。
    if local_rank == 0:
        grouped_paths, dataset_names = _collect_grouped_paths(root_directory)
    else:
        grouped_paths, dataset_names = None, None

    if dist.is_initialized():
        obj = [grouped_paths, dataset_names]
        dist.broadcast_object_list(obj, src=0)
        grouped_paths, dataset_names = obj[0], obj[1]

    mortm_dataset = _build_preloading_dataset(grouped_paths, progress, dataset_names)

    if eval_list_json is None:
        train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=t_args.shuffle)
    else:
        if local_rank == 0:
            eval_grouped_paths, eval_dataset_names = _collect_grouped_paths(eval_list_json)
        else:
            eval_grouped_paths, eval_dataset_names = None, None

        if dist.is_initialized():
            obj = [eval_grouped_paths, eval_dataset_names]
            dist.broadcast_object_list(obj, src=0)
            eval_grouped_paths, eval_dataset_names = obj[0], obj[1]

        val_dataset = _build_preloading_dataset(eval_grouped_paths, progress, eval_dataset_names)
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

    grouped_paths, dataset_names = _collect_grouped_paths(root_directory)
    mortm_dataset = _build_preloading_dataset(grouped_paths, progress, dataset_names)
    if eval_list_json is None:
        train_loader, val_loader = get_data_loader(t_args, mortm_dataset, shuffle=True)
    else:
        eval_grouped_paths, eval_dataset_names = _collect_grouped_paths(eval_list_json)
        val_dataset = _build_preloading_dataset(eval_grouped_paths, progress, eval_dataset_names)
        train_loader, val_loader = get_data_loader(t_args, (mortm_dataset, val_dataset), shuffle=True)

    _train(trainer.args, t_args, save_directory, trainer, message=message, version=version, today_date=today_date,
           train_loader=train_loader, val_loader=val_loader, coll_fn=coll_fn,
           progress=progress)

    if dist.is_initialized():
        dist.destroy_process_group()
