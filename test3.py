import time
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data.dataloader import DataLoader
import json
import random
import os
import sys
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from mortm.train.datasets import MORTM_SEQDataset, PreLoadingDatasets
from mortm.models.modules.progress import _DefaultLearningProgress
import numpy as np

MIN_LENGTH = 40
POSITIONAL_LENGTH = 5000

def create_subset_dataset_with_seq_filter(
        input_json_path,
        output_json_path,
        max_tokens,
        min_seq_length=0,
        max_seq_length=float('inf')
):
    """
    指定されたJSONファイル内のnpzファイルを読み込み、
    条件(min_seq_length <= len <= max_seq_length)を満たすシーケンスのトークン数のみを積算し、
    その合計がmax_tokensに収まるようにファイルリストを作成する関数。

    修正点: npz内に0次元配列(スカラ)が含まれていた場合の len() エラーを回避。
    """

    print(f"Loading input JSON: {input_json_path}")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            dataset_structure = json.load(f)
    except FileNotFoundError:
        print(f"Error: 入力ファイルが見つかりません: {input_json_path}")
        return

    current_valid_tokens = 0
    new_dataset_structure = []

    print(f"Target Max Valid Tokens: {max_tokens}")
    print(f"Filter Condition: {min_seq_length} <= seq_len <= {max_seq_length}")
    print("Counting valid tokens and selecting data...")

    # データセットの構造（グループ単位）をループ
    for i, group in enumerate(dataset_structure):
        group_valid_tokens = 0

        # グループがリストでない場合の対応
        files_in_group = group if isinstance(group, list) else [group]

        # グループ内のファイルをチェック
        for file_path in files_in_group:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                # allow_pickle=True はデータによって必要であれば入れてください
                with np.load(file_path, allow_pickle=True) as data:
                    valid_tokens_in_file = 0
                    for key in data.files:
                        arr = data[key]


                        if arr.ndim == 0:
                            arr = np.atleast_1d(arr)

                        seq_len = len(arr)

                        if min_seq_length <= seq_len <= max_seq_length:
                            valid_tokens_in_file += seq_len

                    group_valid_tokens += valid_tokens_in_file

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        # 累積トークン数が上限を超えないかチェック
        if current_valid_tokens + group_valid_tokens <= max_tokens:
            new_dataset_structure.append(group)
            current_valid_tokens += group_valid_tokens
            print(f"\r Index {i}: Added group with {group_valid_tokens} valid tokens. Total: {current_valid_tokens}", end="")
        else:
            new_dataset_structure.append(group)
            current_valid_tokens += group_valid_tokens
            print(f"Limit reached at index {i}.")
            print(f"Current Valid Total: {current_valid_tokens}, Next Group Valid: {group_valid_tokens}, Limit: {max_tokens}")
            break

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} groups. Current Valid Tokens: {current_valid_tokens}")

    # 保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset_structure, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print("Processing Complete.")
    print(f"Saved to: {output_json_path}")
    print(f"Total Valid Tokens: {current_valid_tokens}")
    print(f"Total Groups: {len(new_dataset_structure)} / {len(dataset_structure)}")


def get_dataset(merge_dataset_paths: List[str], output_dataset_path: str) -> str:
    """
    Merges multiple JSON dataset files, flattens, shuffles, and saves to a specified path.

    :param merge_dataset_paths: List of file paths to the source JSON datasets.
    :param output_dataset_path: Destination file path for the merged dataset.
    :return: The absolute path to the saved dataset.
    """
    merged_data = []
    total_files = len(merge_dataset_paths)

    # 1. Load and Flatten
    for i, path in enumerate(merge_dataset_paths):
        sys.stdout.write(f"\r[Processing] Loading file {i + 1}/{total_files} ...")
        sys.stdout.flush()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Flattening logic for List[List[str]] or List[str]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list):
                            merged_data.extend(item)
                        else:
                            merged_data.append(item)
        except (FileNotFoundError, json.JSONDecodeError):
            # Skip invalid files in research context to maintain flow
            continue

    # 2. Shuffle
    sys.stdout.write(f"\r[Processing] Shuffling {len(merged_data)} samples ...      ")
    sys.stdout.flush()
    random.shuffle(merged_data)

    # 3. Save
    sys.stdout.write(f"\r[Processing] Saving to target path ...                   ")
    sys.stdout.flush()

    # Ensure the directory exists
    output_dir = os.path.dirname(output_dataset_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)

    print("\n[Completed] Dataset generation finished.")

    return os.path.abspath(output_dataset_path)


def collate_fn(batch):
    # バッチ内のテンソルの長さを揃える（パディングする）
    src = pad_sequence(batch, batch_first=True, padding_value=0)
    return src

def _get_padding_mask(input_ids):
    # input_ids が Tensor であることを仮定
    pad_id = (input_ids != 0).to(torch.float)
    padding_mask = pad_id
    return padding_mask

def get_all_tokens_global_step(dataset_json: str, batch: int = 64, mi: int = 180, ma: int = 5000) -> Tuple[int, int]:
    """
    Calculates total tokens and global steps from a dataset JSON configuration.
    """
    base = PreLoadingDatasets(_DefaultLearningProgress())
    base.add_data_json(dataset_json)

    # Outer loader: manages file paths
    dataset = DataLoader(base, batch_size=3072, shuffle=True)

    all_tokens = 0
    global_count = 0
    total_file_batches = len(dataset)

    print(f"Initialization: Target JSON {dataset_json}")

    for i, dt in enumerate(dataset):
        mortm_data = MORTM_SEQDataset(_DefaultLearningProgress(), positional_length=ma, min_length=mi)

        # Load npz files into memory
        # dt is a tuple of filenames from the batch
        for d in dt:
            try:
                np_load_data = np.load(d, allow_pickle=True)
                mortm_data.add_data(np_load_data)
            except Exception:
                # Skip corrupted files to maintain robustness
                continue

        # Inner loader: manages tensor batches
        mini_dataloader = DataLoader(mortm_data, batch_size=batch, shuffle=True, collate_fn=collate_fn)

        for src in mini_dataloader:
            src = src[:, :-1]
            padding_mask_in = _get_padding_mask(src)

            # Count valid tokens (non-padding)
            all_tokens += int(torch.sum(padding_mask_in == 1).item())
            global_count += 1
            print(f"\r[Progress] File Batch {i + 1}/{total_file_batches} | Global Steps: {global_count} | Total Tokens: {all_tokens}", end="")

    print("\nCalculation Completed.")
    return all_tokens, global_count

def analyze_file_strict(file_path):
    """
    180 < len < 5000 の範囲内のみを有効データとしてカウントする。
    ただし、Min/Maxは全データを対象に記録する。
    """
    if not os.path.exists(file_path):
        return None

    try:
        data = np.load(file_path, allow_pickle=True)

        # 生データの統計（フィルタなし）
        raw_min = float('inf')
        raw_max = 0

        # 有効データの統計（フィルタあり）
        valid_tokens = 0
        valid_seqs = 0

        # npz内の全キーを走査
        for key in data.files:
            if key in ['array_0', 'array0', 'meta', 'header'] or not key.startswith('array'):
                continue

            seq = data[key]

            # 0チェック（念のため残します）
            if np.any(seq == 0):
                return {'status': 'ERROR_ZERO', 'file': file_path, 'key': key}

            slen = len(seq)
            if slen == 0:
                continue

            # 1. 生データのMin/Max更新
            if slen < raw_min: raw_min = slen
            if slen > raw_max: raw_max = slen

            # 2. Datasetクラスと全く同じフィルタ条件を適用
            # if self.min_length < len(seq) < self.positional_length:
            if MIN_LENGTH < slen < POSITIONAL_LENGTH:
                valid_tokens += slen
                valid_seqs += 1

        return {
            'status': 'OK',
            'raw_min': raw_min,
            'raw_max': raw_max,
            'valid_tokens': valid_tokens,
            'valid_seqs': valid_seqs
        }

    except Exception as e:
        return {'status': 'LOAD_ERROR'}

def main_strict_validation(dataset_json_path):
    print(f"Target JSON: {dataset_json_path}")
    print(f"Filter Condition: {MIN_LENGTH} < Length < {POSITIONAL_LENGTH}")
    print("-" * 60)

    try:
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            file_paths = json.load(f)
    except Exception as e:
        print(f"JSON Load Error: {e}")
        return

    total_files = len(file_paths)

    # 集計用変数
    global_raw_min = float('inf')
    global_raw_max = 0
    total_valid_tokens = 0
    total_valid_seqs = 0

    start_time = time.time()

    with Pool() as pool:
        iterator = pool.imap_unordered(analyze_file_strict, file_paths, chunksize=50)

        for i, result in enumerate(iterator):
            if result is None or result['status'] == 'LOAD_ERROR':
                continue

            if result['status'] == 'ERROR_ZERO':
                print(f"\n[CRITICAL] Token 0 found in {result['file']}")
                pool.terminate()
                sys.exit(1)

            # 統計更新
            if result['raw_min'] != float('inf'):
                if result['raw_min'] < global_raw_min: global_raw_min = result['raw_min']
                if result['raw_max'] > global_raw_max: global_raw_max = result['raw_max']

            total_valid_tokens += result['valid_tokens']
            total_valid_seqs += result['valid_seqs']

            # ログ出力
            if (i + 1) % 500 == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0

                # 平均有効長
                avg_valid = total_valid_tokens / total_valid_seqs if total_valid_seqs > 0 else 0

                log_msg = (
                    f"\r[Progress] {i + 1}/{total_files} "
                    f"| Valid Tokens: {total_valid_tokens:,} " # カンマ区切りで見やすく
                    f"| Valid Seqs: {total_valid_seqs:,} "
                    f"| Avg Valid Len: {avg_valid:.1f} "
                    f"| Raw Range: [{global_raw_min}, {global_raw_max}]"
                )
                sys.stdout.write(log_msg)
                sys.stdout.flush()

    print("\n" + "=" * 60)
    print("STRICT VALIDATION COMPLETED")
    print(f"Total Valid Tokens : {total_valid_tokens:,}")
    print(f"Total Valid Seqs   : {total_valid_seqs:,}")
    if total_valid_seqs > 0:
        print(f"Avg Valid Length   : {total_valid_tokens / total_valid_seqs:.2f}")
    print(f"Raw Min Length     : {global_raw_min}")
    print(f"Raw Max Length     : {global_raw_max}")
    print("=" * 60)

"""
if __name__ == "__main__":
    input_json = "out/models/mortm/4_5/scaling_test/train_paths_Scaling_.json"
    output_json = "out/models/mortm/4_5/scaling_test/1.6B/train.json"
    target_tokens = 1_600_000_000  # 1億トークン

    min_len = 180
    max_len = 5000

    if os.path.exists(input_json):
        create_subset_dataset_with_seq_filter(
            input_json,
            output_json,
            target_tokens,
            min_seq_length=min_len,
            max_seq_length=max_len
        )
    else:
        print(f"Input file not found: {input_json}")
"""
"""
if __name__ == "__main__":
    datasets = [
        "out/models/mortm/4_5/preview3/eval_cm.json",
        "out/models/mortm/4_5/preview3/eval_music.json"
    ]

    merged_dataset_path = get_dataset(datasets, output_dataset_path="out/models/mortm/4_5/preview3/eval.json")
    print(f"Merged dataset saved at: {merged_dataset_path}")

"""

if __name__ == "__main__":
    dataset_json = "out/models/mortm/4_5/preview3/train.json"
    batch_size = 64
    min_length = 180
    max_length = 5000
    main_strict_validation(dataset_json)
    """
    if os.path.exists(dataset_json):
        total_tokens, global_steps = get_all_tokens_global_step(
            dataset_json,
            batch=batch_size,
            mi=min_length,
            ma=max_length
        )
        print(f"Total Tokens: {total_tokens}, Global Steps: {global_steps}")
    else:
        print(f"Dataset file not found: {dataset_json}")
    """