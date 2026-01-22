import json
import os

import os
import json
import numpy as np

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

                        # 【修正箇所】
                        # 0次元配列(スカラ)の場合、len()を使うとエラーになるため
                        # 強制的に1次元配列として扱って長さを取得する
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
        else:
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

# --- 使用例 ---
if __name__ == "__main__":
    input_json = "out/models/mortm/4_5/train_paths_4.5-Pro-Preview-4.json"
    output_json = "out/models/mortm/4_5/scaled_dataset_filtered.json"
    target_tokens = 1396597367 // 4

    # フィルタ条件 (例: 128トークン以上、2048トークン以下のシーケンスのみカウント)
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