import json
import os.path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
# 可変長のシーケンスをバッチ化するために pad_sequence をインポート
from torch.nn.utils.rnn import pad_sequence

from mortm.models.mortm import MORTM, MORTMArgs
import mortm.train.tokenizer as token
import numpy as np

from mortm.train.tokenizer import TO_MUSIC
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import get_token_converter_pro
from mortm.train.datasets import MORTM_SEQDataset
from mortm.train.train import _set_train_data, collate_fn

# --- 初期設定 (変更なし) ---
progress = _DefaultLearningProgress()
tokenizer = token.Tokenizer(music_token=get_token_converter_pro(TO_MUSIC))
tokenizer.mode()

args = MORTMArgs("configs/models/mortm/A.json")
model = MORTM(progress=progress, args=args)
model.load_state_dict(torch.load("out/model/mortm/MORTM.4.0-PIANO_0.9285.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

datasets = MORTM_SEQDataset(progress, args.position_length, 150)
COUNT = 0
with open("out/np/Piano/pre_train/eval.json", "r") as f:
    json_data = json.load(f)
    data = np.array([])
    for l in json_data:
        if COUNT >= 5:
            break
        data = np.concatenate([data, l])
        COUNT += 1
    directory = [os.path.dirname(d) for d in data]
    file_name = [os.path.basename(d) for d in data]
datasets = _set_train_data(directory, file_name,datasets)

# --- ここからが修正箇所 ---

# 1. DataLoaderのバッチサイズを任意の値に設定可能 (例: 8)
BATCH_SIZE = 32
loader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# ファイル名を一意に管理するためのグローバルカウンター
count = 0

print(f"Start processing with batch size: {BATCH_SIZE}")

with torch.no_grad():
    # enumerate を使ってバッチのインデックス(batch_idx)も取得
    for batch_idx, src_batch in enumerate(loader):
        src_batch: Tensor # (B, S) の形状
        B = src_batch.size(0) # バッチサイズを取得

        processed_src_list = []
        for i in range(B):
            src_single = src_batch[i]
            indices = (src_single == 8).nonzero(as_tuple=True)[0]

            # 5番目のトークン'8'が見つかったら、その位置でスライス
            if indices is not None and len(indices) > 4:
                processed_src_list.append(src_single[:indices[4]])
            else:
                processed_src_list.append(src_single)

        # 3. [変更なし] 長さが異なるシーケンスのリストをパディングし、新しいバッチを作成
        src_padded = pad_sequence(processed_src_list, batch_first=True, padding_value=0).to(device)
        if src_padded.max() > 650:
            raise ValueError(f"入力プロンプト：{src_padded.max()}")

        # 4. [変更なし] 各サンプルを3つに複製し、モデルへの入力を作成
        src_repeated = src_padded.repeat_interleave(3, dim=0)

        cleaned_all, cleaned_generated = model.top_sampling_measure_kv_cache(
            src_repeated, p=0.96, max_measure=20, temperature=1.0, print_log=True # ループ内なのでログはオフ推奨
        )
        for c in cleaned_all:
            if max(c) > 586:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Error: Token exceeds maximum value of 585. {max(c)}  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue
        # バッチ内の各オリジナルサンプルをループ
        for b_idx in range(B):
            # 3つの生成バリエーションをループ
            for r_idx in range(3):
                # フラットなリストから対応する結果を取得
                flat_idx = b_idx * 3 + r_idx
                # プロンプトを含む、クリーンな完全シーケンスを取得
                generated_seq = cleaned_all[flat_idx]

                # [削除] 終了トークン'585'を探す処理は不要
                # model.top_sampling_measure_kv_cache 内部でEOS処理が完了しているため、
                # ここでの再度の探索とスライスは必要ありません。

                # [簡略化] aya_nodeの準備と保存
                # generated_seq は既にクリーンなPythonリストです
                aya_node = [0]
                aya_node.append(generated_seq) # .tolist() は不要

                array_dict = {f'array{c}': arr for c, arr in enumerate(aya_node)}

                if len(array_dict) > 1:
                    file_id = count + b_idx
                    np.savez(f"out/np/Piano/rl/ai/pre/{file_id}_{r_idx}", **array_dict)

        # 進捗表示を更新
        print(f"\rProcessing... Batch {batch_idx + 1}/{len(loader)}", end="")

        # 次のバッチのためにカウンターをバッチサイズ分だけ進める
        count += B
        if count >= 5000:
            break

print("\nProcessing finished.")