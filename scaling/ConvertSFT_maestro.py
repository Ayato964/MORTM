"""MAESTRO Dataset (v3.0.0) 用の SFT 生成タスク データ変換スクリプト。
MAESTRO の 1,276 曲から、ピアノ生成タスク (meta / meta_past / meta_future / infill) のシーケンスを作成し、
.npz として保存。公式の train / validation スプリットに準拠して train.json / eval.json を自動生成する。
"""
import hashlib
import json
import os
import random
import sys
import time
from multiprocessing import Manager, Process, Queue

import numpy as np

# プロジェクトルートを sys.path に追加
sys.path.insert(0, "/home/takaaki-nagoshi/PycharmProjects/MORTM")

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import GenerationDataMaker
from ConvertFoundation import writer_worker, _QUEUE_SENTINEL

DATASETS_DIR = "/home/takaaki-nagoshi/PycharmProjects/maestro-v3.0.0"
JSON_PATH = os.path.join(DATASETS_DIR, "maestro-v3.0.0.json")
SAVE_PATH = "/home/takaaki-nagoshi/PycharmProjects/MORTM/data/sft/maestro"
PROGRAM = ["PIANO"]
MIN_MEASURE = 1
MAX_MEASURE = 8
THREAD_VALUE = 16
SFT_TASKS = ("meta", "meta_past", "meta_future", "infill")


def load_maestro_split_map(json_path: str):
    """maestro-v3.0.0.json から basename -> split ('train', 'validation', 'test') の対応マップを作成。"""
    with open(json_path, "r") as f:
        meta = json.load(f)

    file_to_split = {}
    midi_files = meta.get("midi_filename", {})
    splits = meta.get("split", {})

    for idx, midi_rel in midi_files.items():
        base_name = os.path.basename(midi_rel)
        split = splits.get(idx, "train")
        file_to_split[base_name] = split
        # 完全な相対パスでも引けるように登録
        file_to_split[midi_rel] = split

    return file_to_split


def find_midis(root: str):
    """MAESTRO フォルダ配下の全 MIDI ファイルを探索。"""
    direc, files = [], []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                direc.append(dp)
                files.append(f)
    return direc, files


def convert_worker(pid, tokenizer, directory, md_file, program, min_m, max_m,
                   progress, write_queue, save_path):
    random.seed(os.urandom(32))
    local_count = 0
    local_tokens = 0
    total = len(md_file)

    for i, (d, f) in enumerate(zip(directory, md_file), 1):
        try:
            con = MIDIConverter(tokenizer, d, f, program)
            con.convert()
            if con.is_error:
                continue

            maker = GenerationDataMaker(con, min_m, max_m)
            # ジャンルトークンは付与しない (genre_tokens=None)
            stats = maker.convert_generation_sft(tasks=SFT_TASKS, genre_tokens=None)
            if len(maker.aya_node) <= 1:
                continue

            if any(np.sum(a == 0) != 0 for a in maker.aya_node[1:]):
                print(f"[#{pid}] ({i}/{total}) ERROR zero-token {f}")
                continue

            task = maker.prepare_write_task(save_path, save_stats=False)
            if task is None:
                continue

            out_dir, filename, array_dict, stats_data = task
            tok = stats.get("total_tokens", 0) if stats else 0
            write_queue.put((out_dir, filename, array_dict, stats_data, tok))
            local_count += 1
            local_tokens += tok

            if i % 25 == 0 or i == total:
                print(f"[Worker {pid:02d}] ({i}/{total}) converted={local_count}, tokens={local_tokens:,}", flush=True)
        except Exception as e:
            print(f"[Worker {pid:02d}] ({i}/{total}) EXCEPTION {f}: {e}")
            continue

    progress[pid] = {"count": local_count, "tokens": local_tokens}


def main():
    start_t = time.time()
    os.makedirs(SAVE_PATH, exist_ok=True)
    for h in "0123456789abcdef":
        os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)

    # MAESTRO 公式スプリットの読み込み
    split_map = {}
    if os.path.exists(JSON_PATH):
        split_map = load_maestro_split_map(JSON_PATH)
        print(f"Loaded MAESTRO metadata from {JSON_PATH} ({len(split_map)} mappings)")
    else:
        print(f"[WARN] {JSON_PATH} not found. Fallback to random 95:5 split.")

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    directory, md_file = find_midis(DATASETS_DIR)
    print(f"==================================================")
    print(f"MAESTRO SFT Data Converter")
    print(f"Source: {DATASETS_DIR} ({len(md_file)} files)")
    print(f"Save Path: {SAVE_PATH}")
    print(f"Programs: {PROGRAM}, Tasks: {SFT_TASKS}, Measures: {MIN_MEASURE}-{MAX_MEASURE}")
    print(f"Threads: {THREAD_VALUE}")
    print(f"==================================================")

    dirs_split = np.array_split(directory, THREAD_VALUE)
    files_split = np.array_split(md_file, THREAD_VALUE)

    write_queue = Queue(maxsize=200)
    result_queue = Queue()
    writer = Process(target=writer_worker, args=(write_queue, result_queue, False), daemon=False)
    writer.start()

    manager = Manager()
    progress = manager.dict()

    workers = []
    for pid in range(THREAD_VALUE):
        w = Process(
            target=convert_worker,
            args=(
                pid,
                tokenizer,
                dirs_split[pid].tolist(),
                files_split[pid].tolist(),
                PROGRAM,
                MIN_MEASURE,
                MAX_MEASURE,
                progress,
                write_queue,
                SAVE_PATH,
            ),
        )
        workers.append(w)
        w.start()

    for w in workers:
        w.join()

    # ライター終了シグナル
    write_queue.put(_QUEUE_SENTINEL)
    writer.join()

    res = result_queue.get()
    tot_files = res["count"]
    tot_tokens = res["tokens"]
    elapsed = time.time() - start_t

    print(f"\n[Done] Conversion finished in {elapsed:.1f}s")
    print(f"Converted Files: {tot_files} files, Total Tokens: {tot_tokens:,}")

    # 生成された npz ファイルを探索して公式スプリットに基づき train.json / eval.json を作成
    npz_paths = []
    for dp, _, fs in os.walk(SAVE_PATH):
        for f in fs:
            if f.endswith(".npz"):
                npz_paths.append(os.path.abspath(os.path.join(dp, f)))

    print(f"Found {len(npz_paths)} generated .npz files.")

    train_paths = []
    eval_paths = []

    if split_map:
        for p in npz_paths:
            fname = os.path.basename(p)
            # np.savez は .npz を付加するため、元の midi ファイル名を復元
            orig_name = fname[:-4] if fname.endswith(".npz") else fname
            split = split_map.get(orig_name, "train")
            if split == "train":
                train_paths.append(p)
            else:  # validation または test
                eval_paths.append(p)
    else:
        # フォールバック: ランダム 95:5 分割
        random.seed(42)
        random.shuffle(npz_paths)
        n_train = int(len(npz_paths) * 0.95)
        train_paths = npz_paths[:n_train]
        eval_paths = npz_paths[n_train:]

    train_json = os.path.join(SAVE_PATH, "train.json")
    eval_json = os.path.join(SAVE_PATH, "eval.json")

    with open(train_json, "w") as f:
        json.dump(train_paths, f, indent=2)
    with open(eval_json, "w") as f:
        json.dump(eval_paths, f, indent=2)

    print(f"Train paths ({len(train_paths)}): {train_json}")
    print(f"Eval paths  ({len(eval_paths)}): {eval_json}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
