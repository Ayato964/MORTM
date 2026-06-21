import json
import os
import random
from multiprocessing import Process, Manager, Queue

import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker


def find_midi_files(root_folder: str):
    midi_files = []
    direc = []
    for defpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


_QUEUE_SENTINEL = None


def writer_worker(write_queue: Queue, result_queue: Queue, save_stats: bool):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    written = []  # list of (path, token_count)

    while True:
        task = write_queue.get()
        if task is _QUEUE_SENTINEL:
            break

        out_dir, filename, array_dict, stats_data, token_count = task
        try:
            base_path = os.path.join(out_dir, filename)
            np.savez(base_path, **array_dict)
            npz_path = base_path if base_path.endswith(".npz") else base_path + ".npz"
            if save_stats and stats_data is not None:
                with open(base_path + "_stats.json", "w") as f:
                    json.dump(stats_data, f, indent=2)
            written.append((npz_path, token_count))
        except Exception as e:
            print(f"[writer] ERROR  {filename}  {e}")

    result_queue.put(written)


def convert_worker(pid, tokenizer, directory, md_file, program, min_measure, max_measure,
                   save_stats, progress, write_queue, save_path):
    random.seed(os.urandom(32))
    local_count = 0
    local_tokens = 0
    total = len(md_file)

    for i, (d, f) in enumerate(zip(directory, md_file), 1):
        try:
            con = MIDIConverter(tokenizer, d, f, program)
            con.convert()

            if con.is_error:
                print(f"[#{pid}] ({i}/{total}) SKIP  {f}  ({con.error_reason})")
                continue

            def _run(maker_con, label):
                nonlocal local_count, local_tokens
                m = FoundationDataMaker(maker_con, min_measure, max_measure)
                stats = m.convert()

                if any(np.sum(a == 0) != 0 for a in m.aya_node[1:]):
                    print(f"[#{pid}] ({i}/{total}) ERROR zero-token  {label}")
                    return

                task = m.prepare_write_task(save_path, save_stats)
                if task is None:
                    print(f"[#{pid}] ({i}/{total}) NG(no data)  {label}")
                    return

                out_dir, filename, array_dict, stats_data = task
                token_count = stats.get("total_tokens", 0) if stats else 0

                write_queue.put((out_dir, filename, array_dict, stats_data, token_count))

                local_count += 1
                local_tokens += token_count
                samples = stats["total_samples"] if stats else 0
                print(f"[#{pid}] ({i}/{total}) QUEUED  {label}  samples={samples}  tokens={token_count}")

            _run(con, f)
            # expansion_midi() は呼ばない（スケーリング則実験用: 転調なし）

        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) EXCEPTION  {f}  {e}")
            continue

    progress[pid] = {"count": local_count, "tokens": local_tokens}


def make_dataset_jsons(written_files, save_base, eval_ratio=0.01):
    """
    written_files: list of (path, token_count)
    スケーリング則実験用に 200M / 400M / 800M / 1.6B / 3.2B のサブセット JSON を生成する。
    各 JSON はその閾値までに必要なファイル群のパスリスト。
    """
    TOKEN_TARGETS = [
        200_000_000,
        400_000_000,
        800_000_000,
        1_600_000_000,
    ]
    TARGET_LABELS = ["200M", "400M", "800M", "1.6B"]

    rng = random.Random(42)
    rng.shuffle(written_files)

    n_eval = max(1, int(len(written_files) * eval_ratio))
    eval_files = written_files[:n_eval]
    train_files = written_files[n_eval:]

    json_base = os.path.join(save_base, "json")
    os.makedirs(json_base, exist_ok=True)

    eval_paths = [p for p, _ in eval_files]
    eval_json_path = os.path.join(json_base, "eval.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_paths, f, indent=2)
    eval_tokens = sum(t for _, t in eval_files)
    print(f"eval.json  : {len(eval_paths)} files  {eval_tokens:,} tokens -> {eval_json_path}")

    cumulative_tokens = 0
    cumulative_paths = []
    target_idx = 0

    for path, tokens in train_files:
        cumulative_tokens += tokens
        cumulative_paths.append(path)

        while target_idx < len(TOKEN_TARGETS) and cumulative_tokens >= TOKEN_TARGETS[target_idx]:
            label = TARGET_LABELS[target_idx]
            out_dir = os.path.join(json_base, label)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "train.json")
            with open(out_path, "w") as f:
                json.dump(list(cumulative_paths), f, indent=2)
            print(f"{label}/train.json: {len(cumulative_paths)} files  {cumulative_tokens:,} tokens -> {out_path}")
            target_idx += 1

        if target_idx >= len(TOKEN_TARGETS):
            break

    if target_idx < len(TOKEN_TARGETS):
        for i in range(target_idx, len(TOKEN_TARGETS)):
            label = TARGET_LABELS[i]
            print(f"[WARNING] データ不足: {label} ({cumulative_tokens:,} / {TOKEN_TARGETS[i]:,} tokens) 全データで代用")
            out_dir = os.path.join(json_base, label)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "train.json")
            with open(out_path, "w") as f:
                json.dump(list(cumulative_paths), f, indent=2)


if __name__ == "__main__":
    THREAD_VALUE = 25
    MIN_MEASURE = 1
    MAX_MEASURE = 8
    PROGRAM = ['PIANO', 'SAX']
    DATASETS = "/media/takaaki-nagoshi/C8DCDBF4DCDBDB30/MIDIdatasets/GMD/training"
    # ext4 ファイルシステムに保存（NTFS ntfs-3g の並列書き込みデッドロックを回避）
    SAVE_PATH = "/home/takaaki-nagoshi/data/scaling/music"
    SAVE_BASE = "/home/takaaki-nagoshi/data/scaling"
    SAVE_STATS = False

    os.makedirs(SAVE_PATH, exist_ok=True)
    for h in "0123456789abcdef":
        os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))

    directory, md_file = find_midi_files(DATASETS)
    print(f"Found {len(md_file)} MIDI files.")

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    write_queue = Queue(maxsize=200)
    result_queue = Queue()

    writer = Process(
        target=writer_worker,
        args=(write_queue, result_queue, SAVE_STATS),
        daemon=False,
    )
    writer.start()

    with Manager() as manager:
        progress = manager.dict()
        processes = []
        for t in range(THREAD_VALUE):
            p = Process(
                target=convert_worker,
                args=(t, tokenizer, directory[t].tolist(), md_file[t].tolist(),
                      PROGRAM, MIN_MEASURE, MAX_MEASURE, SAVE_STATS, progress, write_queue, SAVE_PATH),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_queued = sum(v["count"] for v in progress.values())

    write_queue.put(_QUEUE_SENTINEL)
    writer.join()

    written_files = result_queue.get()  # list of (path, token_count)
    total_tokens = sum(t for _, t in written_files)
    print(f"\n変換完了: {len(written_files)} ファイル保存  総トークン数: {total_tokens:,}")
    if total_queued != len(written_files):
        print(f"[警告] キュー投入数({total_queued}) != 書き込み完了数({len(written_files)})")

    print("\nデータセット JSON を生成中...")
    make_dataset_jsons(written_files, SAVE_BASE)
    print("完了")
