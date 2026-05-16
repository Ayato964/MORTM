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


_QUEUE_SENTINEL = None  # ライタープロセスの停止シグナル


def writer_worker(write_queue: Queue, result_queue: Queue, save_stats: bool):
    """
    NTFS への書き込みを単一プロセスで直列化する専用ライター。
    ntfs-3g の並列 write/close/rename による D クラスデッドロックを防ぐ。
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    count = 0
    tokens = 0

    while True:
        task = write_queue.get()
        if task is _QUEUE_SENTINEL:
            break

        out_dir, filename, array_dict, stats_data, token_count = task
        try:
            np.savez(os.path.join(out_dir, filename), **array_dict)
            if save_stats and stats_data is not None:
                with open(os.path.join(out_dir, filename + "_stats.json"), "w") as f:
                    json.dump(stats_data, f, indent=2)
            count += 1
            tokens += token_count
        except Exception as e:
            print(f"[writer] ERROR  {filename}  {e}")

    result_queue.put({"count": count, "tokens": tokens})


def convert_foundation(pid, tokenizer, directory, md_file, program, min_measure, max_measure, save_stats, progress, write_queue, save_path):
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

                # NTFS への書き込みはライタープロセスへ委譲（D クラスデッドロック回避）
                write_queue.put((out_dir, filename, array_dict, stats_data, token_count))

                local_count += 1
                local_tokens += token_count
                samples = stats["total_samples"] if stats else 0
                print(f"[#{pid}] ({i}/{total}) QUEUED  {label}  samples={samples}  tokens={token_count}")

            _run(con, f)

            for cl in con.expansion_midi():
                _run(cl, cl.file_name)

        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) EXCEPTION  {f}  {e}")
            continue

    progress[pid] = {"count": local_count, "tokens": local_tokens}


if __name__ == "__main__":
    THREAD_VALUE = 25
    MIN_MEASURE = 1
    MAX_MEASURE = 8
    PROGRAM = ['PIANO', 'SAX']
    DATASETS = "/media/takaaki-nagoshi/C8DCDBF4DCDBDB30/MIDIdatasets/GMD/training"
    SAVE_PATH = "/media/takaaki-nagoshi/C8DCDBF4DCDBDB30/MORTM/pre_train/ver5/music"
    SAVE_STATS = False

    # NTFS (ntfs-3g) は並列 mkdir で ENOSPC を返すバグがあるため、
    # プロセス起動前にサブディレクトリを一括作成しておく（16個に抑える）
    os.makedirs(SAVE_PATH, exist_ok=True)
    for h in "0123456789abcdef":
        os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))

    directory, md_file = find_midi_files(DATASETS)
    print(f"Found {len(md_file)} MIDI files.")

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    # 書き込みキュー: maxsize でバックプレッシャーをかけてメモリを抑制
    write_queue = Queue(maxsize=200)
    result_queue = Queue()

    # ライタープロセスを先に起動（全 NTFS 書き込みをここで直列化）
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
                target=convert_foundation,
                args=(t, tokenizer, directory[t].tolist(), md_file[t].tolist(),
                      PROGRAM, MIN_MEASURE, MAX_MEASURE, SAVE_STATS, progress, write_queue, SAVE_PATH),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_queued = sum(v["count"] for v in progress.values())

    # 全ワーカー終了後にライターへ停止シグナルを送る
    write_queue.put(_QUEUE_SENTINEL)
    writer.join()

    writer_result = result_queue.get()
    print(f"変換完了: {writer_result['count']} ファイル保存  総トークン数: {writer_result['tokens']}")
    if total_queued != writer_result['count']:
        print(f"[警告] キュー投入数({total_queued}) != 書き込み完了数({writer_result['count']})")
