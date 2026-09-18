"""A3 因子分解アブレーション用データ生成 (§E3)。
ConvertFoundation_noaug.py を流用し、disable_block_augment の代わりに augment_mode を指定する。
  perm_only(A3a) / del_only(A3b) / meta_first(A3c)

- 元MIDI: /media/takaaki-nagoshi/MIDIdatasets/GMD/training
- 対象: paper TRAIN split の曲のみ (docs/splits/train_songs.txt) = リークフリー
- 出力: /home/takaaki-nagoshi/data/scaling/ver5_<mode>/music  (16シャード, ext4)
- 転調(expansion_midi 12種)込み。--limit で曲数制限(400M確保に十分な数だけ変換)。

使い方:
    python scaling/ConvertFoundation_a3.py perm_only --limit 15000
"""
import json
import os
import re
import random
import sys
from multiprocessing import Process, Manager, Queue

import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from ConvertFoundation import writer_worker, _QUEUE_SENTINEL

MODE2DIR = {"perm_only": "A3a", "del_only": "A3b", "meta_first": "A3c"}
TRAIN_SPLIT = "/home/takaaki-nagoshi/PycharmProjects/MORTM/docs/splits/train_songs.txt"


def paper_train_hashes():
    with open(TRAIN_SPLIT) as f:
        return set(x.strip() for x in f if x.strip())


def find_used_midis(root, used):
    direc, files = [], []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                if re.sub(r"\.midi?$", "", f, flags=re.I) in used:
                    direc.append(dp)
                    files.append(f)
    return direc, files


def convert_worker(pid, tokenizer, directory, md_file, program, min_measure, max_measure,
                   save_stats, progress, write_queue, save_path, mode):
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

            def _run(maker_con, label):
                nonlocal local_count, local_tokens
                m = FoundationDataMaker(maker_con, min_measure, max_measure,
                                        disable_block_augment=False, augment_mode=mode)
                stats = m.convert()
                if any(np.sum(a == 0) != 0 for a in m.aya_node[1:]):
                    return
                task = m.prepare_write_task(save_path, save_stats)
                if task is None:
                    return
                out_dir, filename, array_dict, stats_data = task
                token_count = stats.get("total_tokens", 0) if stats else 0
                write_queue.put((out_dir, filename, array_dict, stats_data, token_count))
                local_count += 1
                local_tokens += token_count

            _run(con, f)
            for cl in con.expansion_midi():
                _run(cl, cl.file_name)

        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) EXCEPTION  {f}  {e}")
            continue
        if i % 200 == 0:
            print(f"[#{pid}] ({i}/{total}) count={local_count} tokens={local_tokens:,}", flush=True)

    progress[pid] = {"count": local_count, "tokens": local_tokens}


if __name__ == "__main__":
    THREAD_VALUE = 25
    MIN_MEASURE = 1
    MAX_MEASURE = 8
    PROGRAM = ['PIANO', 'SAX']
    DATASETS = "/media/takaaki-nagoshi/MIDIdatasets/GMD/training"
    SAVE_STATS = False

    mode = sys.argv[1]
    assert mode in MODE2DIR, f"mode must be one of {list(MODE2DIR)}"
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    SAVE_PATH = f"/home/takaaki-nagoshi/data/scaling/ver5_{MODE2DIR[mode]}/music"

    used = paper_train_hashes()
    print(f"paper train曲: {len(used)} 曲")
    directory, md_file = find_used_midis(DATASETS, used)
    print(f"対象MIDI(stem一致): {len(md_file)} 件  mode={mode} -> {SAVE_PATH}")
    if LIMIT:
        # 決定的サブセット(先頭LIMIT件, 3mode共通の曲集合になるようソート)
        order = sorted(range(len(md_file)), key=lambda i: md_file[i])[:LIMIT]
        directory = [directory[i] for i in order]
        md_file = [md_file[i] for i in order]
        print(f"[limit] 先頭 {LIMIT} 曲のみ変換(全mode共通のソート済サブセット)")

    os.makedirs(SAVE_PATH, exist_ok=True)
    for h in "0123456789abcdef":
        os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    write_queue = Queue(maxsize=200)
    result_queue = Queue()
    writer = Process(target=writer_worker, args=(write_queue, result_queue, SAVE_STATS), daemon=False)
    writer.start()

    with Manager() as manager:
        progress = manager.dict()
        procs = []
        for t in range(THREAD_VALUE):
            p = Process(target=convert_worker,
                        args=(t, tokenizer, directory[t].tolist(), md_file[t].tolist(),
                              PROGRAM, MIN_MEASURE, MAX_MEASURE, SAVE_STATS, progress, write_queue, SAVE_PATH, mode))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        total_queued = sum(v["count"] for v in progress.values())

    write_queue.put(_QUEUE_SENTINEL)
    writer.join()
    r = result_queue.get()
    print(f"[{mode}] 変換完了: {r['count']} ファイル保存  総トークン: {r['tokens']:,}")
