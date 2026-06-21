"""ver5_noaug データ生成: 前回スケーリング(json_v5)で使用した楽曲のみを、
FoundationDataMaker の disable_block_augment=True (ブロックの入れ替え・削除なし) で再変換する。
転調(expansion_midi 12種)は v5 と同様に行い、同じ 200M〜3.2B スケーリング規模を作れるようにする。

- 元MIDI: /media/takaaki-nagoshi/MIDIdatasets/GMD/training
- 対象: json_v5 (3.2B train + eval) で使われた 217,024 曲のみ (ハッシュ一致でフィルタ)
- 出力: /media/takaaki-nagoshi/MORTM/pre_train/ver5_noaug/music  (16シャード)
ConvertFoundation.py の writer/worker 構造を流用。違いは (1)対象MIDIの限定 (2)disable_block_augment=True。
"""
import json
import os
import re
import random
from multiprocessing import Process, Manager, Queue

import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from ConvertFoundation import writer_worker, _QUEUE_SENTINEL


def used_song_hashes():
    """json_v5 (3.2B train + eval) で使われた全曲ハッシュ集合。"""
    out = set()
    for p in ("/home/takaaki-nagoshi/data/scaling/json_v5/3.2B/train.json",
              "/home/takaaki-nagoshi/data/scaling/json_v5/eval.json"):
        for x in json.load(open(p)):
            out.add(re.sub(r"\.mid.*", "", os.path.basename(x)))
    return out


def find_used_midis(root, used):
    """root 配下の MIDI のうち、stem が used に含まれるものだけ返す。"""
    direc, files = [], []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                if re.sub(r"\.midi?$", "", f, flags=re.I) in used:
                    direc.append(dp)
                    files.append(f)
    return direc, files


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
                # ★ disable_block_augment=True: ブロックの入れ替え・削除なし(決定的)
                m = FoundationDataMaker(maker_con, min_measure, max_measure,
                                        disable_block_augment=True)
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
            for cl in con.expansion_midi():
                _run(cl, cl.file_name)

        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) EXCEPTION  {f}  {e}")
            continue

    progress[pid] = {"count": local_count, "tokens": local_tokens}


if __name__ == "__main__":
    import sys
    THREAD_VALUE = 25
    MIN_MEASURE = 1
    MAX_MEASURE = 8
    PROGRAM = ['PIANO', 'SAX']
    DATASETS = "/media/takaaki-nagoshi/MIDIdatasets/GMD/training"
    # ★ ext4 (/) に保存。NTFS(/media)は ntfs-3g の偽ENOSPC+I/Oエラーで大量書き込み不可だったため。
    SAVE_PATH = "/home/takaaki-nagoshi/data/scaling/ver5_noaug/music"
    SAVE_STATS = False
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    used = used_song_hashes()
    print(f"前回使用楽曲: {len(used)} 曲")
    directory, md_file = find_used_midis(DATASETS, used)
    print(f"対象MIDI(stem一致): {len(md_file)} 件")
    if LIMIT:
        directory, md_file = directory[:LIMIT], md_file[:LIMIT]
        print(f"[テスト] 先頭 {LIMIT} 件のみ変換")

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
                              PROGRAM, MIN_MEASURE, MAX_MEASURE, SAVE_STATS, progress, write_queue, SAVE_PATH))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        total_queued = sum(v["count"] for v in progress.values())

    write_queue.put(_QUEUE_SENTINEL)
    writer.join()
    r = result_queue.get()
    print(f"変換完了: {r['count']} ファイル保存  総トークン: {r['tokens']:,}")
    if total_queued != r['count']:
        print(f"[警告] キュー投入数({total_queued}) != 書き込み完了数({r['count']})")
