"""E5 (曲一致): Any-Order(完全aug) を A2 が使った曲集合Sに制限して再生成する。
- 曲集合S = data/paper/E5_songmatch_hashes.txt (A2/400M の 52,786 曲)
- 各曲を1転調(A2の曝露~1.14枚/曲に一致, seeded-random で決定的)にして full aug を適用
  → 固定順と同一の曲・同程度の転調曝露で、違いはブロックaug(順列+削除+META random)のみ
  → Any-Order は削除で系列が短くトークンが少ない → 学習側でエポック反復して400Mに一致(=ハンデ)
- 出力: /home/takaaki-nagoshi/data/scaling/ver5_A1songmatch/music (16シャード)
使い方: python scaling/ConvertFoundation_e5.py --limit 2000   (パイロット) / 省略で全S
"""
import os, re, random, sys, hashlib
from multiprocessing import Process, Manager, Queue
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from ConvertFoundation import writer_worker, _QUEUE_SENTINEL

HASHFILE = "/home/takaaki-nagoshi/PycharmProjects/MORTM/data/paper/E5_songmatch_hashes.txt"
DATASETS = "/media/takaaki-nagoshi/MIDIdatasets/GMD/training"
SAVE_PATH = "/home/takaaki-nagoshi/data/scaling/ver5_A1songmatch/music"
MIN_MEASURE, MAX_MEASURE, PROGRAM = 1, 8, ['PIANO', 'SAX']
THREAD_VALUE = 24


def used_hashes():
    with open(HASHFILE) as f:
        return set(x.strip() for x in f if x.strip())


def find_used_midis(root, used):
    direc, files = [], []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")) and re.sub(r"\.midi?$", "", f, flags=re.I) in used:
                direc.append(dp); files.append(f)
    return direc, files


def convert_worker(pid, tokenizer, directory, md_file, progress, write_queue):
    local_count = local_tokens = 0
    total = len(md_file)
    for i, (d, f) in enumerate(zip(directory, md_file), 1):
        try:
            con = MIDIConverter(tokenizer, d, f, PROGRAM)
            con.convert()
            if con.is_error:
                continue
            # 速度優先: オリジナル鍵のみ(転調生成を省く=~12倍速)。A2は転調済のため鍵分布差は限界として明記。
            m = FoundationDataMaker(con, MIN_MEASURE, MAX_MEASURE,
                                    disable_block_augment=False, augment_mode="full")
            stats = m.convert()
            if any(np.sum(a == 0) != 0 for a in m.aya_node[1:]):
                continue
            task = m.prepare_write_task(SAVE_PATH, False)
            if task is None:
                continue
            out_dir, filename, array_dict, stats_data = task
            tc = stats.get("total_tokens", 0) if stats else 0
            write_queue.put((out_dir, filename, array_dict, stats_data, tc))
            local_count += 1; local_tokens += tc
        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) EXC {f} {e}")
            continue
        if i % 500 == 0:
            print(f"[#{pid}] ({i}/{total}) count={local_count} tokens={local_tokens:,}", flush=True)
    progress[pid] = {"count": local_count, "tokens": local_tokens}


if __name__ == "__main__":
    random.seed(42)
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    used = used_hashes()
    print(f"曲集合S: {len(used)} 曲", flush=True)
    directory, md_file = find_used_midis(DATASETS, used)
    print(f"対象MIDI(stem一致): {len(md_file)} 件 -> {SAVE_PATH}", flush=True)
    if LIMIT:
        order = sorted(range(len(md_file)), key=lambda i: md_file[i])[:LIMIT]
        directory = [directory[i] for i in order]; md_file = [md_file[i] for i in order]
        print(f"[limit] 先頭 {LIMIT} 曲(ソート済)")
    os.makedirs(SAVE_PATH, exist_ok=True)
    for h in "0123456789abcdef":
        os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    directory = [x.tolist() for x in np.array_split(directory, THREAD_VALUE)]
    md_file = [x.tolist() for x in np.array_split(md_file, THREAD_VALUE)]
    write_queue = Queue(maxsize=200); result_queue = Queue()
    writer = Process(target=writer_worker, args=(write_queue, result_queue, False), daemon=False)
    writer.start()
    with Manager() as manager:
        progress = manager.dict(); procs = []
        for t in range(THREAD_VALUE):
            p = Process(target=convert_worker, args=(t, tokenizer, directory[t], md_file[t], progress, write_queue))
            procs.append(p); p.start()
        for p in procs: p.join()
    write_queue.put(_QUEUE_SENTINEL); writer.join()
    r = result_queue.get()
    print(f"[E5-full] 変換完了: {r['count']} ファイル  総トークン: {r['tokens']:,}", flush=True)
