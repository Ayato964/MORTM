"""SFT生成タスク データ生成 (MIDICaps, ピアノ&サックスのみ, 転調なし)。
GenerationDataMaker.convert_generation_sft() で 3タスク
(meta / meta_past / meta_future -> CONST) のシーケンスを作る。
出力: ext4 (/home/.../data/sft/generation/music) 16シャード。NTFSは書込不安定のため避ける。
"""
import json, os, re, random
from multiprocessing import Process, Manager, Queue
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import GenerationDataMaker, genre_names_to_ids
from ConvertFoundation import writer_worker, _QUEUE_SENTINEL

DATASETS = "/media/takaaki-nagoshi/MIDIdatasets/MIDI_Caps/lmd_full"
MIDICAPS_ROOT = "/media/takaaki-nagoshi/MIDIdatasets/MIDI_Caps"     # location の基準
MIDICAPS_JSON = "/media/takaaki-nagoshi/MIDIdatasets/MIDI_Caps/json/train.json"
GENRE2_DROP_PROB = 0.5   # 第2ジャンルを落とす確率(推論時に1or2ジャンル指定を両対応)
SAVE_PATH = "/home/takaaki-nagoshi/data/sft/generation/music"   # ext4
PROGRAM = ["PIANO", "SAX"]
MIN_MEASURE = 1
MAX_MEASURE = 8
THREAD_VALUE = 25
SFT_TASKS = ("meta", "meta_past", "meta_future")


def load_genre_map(json_path):
    """MIDICaps train.json (JSONL) から location -> [genre1, genre2] を構築。"""
    gm = {}
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            loc = d.get("location")
            g = d.get("genre", [])
            if loc and g:
                gm[loc] = g
    return gm


def find_midis(root):
    direc, files = [], []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                direc.append(dp); files.append(f)
    return direc, files


def convert_worker(pid, tokenizer, directory, md_file, program, min_m, max_m,
                   progress, write_queue, save_path, genre_map):
    random.seed(os.urandom(32))
    local_count = 0; local_tokens = 0
    total = len(md_file)
    for i, (d, f) in enumerate(zip(directory, md_file), 1):
        try:
            con = MIDIConverter(tokenizer, d, f, program)
            con.convert()
            if con.is_error:
                continue
            # --- ジャンル: location から引き、第2を確率ドロップ → トークンID化 ---
            location = os.path.relpath(os.path.join(d, f), MIDICAPS_ROOT)
            genres = list(genre_map.get(location, []))
            if len(genres) >= 2 and random.random() < GENRE2_DROP_PROB:
                genres = genres[:1]          # 第2ジャンルを落とす
            genre_tokens = genre_names_to_ids(genres, tokenizer)

            m = GenerationDataMaker(con, min_m, max_m)
            stats = m.convert_generation_sft(tasks=SFT_TASKS, genre_tokens=genre_tokens)
            if len(m.aya_node) <= 1:
                continue
            if any(np.sum(a == 0) != 0 for a in m.aya_node[1:]):
                print(f"[#{pid}] ({i}/{total}) ERROR zero-token {f}")
                continue
            task = m.prepare_write_task(save_path, save_stats=False)
            if task is None:
                continue
            out_dir, filename, array_dict, stats_data = task
            tok = stats.get("total_tokens", 0) if stats else 0
            write_queue.put((out_dir, filename, array_dict, stats_data, tok))
            local_count += 1; local_tokens += tok
            if i % 200 == 0:
                print(f"[#{pid}] ({i}/{total}) ok={local_count} tok={local_tokens}", flush=True)
        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) EXCEPTION {f} {e}")
            continue
    progress[pid] = {"count": local_count, "tokens": local_tokens}


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else None
    os.makedirs(SAVE_PATH, exist_ok=True)
    for h in "0123456789abcdef":
        os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    print("MIDICaps ジャンルマップ読込中...")
    genre_map = load_genre_map(MIDICAPS_JSON)
    print(f"  genre_map: {len(genre_map):,} 曲")
    directory, md_file = find_midis(DATASETS)
    print(f"MIDICaps: {len(md_file)} MIDI files (program={PROGRAM}, 転調なし, "
          f"tasks={SFT_TASKS}, 可変長1-{MAX_MEASURE}小節, ジャンル付与)")
    if limit:
        directory, md_file = directory[:limit], md_file[:limit]
        print(f"[test] 先頭 {limit} 曲のみ")

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    write_queue = Queue(maxsize=200)
    result_queue = Queue()
    writer = Process(target=writer_worker, args=(write_queue, result_queue, False), daemon=False)
    writer.start()

    with Manager() as manager:
        progress = manager.dict()
        procs = []
        for t in range(THREAD_VALUE):
            p = Process(target=convert_worker,
                        args=(t, tokenizer, directory[t].tolist(), md_file[t].tolist(),
                              PROGRAM, MIN_MEASURE, MAX_MEASURE, progress, write_queue, SAVE_PATH,
                              genre_map))
            procs.append(p); p.start()
        for p in procs: p.join()
        total_queued = sum(v["count"] for v in progress.values())

    write_queue.put(_QUEUE_SENTINEL); writer.join()
    r = result_queue.get()
    print(f"SFT生成データ完了: {r['count']} ファイル  総トークン {r['tokens']:,}")
    if total_queued != r['count']:
        print(f"[警告] キュー投入({total_queued}) != 書込完了({r['count']})")
