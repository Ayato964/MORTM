"""TEST-TASK 構築（研究設計書 v1.6 §6.1, testset-v1）。

test split の曲から protocol_builder で 5 タスクのプロンプトを決定論的に構築し凍結する。
各タスク N=1000 窓（曲重複なし=1曲1窓、sorted 順に前から）。P 形式（零SFT評価用）。

出力: /home/takaaki-nagoshi/data/paper/TEST-TASK/{task}.jsonl
  各行 = {"song": hash, "programs": [...], "prompt": [token...],
          "ref_const": {program:[token...]},        # 補完/生成系の参照CONST
          "gt_key": int|None, "gt_dense": {prog:int}}  # 分析系のGT(key/density)
"""
import json
import os
import glob
import random

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from mortm.eval.protocol_builder import ProtocolBuilder, TASKS

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "splits")
GMD = "/media/takaaki-nagoshi/MIDIdatasets/GMD/training"
OUT = "/home/takaaki-nagoshi/data/paper/TEST-TASK"
N_PER_TASK = 1000
PROGRAMS = ["PIANO", "SAX"]


def main():
    tk = Tokenizer(get_token_converter_pro(TO_TOKEN))
    pb = ProtocolBuilder(tk, PROGRAMS)
    klo, khi = tk.get_length_tuple("k")
    test_songs = sorted(open(os.path.join(SPLIT_DIR, "test_songs.txt")).read().split())
    sset = set(test_songs)

    # GMD の hash->path
    h2p = {}
    for dp, _, fs in os.walk(GMD):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                s = os.path.splitext(f)[0]
                if s in sset:
                    h2p[s] = os.path.join(dp, f)

    os.makedirs(OUT, exist_ok=True)
    files = {t: open(os.path.join(OUT, f"{t}.jsonl"), "w") for t in TASKS}
    counts = {t: 0 for t in TASKS}
    processed = 0

    for song in test_songs:
        if all(counts[t] >= N_PER_TASK for t in TASKS):
            break
        p = h2p.get(song)
        if not p:
            continue
        d, f = os.path.split(p)
        try:
            con = MIDIConverter(tk, d, f, PROGRAMS)
            con.convert()
            if con.is_error or con.midi2seq is None:
                continue
            maker = FoundationDataMaker(con, 1, 8)
            if maker.is_error:
                continue
            w = next(pb.windows(d, f), None)
        except Exception:
            continue
        if w is None:
            continue
        processed += 1
        key_tok = None
        try:
            meta = pb._meta_block(maker, w.info)
            kc = [int(t) for t in meta if klo <= int(t) < khi]
            key_tok = kc[0] if kc else None
        except Exception:
            pass
        gt_dense = {prog: int(d_) for prog, d_ in w.info}
        for task in TASKS:
            if counts[task] >= N_PER_TASK:
                continue
            try:
                prompt, ref = pb.build(w, maker, task)
            except Exception:
                continue
            rec = {"song": song, "programs": w.programs,
                   "prompt": [int(x) for x in prompt],
                   "ref_const": ({p_: [int(x) for x in a] for p_, a in ref.items()} if ref else None),
                   "gt_key": key_tok, "gt_dense": gt_dense}
            files[task].write(json.dumps(rec) + "\n")
            counts[task] += 1
        if processed % 200 == 0:
            print(f"processed {processed} songs, counts={counts}", flush=True)

    for fp in files.values():
        fp.close()
    print(f"[TEST-TASK] done: processed {processed} songs -> {OUT}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
