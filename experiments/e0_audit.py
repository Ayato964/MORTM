"""E0 データ・被覆監査（研究設計書 v1.3/v1.4 §5-E0, 付録B）。

pattern_key を各サンプルの array 列から復元し、以下を算出する:
- E0-2: 順序・部分集合分布 Π の実測、r_meta(v1.2 定義=META が >=1 音楽ブロックより後)、
        付録B.1 主ラベルへの網羅 assert(`other`=到達不能が 0 であること)、
        META ドロップ率(=SYSTEM 欠落率)、真の無条件率([CONST]単独)。

stats.json は生成時に保存されていないため、データ(npz)から直接集計する。全量は 2.6M npz と
大きいので、16 シャードから層化サンプリングして推定する(--per-shard で調整、--full で全量)。

使い方:
    python experiments/e0_audit.py A1   [--per-shard 400]
    python experiments/e0_audit.py A2   [--per-shard 400]
"""
import os
import sys
import glob
import random
import collections

import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN

DATA = {
    "A1": "/media/takaaki-nagoshi/MORTM/pre_train/ver5/music",
    "A2": "/home/takaaki-nagoshi/data/scaling/ver5_noaug/music",
}
SHARDS = "0123456789abcdef"


def _markers(tk):
    return {tk.get("<SYSTEM>"): "META", tk.get("<PAST_M>"): "PAST",
            tk.get("<CONST_M>"): "CONST", tk.get("<FUTURE_M>"): "FUTURE"}


def block_order(arr, marker_map):
    """array から残存ブロックの順序列(論理名)を復元する。"""
    return [marker_map[int(t)] for t in arr if int(t) in marker_map]


def label_b1(order):
    """付録B.1 主ラベル(優先順・先勝ち)。order=残存ブロック名の順序列。"""
    pos = {b: i for i, b in enumerate(order)}   # 各ブロックの位置(重複は最初)
    has = set(order)
    meta = "META" in has
    const = "CONST" in has
    music_before_meta = meta and any(
        b in pos and pos[b] < pos["META"] for b in ("PAST", "CONST", "FUTURE"))
    p_bc = "PAST" in pos and const and pos["PAST"] < pos["CONST"]
    f_bc = "FUTURE" in pos and const and pos["FUTURE"] < pos["CONST"]
    music_before_const = const and any(
        b in pos and pos[b] < pos["CONST"] for b in ("PAST", "FUTURE"))

    if music_before_meta:                       # 1
        return "analysis"
    if const and p_bc and f_bc:                 # 2
        return "infill"
    if const and f_bc and not p_bc:             # 3 (PAST は前に無い)
        return "anticipation"
    if const and p_bc and not f_bc:             # 4
        return "continuation"
    if const and not music_before_const and ("META" in pos and pos["META"] < pos["CONST"]):  # 5
        return "meta_gen"
    if const and not music_before_const and not meta:  # 6 真の uncond(CONST の前に何もない)
        return "uncond"
    if const and not music_before_const:        # 6' CONST 先頭だが META は後(=uncond 相当)
        return "uncond"
    if not const:                               # 7
        return "no_target"
    return "other"                              # 8 到達不能


def audit(arm, per_shard):
    base = DATA[arm]
    tk = Tokenizer(get_token_converter_pro(TO_TOKEN))
    mm = _markers(tk)
    rng = random.Random(42)

    pat = collections.Counter()
    lab = collections.Counter()
    n_samples = n_meta = n_const = n_true_uncond = 0

    for sh in SHARDS:
        files = glob.glob(os.path.join(base, sh, "*.npz"))
        rng.shuffle(files)
        for f in files[:per_shard]:
            try:
                z = np.load(f)
            except Exception:
                continue
            i = 1
            while f"array{i}" in z.files:
                a = z[f"array{i}"]; i += 1
                if a.ndim == 0:
                    continue
                order = block_order(a, mm)
                n_samples += 1
                pat[",".join(order)] += 1
                lb = label_b1(order)
                lab[lb] += 1
                if "META" in order:
                    n_meta += 1
                if "CONST" in order:
                    n_const += 1
                if order == ["CONST"]:
                    n_true_uncond += 1

    print(f"\n===== E0 audit: {arm}  ({base}) =====")
    print(f"サンプル数(サンプリング): {n_samples:,}  per_shard={per_shard}")
    r_meta = lab["analysis"] / n_samples if n_samples else 0.0
    print(f"r_meta (META が >=1 音楽ブロックより後 = analysis): {r_meta:.4f}")
    print(f"META 欠落率(SYSTEM無): {1 - n_meta / n_samples:.4f}  (A1既定 p_drop=0 なら ~0)")
    print(f"真の無条件率 [CONST]単独: {n_true_uncond / n_samples:.4f}")
    print("B.1 ラベル分布:")
    for k, v in lab.most_common():
        print(f"  {k:14s} {v:8,d}  {100*v/n_samples:5.2f}%")
    other = lab.get("other", 0)
    print(f"[網羅 assert] other(到達不能) = {other}  -> {'OK' if other == 0 else '★仕様バグ'}")
    print(f"順序パターン種類数: {len(pat)}  上位:")
    for k, v in pat.most_common(8):
        print(f"  {v:8,d}  {k}")
    return {"arm": arm, "n": n_samples, "r_meta": r_meta,
            "labels": dict(lab), "other": other, "patterns": len(pat)}


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "A1"
    per_shard = 400
    if "--per-shard" in sys.argv:
        per_shard = int(sys.argv[sys.argv.index("--per-shard") + 1])
    if "--full" in sys.argv:
        per_shard = 10 ** 9
    audit(arm, per_shard)
