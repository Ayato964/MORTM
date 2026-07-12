"""論文用リークフリー学習索引の生成（研究設計書 v1.6 §4.2, testset-v1）。

曲単位 98/1/1 split(docs/splits/*.txt)に基づき、A1/A2 両アームの
train(トークン予算別)/val/test 索引を作り直す。test/val 曲は train から完全除外(リーク防止)。
両アームは同一 split・同一 seed を使う(前処理共通・公平性 §4.2)。

TEST-SEQ(時系列順・全ブロック)= A2(noaug=固定順)の test 索引がそれに相当。
TEST-TASK(5タスク×窓)は別途 protocol_builder で構築(make_paper_testtask.py 予定)。

manifest: (path \t song_hash \t shift \t tokens)。A1=manifest_v5, A2=manifest_noaug。
出力: /home/takaaki-nagoshi/data/paper/{A1,A2}/{200M,400M,800M,1.6B,3.2B}/train.json,
       同 {A1,A2}/val.json, {A1,A2}/test.json。
"""
import json
import os
import glob
import random

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "splits")
OUT_BASE = "/home/takaaki-nagoshi/data/paper"
ARMS = {
    "A1": "/home/takaaki-nagoshi/data/scaling/manifest_v5",
    "A2": "/home/takaaki-nagoshi/data/scaling/manifest_noaug",
}
TOKEN_TARGETS = [(200_000_000, "200M"), (400_000_000, "400M"), (800_000_000, "800M"),
                 (1_600_000_000, "1.6B"), (3_200_000_000, "3.2B")]
SEED = 42


def load_split(name):
    return set(open(os.path.join(SPLIT_DIR, f"{name}_songs.txt")).read().split())


def load_manifest(mdir):
    rows = []
    for tsv in sorted(glob.glob(os.path.join(mdir, "*.tsv"))):
        with open(tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4:
                    rows.append((p[0], p[1], int(p[3])))  # path, song, tokens
    return rows


def write_json(path, paths):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(paths, f)


def main():
    train_s, val_s, test_s = load_split("train"), load_split("val"), load_split("test")
    print(f"split: train {len(train_s):,} / val {len(val_s):,} / test {len(test_s):,}")
    rng = random.Random(SEED)
    for arm, mdir in ARMS.items():
        rows = load_manifest(mdir)
        tr = [(p, t) for p, s, t in rows if s in train_s]
        va = [p for p, s, t in rows if s in val_s]
        te = [p for p, s, t in rows if s in test_s]
        rng.shuffle(tr)
        out = os.path.join(OUT_BASE, arm)
        write_json(os.path.join(out, "val.json"), sorted(va))
        write_json(os.path.join(out, "test.json"), sorted(te))
        # 予算別 train(トークン累積)
        cum = 0
        paths = []
        ti = 0
        reached = {}
        for p, t in tr:
            cum += t
            paths.append(p)
            while ti < len(TOKEN_TARGETS) and cum >= TOKEN_TARGETS[ti][0]:
                label = TOKEN_TARGETS[ti][1]
                write_json(os.path.join(out, label, "train.json"), list(paths))
                reached[label] = (len(paths), cum)
                ti += 1
        print(f"\n[{arm}] train系列={len(tr):,} val={len(va):,} test={len(te):,}  (train総トークン={cum:,})")
        for _, label in TOKEN_TARGETS:
            if label in reached:
                n, c = reached[label]
                print(f"  {label}: {n:,} 系列 / {c:,} tok -> {out}/{label}/train.json")
            else:
                print(f"  {label}: [不足] train総{cum:,} < 目標")


if __name__ == "__main__":
    main()
