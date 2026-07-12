"""論文用 曲単位 98/1/1 split（研究設計書 v1.6 §4.2, testset-v1）。

同一曲の転調・窓・順列違いが split を跨ぐことを禁止（リーク）。ハッシュベースで曲 ID を固定し、
split ファイルを git 管理する。RNG 状態に依存しない決定論割当（md5(song_hash) mod 1000）。

割当: bucket = int(md5(song_hash).hexdigest(),16) % 1000
  0..979  -> train (98.0%)
  980..989 -> val  (1.0%)
  990..999 -> test (1.0%)

出力: docs/splits/{train,val,test}_songs.txt（1行1曲hash、ソート済、git管理）。
E1 の学習索引はこの train のみ、TEST-SEQ/TEST-TASK は test のみから構築する。
"""
import hashlib
import os
import glob

MANIFEST_DIR = "/home/takaaki-nagoshi/data/scaling/manifest_v5"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "splits")


def bucket(song_hash: str) -> int:
    return int(hashlib.md5(song_hash.encode()).hexdigest(), 16) % 1000


def split_of(song_hash: str) -> str:
    b = bucket(song_hash)
    if b < 980:
        return "train"
    if b < 990:
        return "val"
    return "test"


def load_songs():
    songs = set()
    for tsv in glob.glob(os.path.join(MANIFEST_DIR, "*.tsv")):
        with open(tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    songs.add(parts[1])
    return sorted(songs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    songs = load_songs()
    buckets = {"train": [], "val": [], "test": []}
    for s in songs:
        buckets[split_of(s)].append(s)
    for name, lst in buckets.items():
        path = os.path.join(OUT_DIR, f"{name}_songs.txt")
        with open(path, "w") as f:
            f.write("\n".join(sorted(lst)) + "\n")
    n = len(songs)
    print(f"total unique songs: {n:,}")
    for name in ("train", "val", "test"):
        c = len(buckets[name])
        print(f"  {name:5s}: {c:7,d}  {100*c/n:.2f}%  -> {OUT_DIR}/{name}_songs.txt")


if __name__ == "__main__":
    main()
