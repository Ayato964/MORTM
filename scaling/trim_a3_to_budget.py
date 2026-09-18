"""A3a/b/c の過剰生成npzを ~400M トークンにトリムし、train.json 索引を作る。
ディスク満杯対策。ENOSPCで生じた0バイト/壊れnpzも除去する。
- token予算: 目標400M, keep上限420M(余裕)。cum到達後の残npz + 壊れnpzを削除。
- 系列カウント: 40 < len < 5000 (10M config min_length/position_length に一致)。
- 全npzはpaper train曲由来(生成時フィルタ)なのでリークフリー。
- val は学習時の内部split(train_dataset_split=0.99)を使うため別途不要。
出力: /home/takaaki-nagoshi/data/paper/<ARM>/400M/train.json
"""
import os, glob, json, sys
import numpy as np

ARMS = {"A3a": "ver5_A3a", "A3b": "ver5_A3b", "A3c": "ver5_A3c"}
SRC = "/home/takaaki-nagoshi/data/scaling"
OUT = "/home/takaaki-nagoshi/data/paper"
TARGET = 400_000_000
KEEP_CAP = 420_000_000
MINLEN, MAXLEN = 40, 5000
LOG = "/home/takaaki-nagoshi/PycharmProjects/MORTM/_a3trim.txt"


def npz_tokens(path):
    """有効系列(40<len<5000)の総トークン。壊れていれば None。"""
    try:
        with np.load(path, allow_pickle=True, mmap_mode="r") as d:
            tot, i = 0, 1
            while f"array{i}" in d.files:
                a = d[f"array{i}"]
                if getattr(a, "ndim", 0) > 0 and MINLEN < len(a) < MAXLEN:
                    tot += int(len(a))
                i += 1
        return tot
    except Exception:
        return None


def main():
    lines = []
    for arm, sub in ARMS.items():
        root = os.path.join(SRC, sub, "music")
        all_npz = sorted(glob.glob(os.path.join(root, "*", "*.npz")))
        keep, delete, corrupt = [], [], []
        cum = 0
        for p in all_npz:
            if cum >= KEEP_CAP:
                delete.append(p)
                continue
            t = npz_tokens(p)
            if t is None or t == 0:
                corrupt.append(p)
                continue
            keep.append(p)
            cum += t
        # 削除実行(残npz + 壊れnpz)
        freed = 0
        for p in delete + corrupt:
            try:
                freed += os.path.getsize(p)
                os.remove(p)
            except OSError:
                pass
        out_dir = os.path.join(OUT, arm, "400M")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train.json"), "w") as f:
            json.dump(keep, f)
        lines.append(
            f"[{arm}] kept={len(keep):,}npz cum_tokens={cum:,} "
            f"deleted={len(delete):,} corrupt={len(corrupt):,} freed={freed/1e9:.2f}GB "
            f"-> {out_dir}/train.json"
        )
    with open(LOG, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
