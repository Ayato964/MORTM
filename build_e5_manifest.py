"""E5 (曲一致) の学習manifestを作る。
全52,786曲を必ず含めつつ、manifest内でnpzを反復して合計トークンをちょうど400Mに合わせる
(重複エントリ=反復学習=Any-Orderが負う「同一曲反復」ハンデ)。num_epochs=1で消費。
使い方: python build_e5_manifest.py [target_tokens_M]
"""
import json, os, glob, sys, random
import numpy as np

SRC = "/home/takaaki-nagoshi/data/scaling/ver5_A1songmatch/music"
OUT = "/home/takaaki-nagoshi/data/paper/A1songmatch/400M"
TARGET = (float(sys.argv[1]) if len(sys.argv) > 1 else 400.0) * 1e6

def tok(p):
    try:
        with np.load(p, allow_pickle=True) as d:
            return sum(len(np.asarray(d[k])) for k in d.files if np.asarray(d[k]).ndim > 0)
    except Exception:
        return 0

def song_of(p):
    return os.path.basename(p).split(".mid")[0]

def main():
    fs = sorted(glob.glob(f"{SRC}/*/*.npz"))
    toks = {p: tok(p) for p in fs}
    fs = [p for p in fs if toks[p] > 0]
    U = sum(toks[p] for p in fs)
    songs = set(song_of(p) for p in fs)
    print(f"npz={len(fs)} ユニーク曲={len(songs)} ユニークトークン={U/1e6:.1f}M 目標={TARGET/1e6:.0f}M")
    rng = random.Random(42)
    manifest = list(fs)                 # 全npz(全曲)を1回=ユニーク被覆
    cur = U
    pool = list(fs); rng.shuffle(pool); i = 0
    while cur < TARGET:                  # 目標まで反復追加
        p = pool[i % len(pool)]; i += 1
        manifest.append(p); cur += toks[p]
    rng.shuffle(manifest)
    os.makedirs(OUT, exist_ok=True)
    json.dump(manifest, open(f"{OUT}/train.json", "w"))
    rep = len(manifest) / len(fs)
    print(f"→ {OUT}/train.json: {len(manifest)}エントリ 合計{cur/1e6:.1f}M 反復係数{rep:.2f}x 曲被覆{len(songs)}")

if __name__ == "__main__":
    main()
