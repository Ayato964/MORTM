"""ver5_noaug (block拡張なし) から スケーリング用 json_v5_noaug を生成。
v5 と公平に比較できるよう eval は json_v5/eval.json と同じ曲(原曲のみ)を使い、
その全転調を train から除外。train はそれ以外の全曲(全転調)を seed=42 でシャッフルし
累積 200M/400M/800M/1.6B/3.2B を作る。

使い方:
    python make_scaling_json_noaug.py scan    # マニフェスト作成 (ヘッダのみ, ~数分)
    python make_scaling_json_noaug.py build   # json生成 (数秒)
"""
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor

from make_scaling_json_v5 import count_tokens_headeronly, parse_name

DATA_BASE = "/home/takaaki-nagoshi/data/scaling/ver5_noaug/music"
SHARDS = "0123456789abcdef"
MANIFEST_DIR = "/home/takaaki-nagoshi/data/scaling/manifest_noaug"
JSON_BASE = "/home/takaaki-nagoshi/data/scaling/json_v5_noaug"
V5_EVAL = "/home/takaaki-nagoshi/data/scaling/json_v5/eval.json"

TOKEN_TARGETS = [
    (200_000_000, "200M"), (400_000_000, "400M"), (800_000_000, "800M"),
    (1_600_000_000, "1.6B"), (3_200_000_000, "3.2B"),
]
SEED = 42


def _scan_chunk(args):
    shard_dir, names = args
    rows = []
    for n in names:
        parsed = parse_name(n)
        if parsed is None:
            continue
        song, shift = parsed
        p = os.path.join(shard_dir, n)
        try:
            tokens = count_tokens_headeronly(p)
        except Exception as e:
            print(f"[scan] ERROR {p}: {e}", flush=True)
            continue
        rows.append(f"{p}\t{song}\t{shift}\t{tokens}")
    return rows


def scan(workers=12, chunk_size=2000):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    for shard in SHARDS:
        out_tsv = os.path.join(MANIFEST_DIR, f"{shard}.tsv")
        if os.path.exists(out_tsv):
            print(f"[scan] shard {shard}: SKIP", flush=True)
            continue
        shard_dir = os.path.join(DATA_BASE, shard)
        names = [n for n in os.listdir(shard_dir) if n.endswith(".npz")]
        chunks = [(shard_dir, names[i:i + chunk_size]) for i in range(0, len(names), chunk_size)]
        tmp = out_tsv + ".tmp"
        with open(tmp, "w") as f, ProcessPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(_scan_chunk, chunks):
                if rows:
                    f.write("\n".join(rows) + "\n")
        os.replace(tmp, out_tsv)
        print(f"[scan] shard {shard}: DONE", flush=True)


def load_manifest():
    rows = []
    for shard in SHARDS:
        tsv = os.path.join(MANIFEST_DIR, f"{shard}.tsv")
        with open(tsv) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p, song, shift, tokens = line.split("\t")
                rows.append((p, song, int(shift), int(tokens)))
    return rows


def build():
    rows = load_manifest()
    total = sum(t for _, _, _, t in rows)
    songs = sorted({s for _, s, _, _ in rows})
    print(f"manifest: {len(rows):,} files / {len(songs):,} songs / {total:,} tokens")

    # eval は v5 と同じ曲集合(原曲shift0のみ)
    eval_songs = {re.sub(r"\.mid.*", "", os.path.basename(x)) for x in json.load(open(V5_EVAL))}
    print(f"v5 eval songs: {len(eval_songs)}")

    eval_files = [(p, t) for p, s, sh, t in rows if s in eval_songs and sh == 0]
    train_files = [(p, t) for p, s, sh, t in rows if s not in eval_songs]

    os.makedirs(JSON_BASE, exist_ok=True)
    eval_paths = [p for p, _ in eval_files]
    with open(os.path.join(JSON_BASE, "eval.json"), "w") as f:
        json.dump(eval_paths, f, indent=2)
    print(f"eval.json : {len(eval_paths):,} files  {sum(t for _, t in eval_files):,} tokens")

    rng = random.Random(SEED)
    rng.shuffle(train_files)
    cum = 0
    paths = []
    ti = 0
    for p, t in train_files:
        cum += t
        paths.append(p)
        while ti < len(TOKEN_TARGETS) and cum >= TOKEN_TARGETS[ti][0]:
            label = TOKEN_TARGETS[ti][1]
            d = os.path.join(JSON_BASE, label)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "train.json"), "w") as f:
                json.dump(paths, f, indent=2)
            print(f"{label}/train.json: {len(paths):,} files  {cum:,} tokens")
            ti += 1
        if ti >= len(TOKEN_TARGETS):
            break
    if ti < len(TOKEN_TARGETS):
        for i in range(ti, len(TOKEN_TARGETS)):
            print(f"[WARNING] データ不足: {TOKEN_TARGETS[i][1]} ({cum:,}/{TOKEN_TARGETS[i][0]:,})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "scan":
        scan()
    elif mode == "build":
        build()
    else:
        print(__doc__)
        sys.exit(1)
