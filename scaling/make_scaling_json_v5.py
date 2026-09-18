"""ver5 (12転調) データから スケーリング則実験用のデータセット JSON を生成する。

フェーズ1 (scan): 全 npz の zip ヘッダのみ読んでトークン数を数え、シャード毎の
                 マニフェスト TSV (path \t song_hash \t shift \t tokens) を作る。
                 シャード単位でレジューム可能（既存 TSV はスキップ）。
フェーズ2 (build): マニフェストから eval/train を曲単位で分離し、
                  累積 200M/400M/800M/1.6B/3.2B の train.json を生成。

使い方:
    python make_scaling_json_v5.py scan    # マニフェスト作成 (~30-60min, 1回だけ)
    python make_scaling_json_v5.py build   # JSON 生成 (数秒)
"""

import json
import os
import random
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy.lib.format as npy_fmt

DATA_BASE = "/media/takaaki-nagoshi/MORTM/pre_train/ver5/music"
SHARDS = "0123456789abcdef"
MANIFEST_DIR = "/home/takaaki-nagoshi/data/scaling/manifest_v5"
JSON_BASE = "/home/takaaki-nagoshi/data/scaling/json_v5"

TOKEN_TARGETS = [
    (200_000_000, "200M"),
    (400_000_000, "400M"),
    (800_000_000, "800M"),
    (1_600_000_000, "1.6B"),
    (3_200_000_000, "3.2B"),
]

EVAL_RATIO = 0.01  # 曲単位で 1% を eval に（原曲のみ使用、全12転調を train から除外）
SEED = 42

_SHIFT_RE = re.compile(r"\.mid(?:_shift_(-?\d+))?\.npz$")


def count_tokens_headeronly(path: str) -> int:
    """npz の各 .npy メンバのヘッダだけ読み、1次元以上の配列の要素数合計を返す。"""
    total = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as f:
                version = npy_fmt.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = npy_fmt.read_array_header_1_0(f)
                else:
                    shape, _, _ = npy_fmt.read_array_header_2_0(f)
                if len(shape) > 0:
                    n = 1
                    for s in shape:
                        n *= s
                    total += n
    return total


def parse_name(name: str):
    """ファイル名 -> (song_hash, shift)。マッチしなければ None。"""
    m = _SHIFT_RE.search(name)
    if not m:
        return None
    song = name[: m.start()]
    shift = int(m.group(1)) if m.group(1) is not None else 0
    return song, shift


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


def scan(workers: int = 12, chunk_size: int = 2000):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    for shard in SHARDS:
        out_tsv = os.path.join(MANIFEST_DIR, f"{shard}.tsv")
        if os.path.exists(out_tsv):
            print(f"[scan] shard {shard}: SKIP (exists)", flush=True)
            continue
        shard_dir = os.path.join(DATA_BASE, shard)
        names = [n for n in os.listdir(shard_dir) if n.endswith(".npz")]
        chunks = [(shard_dir, names[i : i + chunk_size]) for i in range(0, len(names), chunk_size)]
        done = 0
        tmp = out_tsv + ".tmp"
        with open(tmp, "w") as f, ProcessPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(_scan_chunk, chunks):
                f.write("\n".join(rows) + "\n")
                done += len(rows)
                print(f"[scan] shard {shard}: {done}/{len(names)}", flush=True)
        os.replace(tmp, out_tsv)
        print(f"[scan] shard {shard}: DONE -> {out_tsv}", flush=True)


def load_manifest():
    rows = []  # (path, song, shift, tokens)
    for shard in SHARDS:
        tsv = os.path.join(MANIFEST_DIR, f"{shard}.tsv")
        if not os.path.exists(tsv):
            raise FileNotFoundError(f"manifest がありません: {tsv} (先に scan を実行)")
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
    total_tokens = sum(t for _, _, _, t in rows)
    songs = sorted({song for _, song, _, _ in rows})
    print(f"manifest: {len(rows):,} files / {len(songs):,} songs / {total_tokens:,} tokens")

    rng = random.Random(SEED)
    rng.shuffle(songs)
    n_eval = max(1, int(len(songs) * EVAL_RATIO))
    eval_songs = set(songs[:n_eval])

    # eval = eval曲の原曲(shift 0)のみ / train = eval曲の全転調を除外した残り全部
    eval_files = [(p, t) for p, song, shift, t in rows if song in eval_songs and shift == 0]
    train_files = [(p, t) for p, song, shift, t in rows if song not in eval_songs]

    os.makedirs(JSON_BASE, exist_ok=True)
    eval_paths = [p for p, _ in eval_files]
    eval_json = os.path.join(JSON_BASE, "eval.json")
    with open(eval_json, "w") as f:
        json.dump(eval_paths, f, indent=2)
    print(f"eval.json : {len(eval_paths):,} files  {sum(t for _, t in eval_files):,} tokens -> {eval_json}")

    rng.shuffle(train_files)
    cumulative = 0
    paths = []
    target_idx = 0
    for p, t in train_files:
        cumulative += t
        paths.append(p)
        while target_idx < len(TOKEN_TARGETS) and cumulative >= TOKEN_TARGETS[target_idx][0]:
            label = TOKEN_TARGETS[target_idx][1]
            out_dir = os.path.join(JSON_BASE, label)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "train.json")
            with open(out_path, "w") as f:
                json.dump(paths, f, indent=2)
            print(f"{label}/train.json: {len(paths):,} files  {cumulative:,} tokens -> {out_path}")
            target_idx += 1
        if target_idx >= len(TOKEN_TARGETS):
            break

    if target_idx < len(TOKEN_TARGETS):
        for i in range(target_idx, len(TOKEN_TARGETS)):
            print(f"[WARNING] データ不足: {TOKEN_TARGETS[i][1]} ({cumulative:,} / {TOKEN_TARGETS[i][0]:,})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "scan":
        scan()
    elif mode == "build":
        build()
    else:
        print(__doc__)
        sys.exit(1)
