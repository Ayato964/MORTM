"""SFT分析タスク: 予算prefix索引(50M/200M/800M)を作る。
- リークフリー分割: docs/splits の train/val/test に基づき、eval/test曲の全転調を train から除外。
- 予算 = ユニークトークンの入れ子prefix(50M ⊂ 200M ⊂ 800M)。num_epochs=1 前提。
  同一シャッフル順で累積トークンが各予算に達するまでを train_<budget>.json に書く。
- eval.json = val曲の原曲(shift 0)のみ。
出力: /home/.../data/sft/analysis/{train_50M,train_200M,train_800M,eval}.json + budget_summary.json
"""
import os, sys, json, glob, random
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_scaling_json_v5 import count_tokens_headeronly, parse_name

MUSIC = "/home/takaaki-nagoshi/data/sft/analysis/music"
OUT = "/home/takaaki-nagoshi/data/sft/analysis"
SPLIT_DIR = "/home/takaaki-nagoshi/PycharmProjects/MORTM/docs/splits"
BUDGETS = [("50M", 50_000_000), ("200M", 200_000_000), ("800M", 800_000_000)]
SEED = 1337


def load_songs(name):
    p = os.path.join(SPLIT_DIR, f"{name}_songs.txt")
    return set(open(p).read().split()) if os.path.exists(p) else set()


def _scan(paths):
    out = []
    for p in paths:
        name = os.path.basename(p)
        parsed = parse_name(name)
        if parsed is None:
            continue
        song, shift = parsed
        try:
            tok = count_tokens_headeronly(p)
        except Exception:
            continue
        out.append((p, song, shift, tok))
    return out


def main():
    train_songs = load_songs("train")
    val_songs = load_songs("val")
    test_songs = load_songs("test")
    print(f"splits: train={len(train_songs)} val={len(val_songs)} test={len(test_songs)}")

    files = glob.glob(os.path.join(MUSIC, "**", "*.npz"), recursive=True)
    print(f"npz files: {len(files):,}")

    # トークン計数(並列)
    chunks = [files[i:i+3000] for i in range(0, len(files), 3000)]
    rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, part in enumerate(ex.map(_scan, chunks), 1):
            rows.extend(part)
            print(f"  scanned chunk {i}/{len(chunks)}  rows={len(rows):,}", flush=True)

    # リークフリー: train曲の全転調のみ train。val曲shift0のみ eval。test曲は除外(TEST-TASK用)。
    train_rows = [(p, tok) for (p, song, shift, tok) in rows if song in train_songs]
    eval_rows = [p for (p, song, shift, tok) in rows if song in val_songs and shift == 0]
    print(f"train_rows(全転調)={len(train_rows):,}  eval(val shift0)={len(eval_rows):,}")

    total_train_tok = sum(t for _, t in train_rows)
    print(f"train 総ユニークトークン={total_train_tok/1e6:.1f}M")

    # 同一シャッフル順で入れ子prefix
    rng = random.Random(SEED)
    rng.shuffle(train_rows)

    os.makedirs(OUT, exist_ok=True)
    summary = {"total_train_tokens": total_train_tok, "budgets": {}}
    for label, target in BUDGETS:
        acc = 0
        paths = []
        for p, tok in train_rows:
            if acc >= target:
                break
            paths.append(p)
            acc += tok
        outp = os.path.join(OUT, f"train_{label}.json")
        with open(outp, "w") as f:
            json.dump(paths, f)
        reached = acc >= target
        summary["budgets"][label] = {"target": target, "achieved_tokens": acc,
                                     "n_files": len(paths), "fully_unique": reached}
        flag = "OK(全ユニーク)" if reached else "★不足(反復が必要)"
        print(f"  {label}: files={len(paths):,}  tokens={acc/1e6:.1f}M  {flag} -> {outp}")

    with open(os.path.join(OUT, "eval.json"), "w") as f:
        json.dump(eval_rows, f)
    with open(os.path.join(OUT, "budget_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"eval.json: {len(eval_rows):,} files")
    print("DONE")


if __name__ == "__main__":
    main()
