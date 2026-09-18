"""トークナイザ比較用データセット生成。
共通の S=130,500 ユニーク原曲を両アーム(提案=manifest_v5 / REMI=manifest_noaug)で使う。
eval は json_v5 と同じ 2,217 曲(原曲shift0のみ)を両アームで保持し、その全転調を train から除外。
train 曲(130,500)は eval除外後の母集合から seed=42 で選択(両アーム完全同一集合)。

出力: /home/takaaki-nagoshi/data/scaling/tokencmp/{proposed,remi}/{train.json,eval.json}
"""
import json, os, re, random

MANI_V5 = "/home/takaaki-nagoshi/data/scaling/manifest_v5"      # 提案(block移動あり)
MANI_NA = "/home/takaaki-nagoshi/data/scaling/manifest_noaug"   # REMI(移動なし)
V5_EVAL = "/home/takaaki-nagoshi/data/scaling/json_v5/eval.json"
OUT = "/home/takaaki-nagoshi/data/scaling/tokencmp"
S = 130_500
SEED = 42


def load(mdir):
    rows = []  # (path, song, shift, tok)
    for s in "0123456789abcdef":
        with open(f"{mdir}/{s}.tsv") as f:
            for line in f:
                q = line.rstrip("\n").split("\t")
                if len(q) == 4:
                    rows.append((q[0], q[1], int(q[2]), int(q[3])))
    return rows


def main():
    eval_songs = {re.sub(r"\.mid.*", "", os.path.basename(x)) for x in json.load(open(V5_EVAL))}
    na = load(MANI_NA)
    v5 = load(MANI_V5)
    na_songs = {s for _, s, _, _ in na}          # 217,024 (基準母集合)

    # train母集合 = 基準曲 - eval曲、seed=42で130,500曲選択
    pool = sorted(na_songs - eval_songs)
    rng = random.Random(SEED)
    rng.shuffle(pool)
    train_songs = set(pool[:S])
    assert len(train_songs) == S, len(train_songs)
    print(f"eval曲={len(eval_songs)}  train母集合={len(pool)}  選択train曲={len(train_songs)}")

    for arm, rows in [("proposed", v5), ("remi", na)]:
        d = os.path.join(OUT, arm)
        os.makedirs(d, exist_ok=True)
        train = [(p, t) for p, s, sh, t in rows if s in train_songs]            # 全転調
        ev = [(p, t) for p, s, sh, t in rows if s in eval_songs and sh == 0]    # 原曲のみ
        json.dump([p for p, _ in train], open(f"{d}/train.json", "w"))
        json.dump([p for p, _ in ev], open(f"{d}/eval.json", "w"))
        tt = sum(t for _, t in train); et = sum(t for _, t in ev)
        print(f"[{arm}] train: {len(train):,}npz {tt/1e9:.3f}B tok ({tt/len(train_songs):.0f}/曲) | "
              f"eval: {len(ev):,}npz {et/1e6:.1f}M tok")


if __name__ == "__main__":
    main()
