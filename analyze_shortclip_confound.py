"""短クリップ交絡の診断。
bench_probe --dump_preds で得た te_s順のper-clip予測を、同cacheの小節数(SMEカウント)と結合し、
「全クリップ集約 vs 小節数下限フィルタ後に集約」でfile単位accがどう変わるかを比較する。
仮説: 短い曲/短クリップが精度を落としている → フィルタで file acc が上がるはず。
"""
import json, sys, numpy as np, collections
sys.path.insert(0, ".")
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN

DS = "pianist8"
PREDS = "report/E4/pianist8_frozen80M_preds.json"
tok = Tokenizer(get_token_converter_pro(TO_TOKEN))
sme = tok.get("<SME>")

d = np.load(f"bench/cache/{DS}_test.npz", allow_pickle=True)
seqs = d["seqs"]; labels = d["labels"].astype(int); fids = d["file_ids"].astype(int)
names = [str(x) for x in d["label_names"]]
meas = np.array([int(np.sum(np.asarray(s) == sme)) for s in seqs])  # クリップ小節数

preds = json.load(open(PREDS))
assert len(preds) == len(seqs), f"len mismatch {len(preds)} vs {len(seqs)}"
# te_s順で一致確認(file_id/gt)
for i, r in enumerate(preds):
    assert r["file_id"] == int(fids[i]) and r["gt"] == int(labels[i]), f"order mismatch at {i}"
pred = np.array([r["pred"] for r in preds])


def file_acc(mask):
    """maskされたクリップだけでfile単位多数決 → (file_acc, n_files_evaluated, n_files_total)."""
    fp = collections.defaultdict(list); fg = {}
    for i in range(len(pred)):
        fg[int(fids[i])] = int(labels[i])
        if mask[i]:
            fp[int(fids[i])].append(int(pred[i]))
    all_files = sorted(fg)
    ok = 0; ev = 0
    for f in all_files:
        if fp[f]:
            p = collections.Counter(fp[f]).most_common(1)[0][0]
            ok += int(p == fg[f]); ev += 1
    # 評価対象=クリップが残ったファイルのみ。全ファイル基準accも別途。
    return ok / max(ev, 1), ev, len(all_files), ok


print(f"[{DS}] test clips={len(pred)} files={len(set(fids))}")
print(f"clip-level acc(all) = {(pred==labels).mean():.4f}")
print()
print(f"{'filter':>22} | {'clips_kept':>10} | {'files_ev':>8} | {'file_acc':>8} | {'file_acc(vs all files)':>22}")
for thr in [0, 4, 8, 12, 16, 20, 24]:
    m = meas >= thr
    fa, ev, tot, ok = file_acc(m)
    fa_allfiles = ok / tot  # 分母を全ファイルにした厳しめ版
    print(f"  measures>= {thr:>3} keep | {int(m.sum()):>10} | {ev:>8} | {fa:>8.4f} | {fa_allfiles:>22.4f}")

# 追加: file全体の総小節数でファイルを層別し、短ファイルの正誤率を見る
print("\n--- file単位: そのファイルの総小節数(全クリップ合計)別 正誤 ---")
file_totmeas = collections.defaultdict(int); file_gt = {}
fp_all = collections.defaultdict(list)
for i in range(len(pred)):
    file_totmeas[int(fids[i])] += int(meas[i]); file_gt[int(fids[i])] = int(labels[i])
    fp_all[int(fids[i])].append(int(pred[i]))
rows = []
for f in sorted(file_gt):
    p = collections.Counter(fp_all[f]).most_common(1)[0][0]
    rows.append((file_totmeas[f], int(p == file_gt[f])))
rows.sort()
# 4分位でaccを比較
tm = np.array([r[0] for r in rows]); ok = np.array([r[1] for r in rows])
qs = np.quantile(tm, [0.25, 0.5, 0.75])
print(f"file総小節数 quartiles: {qs.tolist()}  (min={tm.min()} max={tm.max()})")
for lo, hi, name in [(0, qs[0], "Q1(最短)"), (qs[0], qs[1], "Q2"), (qs[1], qs[2], "Q3"), (qs[2], 1e9, "Q4(最長)")]:
    sel = (tm >= lo) & (tm < hi) if hi < 1e9 else (tm >= lo)
    if sel.sum():
        print(f"  {name:>10} tot_meas[{lo:.0f},{hi:.0f}) : files={sel.sum():>3}  file_acc={ok[sel].mean():.4f}")
