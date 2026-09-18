"""Table 2/3: 順方向NLL(TEST-SEQ)と continuation NLL。任意アーム × 複数シード。

- TEST-SEQ NLL: chrono の時系列held-out(=標準ARの土俵)上の全系列平均NLL。
- continuation NLL: 全方向held-out上の continuation バケツ(順方向 p(生成対象継続|過去))。
両方 同一系列ペアの差(arm - baseline)も出す(W1/曝露統制の順方向「税」検証)。

例: python eval/eval_nll.py --arms BR chrono --scale 10M --budget 400M --n 1000
"""
import sys, os, json, argparse
import numpy as np
import torch, torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import common as C, paths as P
except ImportError:
    from eval import common as C, paths as P

from mortm.eval.direction_loss import direction_masks


@torch.no_grad()
def seq_nll(model, seqs):
    out = []
    for s in seqs:
        x = torch.tensor(s, device=C.DEV).unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model.forward(x, padding_mask=(x != 0), is_causal=True, is_save_cache=False)
        out.append(float(F.cross_entropy(lg[0, :-1].float(), torch.tensor(s[1:], device=C.DEV), reduction="mean")))
    return np.array(out)


@torch.no_grad()
def cont_nll(model, seqs, masks, bs=128):
    per = []
    for st in range(0, len(seqs), bs):
        bseq, bmask = seqs[st:st+bs], masks[st:st+bs]
        ml = max(len(s) for s in bseq)
        x = torch.tensor(np.stack([np.pad(s, (0, ml-len(s))) for s in bseq]), dtype=torch.long, device=C.DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model.forward(x, padding_mask=(x != 0), is_causal=True, is_save_cache=False).float()
        for i, (s, mk) in enumerate(zip(bseq, bmask)):
            nll = F.cross_entropy(lg[i, :len(s)-1], torch.tensor(s[1:], device=C.DEV), reduction="none").cpu().numpy()
            m = mk["continuation"][1:]
            if m.any(): per.append(float(nll[m].sum())/int(m.sum()))
    return np.array(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["BR", "chrono"])
    ap.add_argument("--baseline", default="chrono", help="対応差の基準アーム")
    ap.add_argument("--scale", default="10M", choices=["10M", "80M"])
    ap.add_argument("--budget", default="400M"); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", nargs="*", default=P.SEEDS)
    ap.add_argument("--out", default="results_nll.json")
    a = ap.parse_args()
    seeds = [s for s in a.seeds if s] or [""]
    tok = C.tokenizer()
    ts = C.load_seqs(P.TESTSEQ, a.n)                                   # TEST-SEQ(順方向)
    dr = C.load_seqs(P.TESTSEQ_ALLDIR, a.n); dm = [direction_masks(s, tok) for s in dr]  # continuation
    print(f"TEST-SEQ={len(ts)} dir={len(dr)}", flush=True)
    per = {}   # arm -> {seed -> {testseq, cont, ts_arr, cont_arr}}
    for arm in a.arms:
        code = P.ARM[arm]; per[arm] = {}
        for s in seeds:
            ck = C.find_ckpt(code, a.scale, a.budget, s)
            if not ck: print(f"[skip] {arm}/{s}"); continue
            m = C.load_model(ck, a.scale)
            tn, cn = seq_nll(m, ts), cont_nll(m, dr, dm); del m; torch.cuda.empty_cache()
            per[arm][s] = {"testseq": float(tn.mean()), "cont": float(cn.mean())}
            print(f"[{arm}/{s or 'single'}] TEST-SEQ={tn.mean():.4f} cont={cn.mean():.4f}", flush=True)
    summ = {}
    for arm in a.arms:
        d = per[arm]
        summ[arm] = {k: list(C.mean_sd([d[s][k] for s in d])) for k in ("testseq", "cont") if d}
        if arm != a.baseline and per.get(a.baseline):
            for k in ("testseq", "cont"):
                shared = [s for s in d if s in per[a.baseline]]
                if shared:
                    diff = [d[s][k]-per[a.baseline][s][k] for s in shared]
                    summ[arm][f"diff_{k}_vs_{a.baseline}"] = list(C.mean_sd(diff))
    json.dump({"per_seed": per, "summary": summ}, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n=== NLL (mean±σ, ↓良; diff<0 = arm優位) ===")
    for arm in a.arms:
        o = summ[arm]
        line = f"{arm:10} TEST-SEQ={o['testseq'][0]:.4f}±{o['testseq'][1]:.4f}  cont={o['cont'][0]:.4f}±{o['cont'][1]:.4f}"
        if f"diff_testseq_vs_{a.baseline}" in o: line += f"  Δts={o['diff_testseq_vs_'+a.baseline][0]:+.4f}±{o['diff_testseq_vs_'+a.baseline][1]:.4f}"
        print(line)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
