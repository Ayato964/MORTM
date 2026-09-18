"""Table 2 / Table 4: ゼロショット分析(key/density)ACC。任意アーム × 複数シード。

例:
  python eval/eval_analysis.py --arms BR chrono --scale 10M --budget 400M   # 3シードmean±σ
  python eval/eval_analysis.py --arms BR chrono --scale 80M --budget 3.2B --seeds ""  # 単一(80M)
出力: key top-1 / macro-F1 / density top-1 を mean±σ で。
"""
import sys, os, json, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import common as C, paths as P
except ImportError:
    from eval import common as C, paths as P

import eval_analysis_acc as EA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["BR", "chrono"], help="論文アーム名(BR/chrono/perm/del/metafirst/match)")
    ap.add_argument("--scale", default="10M", choices=["10M", "80M"])
    ap.add_argument("--budget", default="400M")
    ap.add_argument("--seeds", nargs="*", default=P.SEEDS, help='空文字列で単一(seedタグ無し)モデル')
    ap.add_argument("--out", default="results_analysis.json")
    a = ap.parse_args()
    seeds = [s for s in a.seeds if s] or [""]
    tok = C.tokenizer(); ki, di, gi = EA.build_class_sets(tok)
    res = {}
    for arm in a.arms:
        code = P.ARM[arm]; R = {"key": [], "key_f1": [], "dense": []}
        for s in seeds:
            ck = C.find_ckpt(code, a.scale, a.budget, s)
            if not ck:
                print(f"[skip] {arm}/{s}: ckpt無し"); continue
            r = EA.eval_model(f"{arm}-{s}", ck, P.CFG[a.scale], tok, ki, di, gi, C.DEV)
            if "key" in r:
                R["key"].append(r["key"]["acc"])
                R["key_f1"].append(r["key"]["macro_f1"])
            if "dense" in r:
                R["dense"].append(r["dense"]["acc"])
            print(f"[{arm}/{s or 'single'}] key={r.get('key',{}).get('acc',0):.4f} f1={r.get('key',{}).get('macro_f1',0):.4f} dense={r.get('dense',{}).get('acc',0):.4f}", flush=True)
        res[arm] = {k: list(C.mean_sd(R[k])) for k in R if R[k]}
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n=== 分析 (mean±σ) ===")
    for arm in a.arms:
        o = res.get(arm, {})
        if o:
            k_acc = f"{o['key'][0]:.3f}±{o['key'][1]:.3f}" if "key" in o else "N/A"
            k_f1 = f"{o['key_f1'][0]:.3f}±{o['key_f1'][1]:.3f}" if "key_f1" in o else "N/A"
            d_acc = f"{o['dense'][0]:.3f}±{o['dense'][1]:.3f}" if "dense" in o else "N/A"
            print(f"{arm:10} key={k_acc}  F1={k_f1}  dense={d_acc}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
