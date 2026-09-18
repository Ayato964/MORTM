"""Table 1 / Table 4: 補完の継ぎ目(位置層別)NLL。first1/first3/interior/last1 を任意アーム×複数シードで。

既存 eval_e2_boundary.py の計算(378窓・中庸密度・単一PIANO)を再利用し、model_set を差し替えて
アーム×シードで回す。出力: 各位置の mean±σ。

例: python eval/eval_boundary.py --arms BR chrono perm del --scale 10M --budget 400M
"""
import sys, os, json, glob, shutil, argparse
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

import eval_e2_boundary as EB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["BR", "chrono", "perm", "del"])
    ap.add_argument("--scale", default="10M", choices=["10M", "80M"])
    ap.add_argument("--budget", default="400M")
    ap.add_argument("--n", type=int, default=378)
    ap.add_argument("--seeds", nargs="*", default=P.SEEDS)
    ap.add_argument("--out", default="results_boundary.json")
    a = ap.parse_args()
    seeds = [s for s in a.seeds if s] or [""]
    pairs = {f"{arm}__{s}": C.find_ckpt(P.ARM[arm], a.scale, a.budget, s)
             for arm in a.arms for s in seeds}
    pairs = {k: v for k, v in pairs.items() if v}
    EB.model_set = lambda size: pairs
    seamfile = os.path.join(_ROOT, "report", "E1", "data", f"seam_boundary_{a.scale}.json")
    os.makedirs(os.path.dirname(seamfile), exist_ok=True)
    bak = seamfile + ".repro_bak"
    if os.path.exists(seamfile):
        shutil.copy(seamfile, bak)
    sys.argv = ["x", a.scale, str(a.n)]
    EB.main()
    d = json.load(open(seamfile))
    if os.path.exists(bak):
        shutil.move(bak, seamfile)
    res = {}
    for arm in a.arms:
        got = {pos: [d[f"{arm}__{s}"][f"{pos}_nll"][0] for s in seeds if f"{arm}__{s}" in d]
               for pos in ("first1", "first3", "interior", "last1")}
        res[arm] = {pos: list(C.mean_sd(v)) for pos, v in got.items() if v}
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("\n=== 継ぎ目NLL (mean±σ, ↓良) ===")
    print(f"{'arm':10}{'first1':>16}{'first3':>16}{'interior':>16}{'last1':>16}")
    for arm in a.arms:
        o = res.get(arm, {})
        if o:
            print(f"{arm:10}" + "".join(f"{o[p][0]:.3f}±{o[p][1]:.3f}".rjust(16) for p in ("first1", "first3", "interior", "last1")))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
