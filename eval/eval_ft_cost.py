"""E2 (H2) Fig.2 曲線: FTトークン量に対する補完(infill)性能。
A2起点フルFTの各ckpt(32M..1.6B) を補完プロトコルで評価し、A1零SFT水準(参照線)と比較。
指標: note_rate(遂行可否), reversal(文法違反率), pc_JS(音高分布, 低=近い), seam(境界密度段差, 低=滑らか)。
生成=決定的greedy(temperature無し), 参照と同小節数で打切り。infillタスクのみ。

使い方: python eval/eval_ft_cost.py [arm] [n_windows]   例: eval_ft_cost.py noaug_lr1.0 200
"""
import json, sys, glob, os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.train.train import _DefaultLearningProgress
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_MUSIC
from mortm.eval.metrics import parse_notes, grammar_violations, pitch_class_js, density_step, bootstrap_ci
from eval_generation_metrics import generate, ref_notes

CFG = os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json")
DATA_DIR = os.environ.get("MORTM_TESTTASK", os.path.expanduser("~/data/paper/TEST-TASK"))
INFILL = os.path.join(DATA_DIR, "infill.jsonl")


def find1(pat):
    fs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def build_points(arm):
    d = f"out/models/paper/E2/{arm}"
    pts = []
    a2 = find1("out/models/paper/E1/A2_80M_3.2B/*_[1-9]*.pth")
    pts.append(("A2_zero(0M,start)", 0, a2))
    for pct, tok_m in [("p2", 32), ("p10", 160), ("p20", 320), ("p50", 800)]:
        c = f"{d}/MORTM.E2-{arm}.ckpt_{pct}.pth"
        if os.path.exists(c):
            pts.append((f"E2_{pct}({tok_m}M)", tok_m, c))
    fin = find1(f"{d}/*_[1-9]*.pth")
    if fin:
        pts.append(("E2_final(1600M)", 1600, fin))
    a1 = find1("out/models/paper/E1/A1_80M_3.2B/*_[1-9]*.pth")
    pts.append(("A1_zeroSFT(ref)", -1, a1))
    return pts


@torch.no_grad()
def eval_ckpt(ckpt, tok, dev, lines):
    prog = _DefaultLearningProgress()
    try:
        prog.set_device(dev)
    except Exception:
        pass
    m = MORTM(MORTMArgs(CFG), prog).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    rev, pcjs, seam = [], [], []
    valid = 0
    for ln in lines:
        rn = ref_notes(ln.get("ref_const"), tok)
        ref_meas = (max((mm for mm, *_ in rn), default=-1) + 1) if rn else 4
        g = generate(m, ln["prompt"], tok, dev, max_measures=max(1, ref_meas))
        gn = parse_notes(g, tok)
        rev.append(grammar_violations(g, tok)["reversal_rate"])
        valid += int(len(gn) > 0)
        if gn and rn:
            pcjs.append(pitch_class_js(gn, rn))
            seam.append(density_step(rn, gn))
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def ci(v):
        return bootstrap_ci(np.array(v))[0] if len(v) >= 5 else float("nan")

    return {
        "note_rate": valid / len(lines) if lines else 0.0,
        "reversal": float(np.mean(rev)) if rev else 0.0,
        "pc_JS": ci(pcjs),
        "seam": ci(seam),
        "n_valid": len(pcjs),
    }


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "noaug_lr1.0"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
    if not os.path.exists(INFILL):
        print(f"[warning] {INFILL} does not exist.")
        return
    lines = [json.loads(l) for l in open(INFILL)][:n]
    curve = {}
    for label, ftm, ckpt in build_points(arm):
        if not ckpt:
            print(f"{label}: no ckpt, skip")
            continue
        r = eval_ckpt(ckpt, tok, dev, lines)
        curve[label] = {"ft_tokens_M": ftm, **r}
        print(
            f"[{label}] ft={ftm}M note_rate={r['note_rate']:.2f} rev={r['reversal']:.3f} pc_JS={r['pc_JS']:.3f} seam={r['seam']:.2f}",
            flush=True,
        )
    out_file = f"_e2_curve_{arm}.json"
    json.dump({"arm": arm, "n": len(lines), "curve": curve}, open(out_file, "w"), indent=2)
    print(f"\n=== E2 Fig.2 curve (arm={arm}, infill) saved {out_file} ===")
    print(f"{'point':22s}{'ftTok':>8s}{'note_rate':>10s}{'pc_JS':>8s}{'seam':>8s}")
    for k, v in curve.items():
        print(f"{k:22s}{v['ft_tokens_M']:7d}M{v['note_rate']:10.2f}{v['pc_JS']:8.3f}{v['seam']:8.2f}")


if __name__ == "__main__":
    main()
