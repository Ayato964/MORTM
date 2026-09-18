"""E2 継ぎ目の客観評価: 補完CONSTの[最初の音/最後の音/内部]別に pitch の NLL と正解率を測る。
teacher-forced(デコード非依存)。infillプロンプト [PAST,FUTURE,<CONST_M>] に正解CONSTを連結し、
CONST内の各pitchトークン位置で予測分布を評価。
仮説: 継ぎ目(first=PAST接続, last=FUTURE接続)で A2≫A1(高NLL/低正解率)、内部は同程度。
特に last(FUTUREへの接続)は A2 が事前学習で未経験ゆえ顕著なはず。

中庸密度・単一PIANO窓のみ(壊れスクレイプ除外)。使い方: python eval/eval_e2_boundary.py [n]
"""
import json, sys, glob, os
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.train.train import _DefaultLearningProgress
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_MUSIC
from mortm.eval.metrics import parse_notes, bootstrap_ci

DATA_DIR = os.environ.get("MORTM_TESTTASK", os.path.expanduser("~/data/paper/TEST-TASK"))
INFILL = os.path.join(DATA_DIR, "infill.jsonl")
DENS_MIN, DENS_MAX = 3.0, 10.0
CFGS = {
    "80M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json"),
    "10M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "scaling", "10M.json"),
    "80M_40B": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json"),
    "160M_40B": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "160M.json"),
}


def find1(p):
    fs = sorted(glob.glob(p), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def model_set(size):
    if size == "80M_40B":
        return {"A1_40B_80M": "out/models/mortm/4_5/MORTM.4.5D-80M_1.1424479484558105.pth"}
    if size == "160M_40B":
        return {"A1_40B_160M": "out/models/4_5/MORTM.4.5D-160M.pth"}
    if size == "80M":
        return {
            "A2_zero": find1("out/models/paper/E1/A2_80M_3.2B/*_[1-9]*.pth"),
            "E2_final": find1("out/models/paper/E2/noaug_lr1.0/*_[1-9]*.pth"),
            "A1_zero": find1("out/models/paper/E1/A1_80M_3.2B/*_[1-9]*.pth"),
            "A1_SFT": find1("out/models/paper/E2/aug_lr1.0/*_[1-9]*.pth"),
        }
    return {
        a: find1(f"out/models/paper/E1/{a}_10M_400M_s42/*_[1-9]*.pth")
        for a in ["A1", "A2", "A3a", "A3b", "A3c"]
    }


def main():
    size = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in CFGS else "80M"
    argn = 2 if (len(sys.argv) > 1 and sys.argv[1] in CFGS) else 1
    n = int(sys.argv[argn]) if len(sys.argv) > argn else 500
    CFG = CFGS[size]
    MODELS = model_set(size)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
    P_LO, P_HI = tok.get_length_tuple("p")
    SME = tok.get("<SME>")
    ESEQ, TAG = tok.get("<ESEQ>"), tok.get("<TAG_END>")
    key_ids = torch.tensor(sorted(range(P_LO, P_HI)), device=dev)

    def dens(notes):
        if not notes:
            return 0.0
        return len(notes) / max(1, max(mm for mm, *_ in notes) + 1)

    samples = []
    if not os.path.exists(INFILL):
        print(f"[warning] {INFILL} does not exist.")
        return
    for ln in [json.loads(l) for l in open(INFILL)]:
        if ln["programs"] != ["PIANO"] or not ln.get("ref_const"):
            continue
        const = [int(x) for x in ln["ref_const"]["PIANO"]]
        cn = parse_notes(const, tok)
        if not (DENS_MIN <= dens(cn) <= DENS_MAX):
            continue
        full = list(ln["prompt"]) + [tok.get("<INST_PIANO>")] + const + [ESEQ, TAG]
        base = len(ln["prompt"]) + 1
        pitch_pos = [base + i for i, t in enumerate(const) if P_LO <= t < P_HI]
        if len(pitch_pos) < 4:
            continue
        samples.append((np.array(full, dtype=np.int64), pitch_pos))
        if len(samples) >= n:
            break
    print(f"評価窓: {len(samples)} (中庸密度・単一PIANO)", flush=True)

    def agg(vals):
        v = [x for x in vals if x == x]
        return (
            (float(np.mean(v)), *bootstrap_ci(np.array(v))[1:])
            if len(v) >= 5
            else (float(np.mean(v)) if v else float("nan"), float("nan"), float("nan"))
        )

    results = {}
    for mname, ck in MODELS.items():
        if not ck:
            continue
        prog = _DefaultLearningProgress()
        try:
            prog.set_device(dev)
        except Exception:
            pass
        m = MORTM(MORTMArgs(CFG), prog).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        m.eval()
        nll = {"first1": [], "first3": [], "interior": [], "last1": []}
        a1_ = {"first1": [], "first3": [], "last1": []}
        a5_ = {"first1": [], "first3": [], "last1": []}

        def score(lg, pos, full):
            tgt = int(full[pos])
            logit = lg[pos - 1]
            nl = float(F.cross_entropy(logit.unsqueeze(0), torch.tensor([tgt], device=dev)))
            sub = logit[key_ids]
            top5 = key_ids[torch.topk(sub, 5).indices].tolist()
            return nl, int(int(key_ids[int(sub.argmax())]) == tgt), int(tgt in top5)

        with torch.no_grad():
            for full, pp in samples:
                x = torch.tensor(full, device=dev).unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    lg = m.forward(x, padding_mask=(x != 0), is_causal=True, is_save_cache=False)[0].float()
                nl, c1, c5 = score(lg, pp[0], full)
                nll["first1"].append(nl)
                a1_["first1"].append(c1)
                a5_["first1"].append(c5)

                k = min(3, len(pp))
                sc = [score(lg, pp[i], full) for i in range(k)]
                nll["first3"].append(float(np.mean([s[0] for s in sc])))
                a1_["first3"].append(float(np.mean([s[1] for s in sc])))
                a5_["first3"].append(float(np.mean([s[2] for s in sc])))

                nl, c1, c5 = score(lg, pp[-1], full)
                nll["last1"].append(nl)
                a1_["last1"].append(c1)
                a5_["last1"].append(c5)

                if len(pp) > 2:
                    nll["interior"].append(float(np.mean([score(lg, pp[i], full)[0] for i in range(1, len(pp) - 1)])))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results[mname] = {k + "_nll": agg(nll[k]) for k in nll}
        for k in ("first1", "first3", "last1"):
            results[mname][k + "_acc1"] = float(np.mean(a1_[k]))
            results[mname][k + "_acc5"] = float(np.mean(a5_[k]))
        r = results[mname]
        print(
            f"[{mname}] NLL first1={r['first1_nll'][0]:.3f} first3={r['first3_nll'][0]:.3f} interior={r['interior_nll'][0]:.3f} last1={r['last1_nll'][0]:.3f} | "
            f"acc5 first1={r['first1_acc5']:.3f} first3={r['first3_acc5']:.3f} last1={r['last1_acc5']:.3f}",
            flush=True,
        )
    out_dir = os.path.join(_ROOT, "report", "E1", "data")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(os.path.join(out_dir, f"seam_boundary_{size}.json"), "w"), indent=2)
    print("\n=== 継ぎ目 pitch NLL [95%CI] ===")
    for k, r in results.items():
        f1 = r['first1_nll']
        f3 = r['first3_nll']
        i_ = r['interior_nll']
        l1 = r['last1_nll']
        print(f"{k:10s}{f'{f1[0]:.3f}[{f1[1]:.2f},{f1[2]:.2f}]':>18s}{f'{f3[0]:.3f}[{f3[1]:.2f},{f3[2]:.2f}]':>18s}{i_[0]:14.3f}{l1[0]:12.3f}")


if __name__ == "__main__":
    main()
