"""Zero-SFT 生成(infill/continuation/condgen/uncond)タスクの実生成メトリクス。
プロンプトからCONSTを実際に自己回帰生成し、参照CONST(ref_const)と比較する。
指標: 文法違反率(shift逆行), 音高クラス分布JS, 音長JS, 音符数比, seam密度段差(infill/continuation)。
生成=自作の決定的greedy(既存KVキャッシュ法はP形式prompt非対応バグのため不使用)。

使い方: python eval/eval_generation_metrics.py 80M 200   (size, n_windows/task)
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
from mortm.eval import metrics as M
from mortm.eval.metrics import parse_notes, grammar_violations, pitch_class_js, duration_js, density_step, bootstrap_ci

GEN_TASKS = ["infill", "continuation", "condgen", "uncond"]
DATA = os.environ.get("MORTM_TESTTASK", os.path.expanduser("~/data/paper/TEST-TASK"))
CFG = {
    "80M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json"),
    "10M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "scaling", "10M.json"),
}
MAX_NEW = 400


def find_ckpt(pat):
    fs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def model_set(size):
    if size == "80M":
        return {
            "A1": find_ckpt("out/models/paper/E1/A1_80M_3.2B/*_[1-9]*.pth"),
            "A2": find_ckpt("out/models/paper/E1/A2_80M_3.2B/*_[1-9]*.pth"),
        }
    return {
        a: find_ckpt(f"out/models/paper/E1/{a}_10M_400M_s42/*_[1-9]*.pth")
        for a in ["A1", "A2"]
    }


@torch.no_grad()
def generate(model, prompt, tok, dev, max_measures, max_new=MAX_NEW):
    """決定的greedy。参照と同じ小節数(max_measures)に達したら打ち切る(length-matched)。
    <TE>/<TAG_END> でも停止。生成部(prompt後)のtoken配列を返す。"""
    TE = tok.get("<TE>")
    TAG = tok.get("<TAG_END>")
    SME = tok.get("<SME>")
    x = torch.tensor(prompt, device=dev, dtype=torch.long).unsqueeze(0)
    gen = []
    sme = 0
    for _ in range(max_new):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(x, padding_mask=(x != 0), is_causal=True, is_save_cache=False)
        nt = int(logits[0, -1].float().argmax())
        if nt in (TE, TAG):
            break
        if nt == SME:
            sme += 1
            if sme > max_measures:
                break
        gen.append(nt)
        x = torch.cat([x, torch.tensor([[nt]], device=dev)], dim=1)
    return np.array(gen, dtype=np.int64)


def ref_notes(ref_const, tok):
    if not ref_const:
        return []
    clip = np.concatenate([np.asarray(v, dtype=np.int64) for v in ref_const.values()])
    return parse_notes(clip, tok)


@torch.no_grad()
def eval_model(name, ckpt, cfg, tok, dev, nwin):
    prog = _DefaultLearningProgress()
    try:
        prog.set_device(dev)
    except Exception:
        pass
    model = MORTM(MORTMArgs(cfg), prog).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    out = {}
    for task in GEN_TASKS:
        task_path = f"{DATA}/{task}.jsonl"
        if not os.path.exists(task_path):
            print(f"[warning] {task_path} does not exist. Skipping {task}.")
            continue
        lines = [json.loads(l) for l in open(task_path)][:nwin]
        rev, pcjs, dujs, ncount, seam = [], [], [], [], []
        valid = 0
        for ln in lines:
            prompt = ln["prompt"]
            ref = ln.get("ref_const")
            rn = ref_notes(ref, tok)
            ref_meas = (max((m for m, *_ in rn), default=-1) + 1) if rn else 4
            g = generate(model, prompt, tok, dev, max_measures=max(1, ref_meas))
            gn = parse_notes(g, tok)
            gv = grammar_violations(g, tok)
            rev.append(gv["reversal_rate"])
            valid += int(len(gn) > 0)
            if gn and rn:
                pcjs.append(pitch_class_js(gn, rn))
                dujs.append(duration_js(gn, rn))
                ncount.append(len(gn) / max(1, len(rn)))
                if task in ("infill", "continuation"):
                    seam.append(density_step(rn, gn))

        def ci(v):
            return bootstrap_ci(np.array(v))[0:3] if len(v) >= 5 else (float("nan"),) * 3

        out[task] = {
            "n": len(lines),
            "note_rate": valid / len(lines) if lines else 0.0,
            "reversal_rate": float(np.mean(rev)) if rev else float("nan"),
            "pc_js": ci(pcjs),
            "dur_js": ci(dujs),
            "note_count_ratio": float(np.mean(ncount)) if ncount else float("nan"),
            "seam_density_step": float(np.mean(seam)) if seam else None,
        }
        print(
            f"[{name}/{task}] note_rate={out[task]['note_rate']:.2f} rev={out[task]['reversal_rate']:.3f} "
            f"pcJS={out[task]['pc_js'][0]:.3f} ncnt={out[task]['note_count_ratio']:.2f}",
            flush=True,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "80M"
    nwin = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
    res = {}
    for name, ckpt in model_set(size).items():
        if not ckpt:
            continue
        res[name] = eval_model(name, ckpt, CFG[size], tok, dev, nwin)
    out_file = f"_gen_metrics_{size}.json"
    json.dump(res, open(out_file, "w"), indent=2, default=list)
    print(f"\n=== SUMMARY size={size} saved {out_file} ===")
    for task in GEN_TASKS:
        if task not in res.get(list(res.keys())[0], {}):
            continue
        print(f"-- {task} --")
        for name in res:
            r = res[name].get(task, {})
            if not r:
                continue
            print(
                f"  {name}: note_rate={r['note_rate']:.2f} reversal={r['reversal_rate']:.3f} "
                f"pc_JS={r['pc_js'][0]:.3f} dur_JS={r['dur_js'][0]:.3f} ncount={r['note_count_ratio']:.2f} "
                f"seam={r['seam_density_step']}"
            )


if __name__ == "__main__":
    main()
