"""E1 確証評価 生成軸(1-4) — bf16ネイティブ・top-p0.9/T1.0・群レベル(§6.2 v1.8c)。
凍結TEST-TASK 生成4タスク(infill/continuation/condgen/uncond)で各モデルが生成し、
軸0ゲート(形式妥当)/軸1スケール一致/軸2分布JS-W1/軸3リズムR-a,R-b/軸4多様性/音域 を群レベルで算出。
使い方: python eval/eval_genaxes.py [size] [n_per_task]
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
from mortm.eval.metrics import parse_notes, grammar_violations, js_divergence, wasserstein1, _hist, context_gen_vocabulary_js
from mortm.eval import axes_v18 as AX
from eval_e2_samples import melody_only, block_from_prompt
from mortm.eval.kv_generate import kv_generate, kv_generate_batch

DATA = os.environ.get("MORTM_TESTTASK", os.path.expanduser("~/data/paper/TEST-TASK"))
TASKS = ["infill", "continuation", "condgen", "uncond"]
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
        return {"A1-40B-80M": "out/models/mortm/4_5/MORTM.4.5D-80M_1.1424479484558105.pth"}
    if size == "160M_40B":
        return {"A1-40B-160M": "out/models/4_5/MORTM.4.5D-160M.pth"}
    if size == "80M":
        return {
            "A1": find1("out/models/paper/E1/A1_80M_3.2B/*_[1-9]*.pth"),
            "A2": find1("out/models/paper/E1/A2_80M_3.2B/*_[1-9]*.pth"),
        }
    return {
        a: find1(f"out/models/paper/E1/{a}_10M_400M_s42/*_[1-9]*.pth")
        for a in ["A1", "A2", "A3a", "A3b", "A3c"]
    }


def pc_hist_group(notes_list):
    pcs = [p % 12 for notes in notes_list for _, _, p, _ in notes]
    return _hist(pcs, 12)


def dur_hist_group(notes_list):
    ds = [d for notes in notes_list for _, _, _, d in notes]
    return _hist(ds, 96 * 3 + 1)


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "80M"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    CFG = CFGS[size]
    MODELS = model_set(size)
    vocab_path = os.path.join(_ROOT, "out", "vocab_list.json")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC), load_data=vocab_path if os.path.exists(vocab_path) else None)
    km = AX.build_key_map(tok)
    data = {t: [json.loads(l) for l in open(f"{DATA}/{t}.jsonl")][:n] for t in TASKS}
    out = {}
    for mname, ck in MODELS.items():
        if not ck:
            print(f"[skip] {mname}: checkpoint not found")
            continue
        prog = _DefaultLearningProgress()
        try:
            prog.set_device(dev)
        except Exception:
            pass
        m = MORTM(MORTMArgs(CFG), prog).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        m.eval()
        out[mname] = {}
        for t in TASKS:
            recs = []
            valid = 0
            ntot = 0
            n_data = len(data[t])
            batch_size = 32 if ("80M" in size or "160M" in size) else 128
            for start_idx in range(0, n_data, batch_size):
                end_idx = min(start_idx + batch_size, n_data)
                batch_lines = data[t][start_idx:end_idx]

                batch_prompts = []
                batch_max_measures = []
                batch_gtcs = []
                batch_pasts = []

                for ln in batch_lines:
                    ref = ln.get("ref_const")
                    gtc = (
                        parse_notes(melody_only(np.concatenate([np.asarray(v) for v in ref.values()]), tok), tok)
                        if ref
                        else []
                    )
                    ref_meas = (max((mm for mm, *_ in gtc), default=-1) + 1) if gtc else 8

                    batch_prompts.append(ln["prompt"])
                    batch_max_measures.append(max(1, ref_meas))
                    batch_gtcs.append(gtc)

                    past = parse_notes(block_from_prompt(ln["prompt"], "<PAST_M>", tok), tok)
                    batch_pasts.append(past)

                batch_gens = kv_generate_batch(
                    m, batch_prompts, tok, dev, batch_max_measures,
                    p=0.9, temperature=1.0, seed=100 + start_idx
                )

                for i, g in enumerate(batch_gens):
                    gn = parse_notes(melody_only(g, tok), tok)
                    ln = batch_lines[i]
                    ntot += 1
                    valid += int(len(gn) > 0)
                    recs.append((gn, batch_gtcs[i], batch_pasts[i], ln.get("gt_key"), g, ln["prompt"]))
            gate = valid / ntot if ntot else float("nan")
            gen_notes = [r[0] for r in recs if r[0]]
            gt_notes = [r[1] for r in recs if r[1]]

            scg = [AX.scale_agreement(r[0], r[3], km) for r in recs if r[0] and r[3] in km]
            sct = [AX.scale_agreement(r[1], r[3], km) for r in recs if r[1] and r[3] in km]
            axis1 = (float(np.mean(scg)) if scg else float("nan"), float(np.mean(sct)) if sct else float("nan"))

            sc_max = [AX.all_scales_agreement(r[0])[0] for r in recs if r[0]]
            axis1_max_scale_gen = float(np.mean(sc_max)) if sc_max else float("nan")

            pcjs = (
                js_divergence(pc_hist_group(gen_notes), pc_hist_group(gt_notes))
                if gen_notes and gt_notes
                else float("nan")
            )
            durjs = (
                js_divergence(dur_hist_group(gen_notes), dur_hist_group(gt_notes))
                if gen_notes and gt_notes
                else float("nan")
            )

            vocab_js_list = [context_gen_vocabulary_js(r[4], r[5], tok) for r in recs if r[0]]
            vocab_js = float(np.nanmean(vocab_js_list)) if vocab_js_list else float("nan")

            ctx = [(r[0], r[2], r[1]) for r in recs if r[2]]
            if ctx:
                ra_gen, ra_gt = AX.groove_agreement([c[0] for c in ctx], [c[1] for c in ctx], [c[2] for c in ctx])
                rb = float(np.mean([AX.phase_js(c[0], c[1]) for c in ctx if c[0] and c[1]]))
            else:
                ra_gen = ra_gt = rb = float("nan")

            div_gen = AX.inter_sample_distinct(gen_notes)
            div_gt = AX.inter_sample_distinct(gt_notes)
            out[mname][t] = {
                "axis0_gate": gate,
                "axis1_scale_gen": axis1[0],
                "axis1_scale_gt": axis1[1],
                "axis1_max_scale_gen": axis1_max_scale_gen,
                "axis2_pcJS": pcjs,
                "axis2_durJS": durjs,
                "axis2_vocabJS": vocab_js,
                "axis3_Ra_gen": ra_gen,
                "axis3_Ra_gt": ra_gt,
                "axis3_Rb_phaseJS": rb,
                "axis4_div_gen": div_gen,
                "axis4_div_gt": div_gt,
            }
            print(
                f"[{mname}/{t}] gate={gate:.2f} scale gen/gt={axis1[0]:.3f}/{axis1[1]:.3f} (max={axis1_max_scale_gen:.3f}) "
                f"pcJS={pcjs:.3f} vocabJS={vocab_js:.3f} Ra gen/gt={ra_gen:.2f}/{ra_gt:.2f} RbJS={rb:.3f} div gen/gt={div_gen:.3f}/{div_gt:.3f}",
                flush=True,
            )
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(os.path.join(_ROOT, "report", "E1", "data"), exist_ok=True)
    out_path = os.path.join(_ROOT, "report", "E1", "data", f"genaxes_{size}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"=== GENAXES DONE size={size} saved {out_path} ===")


if __name__ == "__main__":
    main()
