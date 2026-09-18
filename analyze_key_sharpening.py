"""SFTでtop1↑・top5↓の原因="分布の尖鋭化"仮説を検証。
key位置の softmax(keyクラス限定) の max-prob と エントロピーを zero vs sft で比較。
尖鋭化なら: sft は max-prob↑ / entropy↓ (top1決め打ち↑・近傍の裾が痩せてtop5↓)。
対象: dense-8bar analysis_key。
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import _DefaultLearningProgress
import eval_e3_analysis as E3A
from eval_analysis_acc import build_class_sets, CFG

DEV = torch.device("cuda:0")
S = "/home/takaaki-nagoshi/data/paper/TEST-TASK-MIDICAPS-DENSE-8BAR/analysis_key.jsonl"
CK_ZERO = "out/models/paper/E1/A1_80M_3.2B/MORTM.E1-A1-80M-3.2B_1.3635098934173584.pth"
CK_SFT = "out/models/paper/E3/A1_sft_800M/MORTM.E3-A1-sft-800M.train.0.0.4846.pth"


def stats(name, probs_list, correct1, correct5, n):
    mp = np.array([p.max() for p in probs_list])
    ent = np.array([-(p * np.log(p + 1e-12)).sum() for p in probs_list])
    print(f"[{name}] top1={correct1/n:.3f} top5={correct5/n:.3f} | "
          f"max-prob 平均={mp.mean():.3f} | entropy 平均={ent.mean():.3f} (nats) | n={n}", flush=True)


def main():
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC)); tok.mode(TO_MUSIC)
    ki, di, gi = build_class_sets(tok)
    key_ids = [int(x) for x in ki]; kset = set(key_ids); kt = torch.tensor(key_ids)
    cfg = CFG["80M"]
    lines = [json.loads(l) for l in open(S) if json.loads(l).get("gt_key") is not None]
    META, SYSTEM = tok.get("<META>"), tok.get("<SYSTEM>")

    # ---- zero: 制約付き(教師強制でmeta前置→key位置のlogit) ----
    prog = _DefaultLearningProgress()
    try: prog.set_device(DEV)
    except Exception: pass
    mz = MORTM(MORTMArgs(cfg), prog).to(DEV); mz.load_state_dict(torch.load(CK_ZERO, map_location=DEV)); mz.eval()
    probs_z = []; c1 = c5 = n = 0
    with torch.no_grad():
        for ln in lines:
            gtv = int(ln["gt_key"]); gd = ln.get("gt_dense", {}); pr = list(ln["prompt"])
            meta = []
            for p_ in ln["programs"]:
                meta.append(tok.get(f"<INST_{p_}>"))
                if gd.get(p_) is not None: meta.append(int(gd[p_]))
            seq = pr + meta
            x = torch.tensor([seq], device=DEV); pad = (x != 0).float().to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = mz.forward(x, padding_mask=pad, is_causal=True, is_save_cache=False)
            sub = lg[0, len(seq) - 1].float()[kt]
            p = F.softmax(sub, dim=-1).cpu().numpy()
            probs_z.append(p); n += 1
            order = np.argsort(-p)
            if key_ids[order[0]] == gtv: c1 += 1
            if gtv in [key_ids[j] for j in order[:5]]: c5 += 1
    del mz; torch.cuda.empty_cache()
    stats("A1-zero (制約付き)", probs_z, c1, c5, n)

    # ---- sft: 自由生成→key位置のlogit ----
    prompts = []
    for ln in lines:
        p = list(ln["prompt"])
        if p and p[-1] == SYSTEM: p = p[:-1]
        prompts.append(p + [META, SYSTEM])
    ms = E3A.load_model(CK_SFT, cfg, True)
    gens = E3A.generate(ms, prompts, tok)
    probs_s = []; c1 = c5 = n = 0
    for ln, (gt, gl) in zip(lines, gens):
        gtv = int(ln["gt_key"]); found = None
        for tok_i, lgv in zip(gt, gl):
            if tok_i in kset: found = lgv; break
        if found is None: continue
        sub = found[kt].float()
        p = F.softmax(sub, dim=-1).cpu().numpy()
        probs_s.append(p); n += 1
        order = np.argsort(-p)
        if key_ids[order[0]] == gtv: c1 += 1
        if gtv in [key_ids[j] for j in order[:5]]: c5 += 1
    del ms; torch.cuda.empty_cache()
    stats("A1-sft-800M (自由生成)", probs_s, c1, c5, n)


if __name__ == "__main__":
    main()
