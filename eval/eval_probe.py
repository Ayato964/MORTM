"""E3 head評価: 学習済み分類ヘッド(ClassifierModel)を TEST-TASK で評価。
music_block(=<CONST_M>[music]<TAG_END>)を mean-pool→linear→argmax。
指標: top-1 acc / top-5 / macro-F1 (+ key は MIREX)。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import LearningProgress
from train_head import ClassifierModel, build_class_sets
from eval_analysis_acc import mirex_score, macro_f1, TASKS, CFG

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _Prog(LearningProgress):
    def get_device(self):
        return DEV


def load_head(ckpt, cfg, num_classes):
    prog = _Prog()
    bargs = MORTMArgs(cfg)
    backbone = MORTM(bargs, prog)
    model = ClassifierModel(backbone, num_classes).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    model.eval()
    return model


@torch.no_grad()
def eval_head(model, lines, task, gt_map, tok, class_tokens, batch_size=64):
    SYSTEM = tok.get("<SYSTEM>")
    idx2tok = {i: int(t) for i, t in enumerate(class_tokens)}
    tok2idx = {int(t): i for i, t in enumerate(class_tokens)}
    preds = []; gts = []; n_top5 = 0; n = 0
    buf = []
    def flush(buf):
        nonlocal n_top5, n
        xs = [torch.tensor(mb, dtype=torch.long) for mb, _ in buf]
        maxlen = max(len(x) for x in xs)
        xb = torch.stack([F.pad(x, (0, maxlen - len(x)), value=0) for x in xs]).to(DEV)
        mask = (xb != 0).to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(xb, mask).float()
        for j, (_, gidx) in enumerate(buf):
            lg = logits[j]
            top5 = torch.topk(lg, min(5, lg.shape[-1])).indices.tolist()
            p = int(lg.argmax())
            preds.append(p); gts.append(gidx)
            n += 1; n_top5 += int(gidx in top5)

    for ln in lines:
        gt_tok = gt_map(ln)
        if gt_tok is None or int(gt_tok) not in tok2idx:
            continue
        prompt = list(ln["prompt"])
        mb = prompt[1:-1] if (prompt and prompt[-1] == SYSTEM) else prompt[1:]
        buf.append((mb, tok2idx[int(gt_tok)]))
        if len(buf) >= batch_size:
            flush(buf); buf = []
    if buf:
        flush(buf)

    n_ok = sum(int(p == g) for p, g in zip(preds, gts))
    f1 = macro_f1(gts, preds)
    d = {"acc": n_ok / n if n else 0.0, "top5_acc": n_top5 / n if n else 0.0,
         "macro_f1": f1, "n": n, "n_classes": len(class_tokens)}
    if task == "key":
        d["mirex"] = (sum(mirex_score(tok.rev_tokens[idx2tok[g]], tok.rev_tokens[idx2tok[p]])
                          for p, g in zip(preds, gts)) / len(preds)) if preds else 0.0
    return d


def main():
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC)); tok.mode(TO_MUSIC)
    key_tokens, dense_tokens, genre_tokens = build_class_sets(tok)
    ctoks = {"key": key_tokens, "dense": dense_tokens, "genre": genre_tokens}
    cfg = CFG["80M"]
    CK = os.path.join(_ROOT, "out", "models", "paper", "E3")

    def gt_key(ln): return ln.get("gt_key")
    def gt_genre(ln): return ln.get("gt_genre")
    def gt_dense(ln):
        gd = ln.get("gt_dense"); p = ln["programs"][0]
        return gd[p] if isinstance(gd, dict) and gd.get(p) is not None else None
    gt_fns = {"key": gt_key, "dense": gt_dense, "genre": gt_genre}

    out = {}
    for arm in ("A1", "A2"):
        out[f"{arm}-head"] = {}
        for task in ("key", "dense", "genre"):
            ckpt = f"{CK}/{arm}_{task}/best_model.pth"
            if not os.path.exists(ckpt):
                print(f"[{arm}-head/{task}] ckpt無し: {ckpt}"); continue
            if not os.path.exists(TASKS[task]):
                print(f"[{arm}-head/{task}] task file 無し: {TASKS[task]}"); continue
            lines = [json.loads(l) for l in open(TASKS[task])]
            model = load_head(ckpt, cfg, len(ctoks[task]))
            r = eval_head(model, lines, task, gt_fns[task], tok, ctoks[task])
            out[f"{arm}-head"][task] = r
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            extra = f" mirex={r.get('mirex',0):.3f}" if task == "key" else ""
            print(f"[{arm}-head/{task}] acc={r['acc']:.3f} top5={r['top5_acc']:.3f} f1={r['macro_f1']:.3f}{extra} n={r['n']}", flush=True)

    report_dir = os.path.join(_ROOT, "report", "E3", "data")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "e3_head_eval.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
