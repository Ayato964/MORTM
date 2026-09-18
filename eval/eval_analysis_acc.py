"""Zero-SFT 推論(analysis)タスクの ACC / macro-F1 評価。
analysisは分類タスク(音楽→key/density推論)。teacher-forcingで各予測位置のlogitを、
候補クラス(key集合/density集合)に制限してargmax=強制選択分類し、gtと比較。
META構造(make_system_prompt): <SYSTEM> [<INST_p> dense_p]* k_key <TAG_END>。
  density_p 位置の予測 prefix=..<INST_p> / key 位置の予測 prefix=..全INST+dense。

使い方: python eval/eval_analysis_acc.py 80M   /  python eval/eval_analysis_acc.py 10M
"""
import json, sys, glob, os, collections
from collections import defaultdict
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

DATA_DIR = os.environ.get("MORTM_TESTTASK", os.path.expanduser("~/data/paper/TEST-TASK"))
TASKS = {
    "key": os.path.join(DATA_DIR, "analysis_key.jsonl"),
    "dense": os.path.join(DATA_DIR, "analysis_dense.jsonl"),
    "genre": os.path.join(DATA_DIR, "analysis_genre.jsonl"),
}
CFG = {
    "80M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json"),
    "10M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "scaling", "10M.json"),
}

ENHARMONIC_MAP = {
    "k_C#M": "k_DbM",
    "k_D#M": "k_EbM",
    "k_F#M": "k_GbM",
    "k_G#M": "k_AbM",
    "k_A#M": "k_BbM",
    "k_C#m": "k_Dbm",
    "k_D#m": "k_Ebm",
    "k_F#m": "k_Gbm",
    "k_G#m": "k_Abm",
    "k_A#m": "k_Bbm"
}


def normalize_key_enharmonics(key_name):
    return ENHARMONIC_MAP.get(key_name, key_name)


def parse_key_to_pc_mode(key_name):
    k = key_name[2:] if key_name.startswith("k_") else key_name
    if k in ("Unknown", "None", "<TAG_END>"):
        return None
    mode = 'M' if k.endswith('M') else 'm'
    root = k[:-1]
    
    root_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
        'A#': 10, 'Bb': 10, 'B': 11
    }
    pc = root_to_pc.get(root)
    if pc is None:
        return None
    return (pc, mode)


def mirex_score(gt_name, pred_name):
    if gt_name == pred_name:
        return 1.0
    p1 = parse_key_to_pc_mode(gt_name)
    p2 = parse_key_to_pc_mode(pred_name)
    if p1 is None or p2 is None:
        return 0.0
    pc1, m1 = p1
    pc2, m2 = p2
    
    if pc1 == pc2 and m1 == m2:
        return 1.0  # Enharmonic match
    
    if m1 == m2:
        if (pc1 - pc2) % 12 in (5, 7):
            return 0.5
    else:
        if pc1 == pc2:
            return 0.2
        pc_M = pc1 if m1 == 'M' else pc2
        pc_m = pc1 if m1 == 'm' else pc2
        if (pc_M - pc_m) % 12 == 3:
            return 0.3
            
    return 0.0


def find_ckpt(pat):
    fs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def model_set(size):
    if size == "80M":
        return {"A1": find_ckpt("out/models/paper/E1/A1_80M_3.2B/*_[1-9]*.pth"),
                "A2": find_ckpt("out/models/paper/E1/A2_80M_3.2B/*_[1-9]*.pth")}
    return {a: find_ckpt(f"out/models/paper/E1/{a}_10M_400M_s42/*_[1-9]*.pth")
            for a in ["A1", "A2", "A3a", "A3b", "A3c"]}


def build_class_sets(tok):
    keys = sorted(i for n, i in tok.tokens.items() if str(n).startswith("k_"))
    dens = sorted(i for n, i in tok.tokens.items()
                  if ("DENSE" in str(n).upper() or "DENSITY" in str(n).upper()
                      or str(n).upper().startswith("<NOTE_DENSE")))
    genres = sorted(i for n, i in tok.tokens.items() if str(n).startswith("<GENRE_"))
    return np.array(keys), np.array(dens), np.array(genres)


@torch.no_grad()
def eval_model(name, ckpt, cfg, tok, key_ids, dense_ids, genre_ids, dev, batch_size=128, use_trigger=False):
    prog = _DefaultLearningProgress()
    try: prog.set_device(dev)
    except Exception: pass
    
    m_args = MORTMArgs(cfg)
    is_sft = "sft" in name.lower() or "sft" in ckpt.lower()
    if is_sft:
        m_args.use_attn_lora = True
        m_args.use_ffn_lora = True
        m_args.lora_r = 4
        
    model = MORTM(m_args, prog).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    torch.manual_seed(0)
    res = {}
    
    META = tok.get("<META>")
    SYSTEM = tok.get("<SYSTEM>")
    TAG_END = tok.get("<TAG_END>")
    
    for which, ids in (("key", key_ids), ("dense", dense_ids), ("genre", genre_ids)):
        task_path = TASKS[which]
        if not os.path.exists(task_path):
            print(f"[warning] {task_path} does not exist. Skipping {which}.")
            continue
        lines = [json.loads(l) for l in open(task_path)]
        ids_t = torch.tensor(ids, device=dev)
        n_ok = 0; n = 0; emit_ok = 0; n_top5 = 0
        preds = []; gts = []
        
        n_lines = len(lines)
        for start_idx in range(0, n_lines, batch_size):
            end_idx = min(start_idx + batch_size, n_lines)
            batch_lines = lines[start_idx:end_idx]
            
            batch_fulls = []
            batch_key_pos = []
            batch_dense_pos = []
            batch_genre_pos = []
            batch_gts = []
            batch_progs = []
            
            for ln in batch_lines:
                prompt = list(ln["prompt"])
                progs = ln["programs"]
                
                if is_sft:
                    music = prompt[:-1] if (prompt and prompt[-1] == SYSTEM) else list(prompt)
                    base = music + [META, SYSTEM]
                    gt_dense = ln.get("gt_dense")
                    def _dens(p_):
                        return int(gt_dense[p_]) if (isinstance(gt_dense, dict) and gt_dense.get(p_) is not None) else None
                    if which == "key":
                        if ln["gt_key"] is None:
                            continue
                        gt_key = int(ln["gt_key"])
                        meta = []
                        for p_ in progs:
                            meta.append(tok.get(f"<INST_{p_}>"))
                            dv = _dens(p_)
                            if dv is not None:
                                meta.append(dv)
                        pos = len(base) + len(meta)
                        full = np.array(base + meta + [gt_key, TAG_END], dtype=np.int64)
                        batch_fulls.append(full); batch_key_pos.append(pos)
                        batch_gts.append(gt_key); batch_progs.append(progs)
                    elif which == "dense":
                        target_p = progs[0]
                        if _dens(target_p) is None:
                            continue
                        meta = []; done = False
                        for p_ in sorted(progs):
                            meta.append(tok.get(f"<INST_{p_}>"))
                            if p_ == target_p:
                                pos = len(base) + len(meta)
                                full = np.array(base + meta + [_dens(p_), TAG_END], dtype=np.int64)
                                batch_fulls.append(full); batch_key_pos.append(pos)
                                batch_gts.append(_dens(p_)); batch_progs.append(progs)
                                done = True
                                break
                            dv = _dens(p_)
                            if dv is not None:
                                meta.append(dv)
                        if not done:
                            continue
                    elif which == "genre":
                        if ln.get("gt_genre") is None:
                            continue
                        gt_genre = int(ln["gt_genre"])
                        meta = []
                        for p_ in progs:
                            meta.append(tok.get(f"<INST_{p_}>"))
                            dv = _dens(p_)
                            if dv is not None:
                                meta.append(dv)
                        pos = len(base) + len(meta)
                        full = np.array(base + meta + [gt_genre, TAG_END], dtype=np.int64)
                        batch_fulls.append(full); batch_key_pos.append(pos)
                        batch_gts.append(gt_genre); batch_progs.append(progs)
                else:
                    if which == "key":
                        if ln["gt_key"] is None:
                            continue
                        gt_key = int(ln["gt_key"])
                        
                        if use_trigger:
                            meta = [tok.get("<KEY>"), gt_key]
                            key_pos = len(prompt) + 1
                            full = np.array(prompt + meta, dtype=np.int64)
                            batch_fulls.append(full)
                            batch_key_pos.append(key_pos)
                            batch_gts.append(gt_key)
                            batch_progs.append(progs)
                        else:
                            gt_dense = ln["gt_dense"]
                            meta = []
                            for p in progs:
                                inst = tok.get(f"<INST_{p}>")
                                meta.append(inst)
                                meta.append(int(gt_dense[p]))
                            key_pos = len(prompt) + len(meta)
                            meta.append(gt_key); meta.append(tok.get("<TAG_END>"))
                            
                            full = np.array(prompt + meta, dtype=np.int64)
                            batch_fulls.append(full)
                            batch_key_pos.append(key_pos)
                            batch_gts.append(gt_key)
                            batch_progs.append(progs)
                    elif which == "dense":
                        gt_dense = ln["gt_dense"]
                        p = progs[0]
                        if use_trigger:
                            meta = [tok.get("<DENCE>"), tok.get(f"<INST_{p}>"), int(gt_dense[p])]
                            dense_pos = {p: len(prompt) + 2}
                            full = np.array(prompt + meta, dtype=np.int64)
                            batch_fulls.append(full)
                            batch_dense_pos.append(dense_pos)
                            batch_gts.append(int(gt_dense[p]))
                            batch_progs.append(progs)
                        else:
                            meta = []
                            dense_pos = {}
                            for p_ in progs:
                                inst = tok.get(f"<INST_{p_}>")
                                meta.append(inst)
                                dense_pos[p_] = len(prompt) + len(meta)
                                meta.append(int(gt_dense[p_]))
                            gt_key = int(ln["gt_key"]) if ln["gt_key"] is not None else tok.get("<TAG_END>")
                            meta.append(gt_key); meta.append(tok.get("<TAG_END>"))
                            
                            full = np.array(prompt + meta, dtype=np.int64)
                            batch_fulls.append(full)
                            batch_dense_pos.append(dense_pos)
                            batch_gts.append(int(gt_dense[p]))
                            batch_progs.append(progs)
                    elif which == "genre":
                        gt_genre = int(ln["gt_genre"])
                        if use_trigger:
                            meta = [tok.get("<GENRE>"), gt_genre]
                            genre_pos = len(prompt) + 1
                            full = np.array(prompt + meta, dtype=np.int64)
                            batch_fulls.append(full)
                            batch_genre_pos.append(genre_pos)
                            batch_gts.append(gt_genre)
                            batch_progs.append(progs)
                        else:
                            genre_pos = len(prompt)
                            full = np.array(prompt + [gt_genre, tok.get("<TAG_END>")], dtype=np.int64)
                            batch_fulls.append(full)
                            batch_genre_pos.append(genre_pos)
                            batch_gts.append(gt_genre)
                            batch_progs.append(progs)
                
            if not batch_fulls:
                continue
                
            max_len = max(len(f) for f in batch_fulls)
            padded_fulls = []
            for f in batch_fulls:
                pad_len = max_len - len(f)
                padded_fulls.append(np.pad(f, (0, pad_len), mode="constant", constant_values=0))
                
            x = torch.tensor(np.stack(padded_fulls), dtype=torch.long, device=dev)
            padding_mask = (x != 0)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward(x, padding_mask=padding_mask, is_causal=True, is_save_cache=False)
            
            logits = logits.float()
            
            for i in range(len(batch_fulls)):
                progs = batch_progs[i]
                if is_sft:
                    pos = batch_key_pos[i]
                else:
                    if which == "key":
                        pos = batch_key_pos[i]
                    elif which == "dense":
                        pos = batch_dense_pos[i][progs[0]]
                    elif which == "genre":
                        pos = batch_genre_pos[i]
                    
                pl = logits[i, pos - 1]
                
                sub = pl[ids_t]
                k = min(5, len(ids_t))
                topk = ids_t[torch.topk(sub, k).indices].tolist()
                pred = int(topk[0])
                gt = batch_gts[i]
                
                preds.append(pred); gts.append(gt)
                n_ok += int(pred == gt); n += 1
                n_top5 += int(gt in topk)
                emit_ok += int(int(pl.argmax()) in set(ids.tolist()))
                
        if n == 0:
            continue
        f1 = macro_f1(gts, preds)
        
        if which == "key":
            key_counts = collections.Counter(gts)
            maj_cnt = key_counts.most_common(1)[0][1] if key_counts else 0
            maj_baseline = maj_cnt / len(gts) if gts else 0.0
            
            n_enh_ok = sum(1 for p, g in zip(preds, gts) 
                           if normalize_key_enharmonics(tok.rev_tokens[p]) == normalize_key_enharmonics(tok.rev_tokens[g]))
            enh_acc = n_enh_ok / len(preds) if preds else 0.0
            
            total_mirex = sum(mirex_score(tok.rev_tokens[g], tok.rev_tokens[p]) for p, g in zip(preds, gts))
            avg_mirex = total_mirex / len(preds) if preds else 0.0
            
            res["key"] = {"acc": n_ok / n, "top5_acc": n_top5 / n, "macro_f1": f1, "emit_rate": emit_ok / n,
                          "enh_acc": enh_acc, "mirex": avg_mirex, "majority_baseline": maj_baseline,
                          "n": n, "n_classes": len(ids), "chance": 1.0 / len(ids)}
        elif which == "dense":
            dense_counts = collections.Counter(gts)
            maj_cnt = dense_counts.most_common(1)[0][1] if dense_counts else 0
            maj_baseline = maj_cnt / len(gts) if gts else 0.0
            
            res["dense"] = {"acc": n_ok / n, "top5_acc": n_top5 / n, "macro_f1": f1, "emit_rate": emit_ok / n,
                            "majority_baseline": maj_baseline,
                            "n": n, "n_classes": len(ids), "chance": 1.0 / len(ids)}
        elif which == "genre":
            genre_counts = collections.Counter(gts)
            maj_cnt = genre_counts.most_common(1)[0][1] if genre_counts else 0
            maj_baseline = maj_cnt / len(gts) if gts else 0.0
            
            res["genre"] = {"acc": n_ok / n, "top5_acc": n_top5 / n, "macro_f1": f1, "emit_rate": emit_ok / n,
                            "majority_baseline": maj_baseline,
                            "n": n, "n_classes": len(ids), "chance": 1.0 / len(ids)}
            
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def macro_f1(gts, preds):
    labels = set(gts)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for g, p in zip(gts, preds):
        if p == g: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    f1s = []
    for c in labels:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, nargs="?", default="80M", help="Model size: 80M or 10M")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a specific .pth file to evaluate")
    parser.add_argument("--name", type=str, default="SFT", help="Name of the model being evaluated")
    parser.add_argument("--cfg", type=str, default=None, help="Path to model config JSON")
    parser.add_argument("--use_trigger", action="store_true", help="Evaluate SFT models using analysis triggers")
    args = parser.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
    tok.mode(TO_MUSIC)
    key_ids, dense_ids, genre_ids = build_class_sets(tok)
    print(f"key={len(key_ids)} (chance {1/len(key_ids):.3f}), density={len(dense_ids)} (chance {1/len(dense_ids):.3f}), genre={len(genre_ids)} (chance {1/len(genre_ids):.3f})")
    
    out = {}
    if args.ckpt is not None:
        cfg_file = args.cfg if args.cfg is not None else CFG[args.size]
        print(f"Evaluating custom checkpoint: {args.ckpt} with config {cfg_file} (use_trigger={args.use_trigger})")
        out[args.name] = eval_model(args.name, args.ckpt, cfg_file, tok, key_ids, dense_ids, genre_ids, dev, use_trigger=args.use_trigger)
        r = out[args.name]
        print(f"[{args.name}] KEY acc={r.get('key',{}).get('acc',0):.3f} | DENSE acc={r.get('dense',{}).get('acc',0):.3f}", flush=True)
    else:
        for name, ckpt in model_set(args.size).items():
            if not ckpt: print(f"{name}: no ckpt"); continue
            out[name] = eval_model(name, ckpt, CFG[args.size], tok, key_ids, dense_ids, genre_ids, dev, use_trigger=args.use_trigger)
            r = out[name]
            print(f"[{name}] KEY acc={r.get('key',{}).get('acc',0):.3f} | DENSE acc={r.get('dense',{}).get('acc',0):.3f}", flush=True)
        os.makedirs(os.path.join(_ROOT, "report", "E1", "data"), exist_ok=True)
        json.dump(out, open(os.path.join(_ROOT, "report", "E1", "data", f"analysis_acc_{args.size}.json"), "w"), indent=2)

    print(f"\n=== SUMMARY ===")
    for name in out:
        r = out[name]
        k = r.get('key', {})
        d = r.get('dense', {})
        print(f"{name:6s} key_acc={k.get('acc', 0):.3f} key_f1={k.get('macro_f1', 0):.3f} dense_acc={d.get('acc', 0):.3f} dense_f1={d.get('macro_f1', 0):.3f}")


if __name__ == "__main__":
    main()
