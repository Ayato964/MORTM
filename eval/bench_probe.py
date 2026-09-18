"""Moonbeam系 外部ベンチマーク: frozen-probe 評価。
SOTA 4.5D-80M を凍結し、PMA分類ヘッド(ClassificationMORTM)のみ学習して test 分類精度を測る。
入力=bench_prep.py が作った音楽ブロック(<EOS>..<TAG_END><TE>)。E3 frozen-probe と同一方式。

指標: clip単位 acc / macro-F1。best は val acc で選択。
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.models.mortm import ClassificationMORTM, MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import LearningProgress
from mortm.models.modules.layers import AttentionPool
from eval_analysis_acc import macro_f1
from flash_attn.bert_padding import pad_input, unpad_input


class AttnPoolClassifier(nn.Module):
    def __init__(self, backbone, args, class_num, dropout=0.3, head="linear"):
        super().__init__()
        self.backbone = backbone
        self.d_model = backbone.d_model
        self.pool = AttentionPool(args)
        d = self.d_model
        if head == "mlp":
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d // 2),
                                            nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, class_num))
        else:
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, class_num))

    def forward(self, x, padding_mask=None, is_causal=True, **kw):
        xe = self.backbone.embedding(x).to(dtype=torch.bfloat16)
        b, s, e = xe.size()
        xu, indices, cu, maxs, used = unpad_input(xe, padding_mask)
        out = self.backbone.decoder(tgt=xu, tgt_is_causal=is_causal, cu_seqlens=cu,
                                    max_seqlen=maxs, batch_size=b, indices=indices, is_save_cache=False)
        pooled = self.pool(out, cu)
        return self.classifier(pooled)


class MeanPoolClassifier(nn.Module):
    def __init__(self, backbone, class_num, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.d_model = backbone.d_model
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_model, class_num))

    def forward(self, x, padding_mask=None, is_causal=True, **kw):
        xe = self.backbone.embedding(x).to(dtype=torch.bfloat16)
        b, s, e = xe.size()
        xu, indices, cu, maxs, used = unpad_input(xe, padding_mask)
        out = self.backbone.decoder(tgt=xu, tgt_is_causal=is_causal, cu_seqlens=cu,
                                    max_seqlen=maxs, batch_size=b, indices=indices, is_save_cache=False)
        out = pad_input(out, indices, b, s)
        m = padding_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.classifier(pooled)


DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANK, WORLD, DDP_ON = 0, 1, False
CACHE = os.path.join(_ROOT, "bench", "cache")


def setup_ddp():
    global DEV, RANK, WORLD, DDP_ON
    lr = int(os.environ.get("LOCAL_RANK", -1))
    if lr < 0:
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(lr)
    DEV = torch.device(f"cuda:{lr}")
    RANK = dist.get_rank(); WORLD = dist.get_world_size(); DDP_ON = True


BACKBONES = {
    "80M": (os.path.join(_ROOT, "configs/models/mortm/foundation/80M.json"),
            os.path.join(_ROOT, "out/models/mortm/4_5/MORTM.4.5D-80M_1.1424479484558105.pth")),
    "160M": (os.path.join(_ROOT, "configs/models/mortm/foundation/160M.json"),
             os.path.join(_ROOT, "out/models/4_5/MORTM.4.5D-160M.pth")),
    "A1": (os.path.join(_ROOT, "configs/models/mortm/foundation/80M.json"),
           os.path.join(_ROOT, "out/models/paper/E1/A1_80M_3.2B/MORTM.E1-A1-80M-3.2B_1.3635098934173584.pth")),
    "A2": (os.path.join(_ROOT, "configs/models/mortm/foundation/80M.json"),
           os.path.join(_ROOT, "out/models/paper/E1/A2_80M_3.2B/MORTM.E1-A2-80M-3.2B_1.3892196416854858.pth")),
}


class _Prog(LearningProgress):
    def get_device(self): return DEV


def load_split(dataset, split, tag=""):
    d = np.load(os.path.join(CACHE, f"{dataset}{tag}_{split}.npz"), allow_pickle=True)
    fids = d["file_ids"].astype(np.int64) if "file_ids" in d.files else np.arange(len(d["labels"]))
    return list(d["seqs"]), d["labels"].astype(np.int64), [str(x) for x in d["label_names"]], fids


def pooled_kfold(dataset, stag, k, fold, seed=42):
    import random as _r
    allseq, ally, allfk = [], [], []
    names = None
    for sp in ("train", "val", "test"):
        try:
            s, y, nm, fk = load_split(dataset, sp, stag)
        except FileNotFoundError:
            continue
        names = nm
        for seq, lab, f in zip(s, y, fk):
            allseq.append(seq); ally.append(int(lab)); allfk.append(f"{sp}:{int(f)}")
    file_lab = {}
    for fk, lab in zip(allfk, ally):
        file_lab[fk] = lab
    by_lab = {}
    for fk, lab in file_lab.items():
        by_lab.setdefault(lab, []).append(fk)
    rng = _r.Random(seed)
    fold_of = {}
    for lab, files in by_lab.items():
        fs = files[:]; rng.shuffle(fs)
        for i, f in enumerate(fs):
            fold_of[f] = i % k
    fkey2id = {fk: i for i, fk in enumerate(sorted(file_lab))}
    tf, vf = fold, (fold + 1) % k
    out = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    for seq, lab, fk in zip(allseq, ally, allfk):
        fo = fold_of[fk]
        b = "test" if fo == tf else ("val" if fo == vf else "train")
        out[b][0].append(seq); out[b][1].append(lab); out[b][2].append(fkey2id[fk])
    def y(b): return np.array(out[b][1], dtype=np.int64)
    def f(b): return np.array(out[b][2], dtype=np.int64)
    return (out["train"][0], y("train"),
            out["val"][0], y("val"), f("val"),
            out["test"][0], y("test"), f("test"), names)


@torch.no_grad()
def perclass_file_report(model, seqs, labels, file_ids, names, bs=64, max_len=0):
    model.eval()
    preds = []
    for xb, mask, yb in batches(seqs, labels, bs, shuffle=False, max_len=max_len):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds.extend(model(x=xb, padding_mask=mask, is_causal=True).float().argmax(-1).cpu().tolist())
    from collections import Counter, defaultdict
    fp = defaultdict(list); fg = {}
    for p, g, fid in zip(preds, labels, file_ids):
        fp[int(fid)].append(p); fg[int(fid)] = int(g)
    per = defaultdict(lambda: [0, 0])
    for fid, ps in fp.items():
        pred = Counter(ps).most_common(1)[0][0]; gt = fg[fid]
        per[gt][1] += 1; per[gt][0] += int(pred == gt)
    return {names[c]: (v[0], v[1]) for c, v in sorted(per.items())}


def batches(seqs, labels, bs, shuffle, max_len=0):
    idx = np.arange(len(seqs))
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, len(idx), bs):
        bi = idx[s:s + bs]
        xs = [torch.tensor(seqs[i][:max_len] if max_len else seqs[i], dtype=torch.long) for i in bi]
        maxlen = max(len(x) for x in xs)
        xb = torch.stack([F.pad(x, (0, maxlen - len(x)), value=0) for x in xs]).to(DEV)
        yb = torch.tensor([labels[i] for i in bi], dtype=torch.long).to(DEV)
        yield xb, (xb != 0).to(DEV), yb


@torch.no_grad()
def evaluate(model, seqs, labels, bs=64, file_ids=None, max_len=0):
    model.eval()
    preds = []
    for xb, mask, yb in batches(seqs, labels, bs, shuffle=False, max_len=max_len):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x=xb, padding_mask=mask, is_causal=True).float()
        preds.extend(logits.argmax(-1).cpu().tolist())
    gts = list(labels)
    clip_acc = sum(int(p == g) for p, g in zip(preds, gts)) / len(gts) if gts else 0.0
    clip_f1 = macro_f1(gts, preds) if gts else 0.0
    if file_ids is None:
        return clip_acc, clip_f1, None, None
    from collections import Counter, defaultdict
    fp = defaultdict(list); fg = {}
    for p, g, fid in zip(preds, gts, file_ids):
        fp[int(fid)].append(p); fg[int(fid)] = g
    f_pred = {fid: Counter(ps).most_common(1)[0][0] for fid, ps in fp.items()}
    fids = sorted(fg)
    fgts = [fg[i] for i in fids]; fprs = [f_pred[i] for i in fids]
    file_acc = sum(int(p == g) for p, g in zip(fprs, fgts)) / len(fgts) if fgts else 0.0
    return clip_acc, clip_f1, file_acc, macro_f1(fgts, fprs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emopia", "pianist8", "midikong"])
    ap.add_argument("--backbone", default="80M", choices=["80M", "160M", "A1", "A2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--pma_mult", type=float, default=1.0)
    ap.add_argument("--mode", default="frozen", choices=["frozen", "lora"])
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--head", default="mlp", choices=["mlp","linear"])
    ap.add_argument("--pool", default="pma", choices=["pma","mean","attn"])
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--kfold", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--dump_preds", default="")
    a = ap.parse_args()

    setup_ddp()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    stag = "" if a.split_seed == 42 else f"_s{a.split_seed}"
    if a.kfold > 0:
        (tr_s, tr_y, va_s, va_y, va_f, te_s, te_y, te_f, names) = pooled_kfold(a.dataset, stag, a.kfold, a.fold, a.split_seed)
        if RANK == 0:
            print(f"  [k-fold] k={a.kfold} fold={a.fold} (test=fold{a.fold})")
    else:
        tr_s, tr_y, names, _ = load_split(a.dataset, "train", stag)
        va_s, va_y, _, va_f = load_split(a.dataset, "val", stag)
        te_s, te_y, _, te_f = load_split(a.dataset, "test", stag)
    nc = len(names)
    print(f"[{a.dataset}] classes={nc} {names}")
    print(f"  clips train={len(tr_s)} val={len(va_s)} test={len(te_s)}")
    print(f"  chance={1/nc:.3f}  majority(test)={np.bincount(te_y, minlength=nc).max()/len(te_y):.3f}")

    cfg, ckpt_path = BACKBONES[a.backbone]
    print(f"  backbone={a.backbone} cfg={cfg}")
    args = MORTMArgs(cfg)
    if a.mode == "lora":
        args.use_attn_lora = True; args.use_ffn_lora = True; args.use_gate_lora = False
        args.lora_r = a.lora_r
    sd = torch.load(ckpt_path, map_location=DEV)
    d = args.d_model
    if a.pool in ("mean", "attn"):
        bb = MORTM(args, _Prog()).to(DEV)
        mi, un = bb.load_state_dict(sd, strict=False)
        model = (MeanPoolClassifier(bb, nc, a.dropout) if a.pool == "mean"
                 else AttnPoolClassifier(bb, args, nc, a.dropout, head=a.head)).to(DEV)
        pod = d
        print(f"  pool={a.pool} 単層Linear head, backbone missing={len(mi)} unexpected={len(un)}")
    else:
        pod = int(d * a.pma_mult)
        model = ClassificationMORTM(args, nc, _Prog(), pma_out_dim=pod).to(DEV)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        nhm = [k for k in missing if not (k.startswith("pma") or k.startswith("classifier"))]
        print(f"  backbone loaded: missing_head={len(missing)-len(nhm)} missing_other={len(nhm)} unexpected={len(unexpected)}")
        if a.head == "linear":
            model.classifier = nn.Sequential(nn.Dropout(a.dropout), nn.Linear(pod, nc)).to(DEV)
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(a.dropout),
                nn.Linear(pod, d // 2), nn.ReLU(), nn.Dropout(a.dropout),
                nn.Linear(d // 2, nc),
            ).to(DEV)
    print(f"  pool={a.pool} head={a.head} pod={pod} dropout={a.dropout} weight_decay={a.weight_decay}")

    def _is_head(name):
        return name.startswith("pma") or name.startswith("classifier")
    def _is_lora_ft(name):
        return ("lora_" in name or "embedding" in name.lower()
                or name.endswith("Wout.weight") or ".Wout." in name)
    n_train = 0
    for name, p in model.named_parameters():
        tr = _is_head(name) or (a.mode == "lora" and _is_lora_ft(name))
        p.requires_grad = tr
        if tr:
            n_train += p.numel()
    print(f"  mode={a.mode} 学習可能={n_train/1e6:.3f}M")

    net = model
    if DDP_ON:
        net = DDP(model, device_ids=[DEV.index], find_unused_parameters=True)
    eff = a.batch_size * a.accum * WORLD
    if RANK == 0:
        print(f"  DDP={DDP_ON} world={WORLD} micro={a.batch_size} accum={a.accum} => 実効batch={eff}, max_len={a.max_len}")

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    crit = nn.CrossEntropyLoss()

    per = len(tr_s) // WORLD
    best_va = -1; best = None
    curve = []
    for ep in range(a.epochs):
        net.train()
        g = np.random.RandomState(a.seed + ep)
        perm = g.permutation(len(tr_s))
        if WORLD > 1:
            perm = perm[:per * WORLD].reshape(WORLD, per)[RANK]
        sh_s = [tr_s[i] for i in perm]; sh_y = tr_y[perm]
        tot = 0.0; nb = 0; tr_ok = 0; tr_n = 0
        opt.zero_grad()
        for i, (xb, mask, yb) in enumerate(batches(sh_s, sh_y, a.batch_size, shuffle=False, max_len=a.max_len)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = net(x=xb, padding_mask=mask, is_causal=True)
                loss = crit(logits.float(), yb) / a.accum
            loss.backward()
            if (i + 1) % a.accum == 0:
                opt.step(); opt.zero_grad()
            tot += loss.item() * a.accum; nb += 1
            tr_ok += (logits.argmax(-1) == yb).sum().item(); tr_n += yb.numel()
        train_acc = tr_ok / max(tr_n, 1)
        if RANK == 0:
            core = net.module if DDP_ON else net
            va_acc, va_f1, va_facc, _ = evaluate(core, va_s, va_y, file_ids=va_f, max_len=a.max_len)
            if a.curve:
                tc, _, tf, _ = evaluate(core, te_s, te_y, file_ids=te_f, max_len=a.max_len)
            else:
                tc = tf = None
            curve.append((ep + 1, tot / max(nb, 1), train_acc, va_acc, va_facc, tc, tf))
            print(f"  ep{ep+1}/{a.epochs} tr_loss={tot/max(nb,1):.4f} tr_acc={train_acc:.3f} "
                  f"val_clip={va_acc:.3f} val_file={va_facc:.3f}"
                  + (f" test_clip={tc:.3f} test_file={tf:.3f}" if a.curve else ""), flush=True)
            if va_facc > best_va:
                best_va = va_facc
                best = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        if DDP_ON:
            dist.barrier()

    if RANK == 0:
        core = net.module if DDP_ON else net
        if best is not None:
            core.load_state_dict(best)
        if a.dump_preds:
            core.eval()
            dp = []
            with torch.no_grad():
                for xb, mask, yb in batches(te_s, te_y, 64, shuffle=False, max_len=a.max_len):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pr = core(x=xb, padding_mask=mask, is_causal=True).float().argmax(-1).cpu().tolist()
                    dp.extend(pr)
            import json as _dj
            rec = [{"file_id": int(te_f[i]), "gt": int(te_y[i]), "pred": int(dp[i])} for i in range(len(dp))]
            _dj.dump(rec, open(a.dump_preds, "w"))
            print(f"saved {a.dump_preds}: {len(rec)} clip preds (te_s order)")
        te_acc, te_f1, te_facc, te_ff1 = evaluate(core, te_s, te_y, file_ids=te_f, max_len=a.max_len)
        print(f"\n[RESULT] {a.dataset}/{a.backbone}: "
              f"clip acc={te_acc:.4f}/f1={te_f1:.4f} | file acc={te_facc:.4f}/f1={te_ff1:.4f} "
              f"(best_val_file_acc={best_va:.3f}, classes={nc})")

        import json
        out_dir = os.path.join(_ROOT, "report", "E4")
        os.makedirs(out_dir, exist_ok=True)
        hsuf = "" if (a.pool == "pma" and a.head == "mlp") else f"_{a.pool}{a.head}"
        out = os.path.join(out_dir, f"e4_bench{'_lora' if a.mode == 'lora' else ''}{hsuf}.json")
        allr = json.load(open(out)) if os.path.exists(out) else {}
        allr.setdefault(a.backbone, {})
        allr[a.backbone][a.dataset] = {
            "clip_acc": te_acc, "clip_macro_f1": te_f1,
            "file_acc": te_facc, "file_macro_f1": te_ff1,
            "best_val_file_acc": best_va, "n_classes": nc, "classes": names,
            "n_test_clips": len(te_s), "backbone": f"SOTA-4.5D-{a.backbone}",
            "mode": a.mode, "lora_r": (a.lora_r if a.mode == "lora" else None),
            "pma_out_dim": pod, "dropout": a.dropout, "weight_decay": a.weight_decay,
            "lr": a.lr, "epochs": a.epochs, "eff_batch": eff,
            "batch_size": a.batch_size, "accum": a.accum, "world": WORLD, "max_len": a.max_len, "split_seed": a.split_seed, "head": a.head, "pool": a.pool
        }
        pc = perclass_file_report(core, te_s, te_y, te_f, names, max_len=a.max_len)
        print("  per-class(file): " + " ".join(f"{k}={c}/{t}" for k,(c,t) in pc.items()))
        json.dump(allr, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"saved {out}")
    if DDP_ON:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
