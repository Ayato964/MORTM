"""Moonbeam系 外部ベンチマーク データ準備。
各MIDIを MORTMトークナイザで変換し、head互換の音楽ブロック
  <EOS> <CONST_M> <INST_xxx> ... <ESEQ> <TAG_END> <TE>
(=AnalysisDataMaker.convert_analysis_sft の CONST部を <META> 前まで抽出+<TE>) を作り、
外部ラベル(感情/作曲家/奏者)を付与して split ごとにキャッシュする。

対応: emopia(感情4, 公式split) / pianist8(作曲家8, 層化split) / midikong(奏者top8, 層化split)。
出力: bench/cache/{dataset}_{split}.npz  (seqs=object配列, labels=int配列, label_names=str配列)
"""
import os, sys, io, csv, zipfile, random, tempfile, shutil, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker

DATA_DIR = os.path.join(_ROOT, "data")
CACHE = os.path.join(_ROOT, "bench", "cache")
BLOCK_M = 8
PROGRAM = ["PIANO", "SAX"]
SEED = 42


class ContextDataMaker(FoundationDataMaker):
    def convert_greedy_context(self, block_max=8):
        if self.is_error:
            return
        sme = self.tokenizer.get("<SME>")
        s_e = self.tokenizer.get_length_tuple("s")
        eos = self.tokenizer.get("<EOS>"); te = self.tokenizer.get("<TE>")
        valid = [p for p in self.converter.program_list if p in self.seq_dict]
        if not valid:
            return
        seq_inds = {p: np.where(self.seq_dict[p] == sme)[0] for p in valid}
        nmeas = {p: len(seq_inds[p]) - 1 for p in valid}
        max_meas = max(nmeas.values()) if nmeas else 0

        def _slice(p, a, b):
            inds = seq_inds[p]
            seq = self.seq_dict[p][inds[a]:inds[b]]
            return seq if np.any(np.isin(seq, np.arange(s_e[0], s_e[1]))) else None

        now = 0
        while now < max_meas:
            avail_global = max_meas - now
            past_s, const_s, future_s = {}, {}, {}
            active = []
            for p in valid:
                avail = nmeas[p] - now
                if avail <= 0:
                    continue
                pl = min(block_max, avail)
                cl = min(block_max, avail - pl)
                fl = min(block_max, avail - pl - cl)
                if pl > 0:
                    s = _slice(p, now, now + pl)
                    if s is not None:
                        past_s[p] = s
                if cl > 0:
                    s = _slice(p, now + pl, now + pl + cl)
                    if s is not None:
                        const_s[p] = s
                if fl > 0:
                    s = _slice(p, now + pl + cl, now + pl + cl + fl)
                    if s is not None:
                        future_s[p] = s
                active.append(p)
            if not active:
                break
            parts = [np.array([eos], dtype=int)]
            for marker, seqs in (("<PAST_M>", past_s), ("<CONST_M>", const_s), ("<FUTURE_M>", future_s)):
                progs = [p for p in active if p in seqs]
                if progs:
                    parts.append(self._build_melody_block(marker, seqs, progs))
            parts.append(np.array([te], dtype=int))
            self.aya_node.append(np.concatenate(parts))
            now += min(block_max * 3, avail_global)


def midi_to_blocks(tok, ids, midi_dir, midi_file):
    try:
        con = MIDIConverter(tok, midi_dir, midi_file, PROGRAM)
        con.convert()
        if con.is_error:
            return []
        m = ContextDataMaker(con, min_measure=BLOCK_M, max_measure=BLOCK_M, disable_block_augment=True)
        m.convert_greedy_context(block_max=BLOCK_M)
        blocks = []
        for a in m.aya_node:
            sl = np.asarray(a)
            if sl.ndim == 0 or len(sl) <= 4:
                continue
            blocks.append(sl.astype(np.int64))
        return blocks
    except Exception:
        return []


def build(dataset, items, label_names, tok, ids, tag=""):
    os.makedirs(CACHE, exist_ok=True)
    by_split = {}
    n_ok_files = {}; n_err = {}
    for j, (split, d, f, lab) in enumerate(items, 1):
        blocks = midi_to_blocks(tok, ids, d, f)
        by_split.setdefault(split, {"seqs": [], "labels": [], "file_ids": []})
        n_ok_files.setdefault(split, 0); n_err.setdefault(split, 0)
        if not blocks:
            n_err[split] += 1
        else:
            n_ok_files[split] += 1
            for b in blocks:
                by_split[split]["seqs"].append(b)
                by_split[split]["labels"].append(lab)
                by_split[split]["file_ids"].append(j)
        if j % 200 == 0:
            print(f"  [{dataset}] {j}/{len(items)} 変換済", flush=True)
    for split, dd in by_split.items():
        out = os.path.join(CACHE, f"{dataset}{tag}_{split}.npz")
        np.savez(out,
                 seqs=np.array(dd["seqs"], dtype=object),
                 labels=np.array(dd["labels"], dtype=np.int64),
                 file_ids=np.array(dd["file_ids"], dtype=np.int64),
                 label_names=np.array(label_names, dtype=object))
        print(f"  saved {out}: clips={len(dd['labels'])} files_ok={n_ok_files[split]} files_err={n_err[split]}")


def prep_emopia(tok, ids, split_seed=42, tag=""):
    zp = os.path.join(DATA_DIR, "EMOPIA+.zip")
    Q = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    label_names = ["Q1", "Q2", "Q3", "Q4"]
    tmp = tempfile.mkdtemp(prefix="emopia_")
    items = []
    with zipfile.ZipFile(zp) as z:
        split_map = {"train_clip": "train", "val_clip": "val", "test_clip": "test"}
        clip_to_split = {}
        for csvn, split in split_map.items():
            data = z.read(f"EMOPIA+/split/{csvn}.csv").decode()
            for row in csv.DictReader(io.StringIO(data)):
                clip_to_split[row["clip_name"]] = split
        for n in z.namelist():
            if n.startswith("EMOPIA+/midis/") and n.endswith(".mid"):
                base = os.path.basename(n)
                if base in clip_to_split:
                    z.extract(n, tmp)
                    items.append((clip_to_split[base], os.path.join(tmp, "EMOPIA+/midis"),
                                  base, Q[base.split("_")[0]]))
    print(f"[emopia] {len(items)} clips (train/val/test)")
    build("emopia", items, label_names, tok, ids, tag=tag)
    shutil.rmtree(tmp, ignore_errors=True)


def _stratified_split(by_label, seed=SEED, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    out = []
    for lab, files in by_label.items():
        fs = files[:]; rng.shuffle(fs)
        n = len(fs); ntr = int(n * ratios[0]); nva = int(n * ratios[1])
        for i, fp in enumerate(fs):
            sp = "train" if i < ntr else ("val" if i < ntr + nva else "test")
            out.append((sp, fp, lab))
    return out


def prep_pianist8(tok, ids, split_seed=42, tag=""):
    zp = os.path.join(DATA_DIR, "Pianist8-v1.0.0.zip")
    tmp = tempfile.mkdtemp(prefix="pianist8_")
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    root = None
    for dp, dns, fns in os.walk(tmp):
        if os.path.basename(dp) == "midi" and dns:
            root = dp; break
    if not root or not os.path.exists(root):
        print(f"Error: could not find midi/ directory in {zp}")
        shutil.rmtree(tmp, ignore_errors=True)
        return
    composers = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    label_names = composers
    lab_id = {c: i for i, c in enumerate(composers)}
    by_label = {}
    for c in composers:
        cd = os.path.join(root, c)
        fs = [os.path.join(cd, f) for f in os.listdir(cd) if f.lower().endswith((".mid", ".midi"))]
        by_label[lab_id[c]] = fs
    assigned = _stratified_split(by_label, seed=split_seed)
    items = [(sp, os.path.dirname(fp), os.path.basename(fp), lab) for sp, fp, lab in assigned]
    print(f"[pianist8] {len(composers)}クラス {len(items)}曲: {composers}")
    build("pianist8", items, label_names, tok, ids, tag=tag)
    shutil.rmtree(tmp, ignore_errors=True)


def prep_midikong(tok, ids, split_seed=42, tag="", topk=30):
    zp = os.path.join(DATA_DIR, "midi_kong.zip")
    tmp = tempfile.mkdtemp(prefix="midikong_")
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    perf_files = {}
    base = os.path.join(tmp, "midi_kong")
    for cat in ("live", "studio"):
        cd = os.path.join(base, cat)
        if not os.path.isdir(cd):
            continue
        for perf in os.listdir(cd):
            pd = os.path.join(cd, perf)
            if not os.path.isdir(pd):
                continue
            for dp, _, fns in os.walk(pd):
                for f in fns:
                    if f.lower().endswith((".mid", ".midi")):
                        perf_files.setdefault(perf, []).append(os.path.join(dp, f))
    top = sorted(perf_files, key=lambda p: len(perf_files[p]), reverse=True)[:topk]
    label_names = top
    lab_id = {p: i for i, p in enumerate(top)}
    by_label = {lab_id[p]: perf_files[p] for p in top}
    assigned = _stratified_split(by_label, seed=split_seed)
    items = [(sp, os.path.dirname(fp), os.path.basename(fp), lab) for sp, fp, lab in assigned]
    print(f"[midikong] top{topk}奏者 {len(items)}曲: {[(p, len(perf_files[p])) for p in top]}")
    build("midikong", items, label_names, tok, ids, tag=tag)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["emopia", "pianist8", "midikong"])
    ap.add_argument("--split_seed", type=int, default=42, help="層化splitのseed(複数split検証用)")
    a = ap.parse_args()
    tok = Tokenizer(get_token_converter_pro(TO_TOKEN))
    ids = (tok.get("<EOS>"), tok.get("<PAST_M>"), tok.get("<TE>"))
    tag = "" if a.split_seed == 42 else f"_s{a.split_seed}"
    {"emopia": prep_emopia, "pianist8": prep_pianist8, "midikong": prep_midikong}[a.dataset](tok, ids, split_seed=a.split_seed, tag=tag)
    print("done.")
