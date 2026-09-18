"""E2 補完サンプル(改訂): 密な窓 + top-pサンプリング(設計§4.3: p=0.9,temp=1.0)で生成。
PAST+生成CONST+FUTURE を1旋律に繋いでMIDI化(継ぎ目を聴く用)。
各モデル(A2起点/E2最終/A1零SFT)+GT。単一楽器(PIANO)・音符が十分多い窓のみ。

出力: report/E2/samples/win{i}__{model}.mid
使い方: python eval/eval_e2_samples.py [n_windows]
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
from mortm.utils.de_convert import ct_token_to_midi
from mortm.eval.metrics import parse_notes
from mortm.eval.kv_generate import kv_generate, kv_generate_batch

CFG = os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json")
DATA_DIR = os.environ.get("MORTM_TESTTASK", os.path.expanduser("~/data/paper/TEST-TASK"))
INFILL = os.path.join(DATA_DIR, "infill.jsonl")
OUT = os.path.join(_ROOT, "report", "E2", "samples")

DENS_MIN, DENS_MAX, DENS_TARGET = 3.0, 10.0, 6.0


def find1(pat):
    fs = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


MODELS = {
    "A2_zero_start": find1("out/models/paper/E1/A2_80M_3.2B/*_[1-9]*.pth"),
    "E2_final_1600M": find1("out/models/paper/E2/noaug_lr1.0/*_[1-9]*.pth"),
    "A1_zeroSFT": find1("out/models/paper/E1/A1_80M_3.2B/*_[1-9]*.pth"),
}


def melody_only(tokens, tok):
    s_lo, s_hi = tok.get_length_tuple("s")
    p_lo, p_hi = tok.get_length_tuple("p")
    d_lo, d_hi = tok.get_length_tuple("d")
    sme = tok.get("<SME>")
    out = []
    for t in tokens:
        t = int(t)
        if t == sme or (s_lo <= t < s_hi) or (p_lo <= t < p_hi) or (d_lo <= t < d_hi):
            out.append(t)
    return out


def block_from_prompt(prompt, start_marker, tok):
    sm = tok.get(start_marker)
    tag = tok.get("<TAG_END>")
    try:
        i = prompt.index(sm)
    except ValueError:
        return []
    j = i + 1
    seg = []
    while j < len(prompt) and prompt[j] != tag:
        seg.append(prompt[j])
        j += 1
    return melody_only(seg, tok)


def notes_of(seq, tok):
    return parse_notes(seq, tok)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(OUT, exist_ok=True)
    vocab_path = os.path.join(_ROOT, "out", "vocab_list.json")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC), load_data=vocab_path if os.path.exists(vocab_path) else None)
    EOS = tok.get("<EOS>")
    INST = tok.get("<INST_PIANO>")
    TE = tok.get("<TE>")
    if not os.path.exists(INFILL):
        print(f"[warning] {INFILL} does not exist.")
        return
    all_lines = [json.loads(l) for l in open(INFILL)]

    def dens(notes):
        if not notes:
            return 0.0
        nm = max(mm for mm, *_ in notes) + 1
        return len(notes) / max(1, nm)

    cand = []
    for ln in all_lines:
        if ln["programs"] != ["PIANO"]:
            continue
        past = notes_of(block_from_prompt(ln["prompt"], "<PAST_M>", tok), tok)
        fut = notes_of(block_from_prompt(ln["prompt"], "<FUTURE_M>", tok), tok)
        gtc = (
            notes_of(melody_only(np.concatenate([np.asarray(v) for v in ln["ref_const"].values()]), tok), tok)
            if ln.get("ref_const")
            else []
        )
        dp, dc, df = dens(past), dens(gtc), dens(fut)
        if all(DENS_MIN <= d <= DENS_MAX for d in (dp, dc, df)):
            score = abs((dp + dc + df) / 3 - DENS_TARGET)
            cand.append((score, ln, dp, dc, df))
    cand.sort(key=lambda z: z[0])
    lines = [c[1] for c in cand[:n]]
    print(f"中庸密度窓 選抜: {len(lines)}/{len(cand)}候補 (密度帯{DENS_MIN}-{DENS_MAX}音符/小節)", flush=True)

    gens = {}
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
        gens[mname] = {}
        batch_prompts = []
        batch_max_measures = []
        for ln in lines:
            gtc = notes_of(melody_only(np.concatenate([np.asarray(v) for v in ln["ref_const"].values()]), tok), tok)
            ref_meas = (max((mm for mm, *_ in gtc), default=-1) + 1) if gtc else 4
            batch_prompts.append(ln["prompt"])
            batch_max_measures.append(max(1, ref_meas))

        batch_size = 16
        batch_gens = []
        for start_idx in range(0, len(batch_prompts), batch_size):
            end_idx = min(start_idx + batch_size, len(batch_prompts))
            sub_gens = kv_generate_batch(
                m, batch_prompts[start_idx:end_idx], tok, dev,
                batch_max_measures[start_idx:end_idx], seed=1234
            )
            batch_gens.extend(sub_gens)

        for wi, g in enumerate(batch_gens):
            gens[mname][wi] = melody_only(g, tok)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{mname}] generated {len(lines)} windows (top-p 0.9)", flush=True)

    tok.mode(to=TO_MUSIC)
    written = []
    for wi, ln in enumerate(lines):
        past = block_from_prompt(ln["prompt"], "<PAST_M>", tok)
        future = block_from_prompt(ln["prompt"], "<FUTURE_M>", tok)
        gt_const = melody_only(np.concatenate([np.asarray(v) for v in ln["ref_const"].values()]), tok)
        variants = {"GT": gt_const, **{mn: gens[mn][wi] for mn in gens if wi in gens[mn]}}
        for vname, const in variants.items():
            seq = [EOS, INST] + past + const + future + [TE]
            path = os.path.join(OUT, f"win{wi}__{vname}.mid")
            try:
                ct_token_to_midi(tok, torch.tensor(seq), path)
                written.append((wi, vname, len(notes_of(const, tok))))
            except Exception as e:
                print(f"  [{vname} win{wi}] decode err: {e}")
    print(f"\nwrote {len(written)} MIDI files. 生成CONST音符数:")
    for wi in range(len(lines)):
        row = " ".join(f"{vn}={ln}" for (w, vn, ln) in written if w == wi)
        print(f"  win{wi}: {row}")


if __name__ == "__main__":
    main()
