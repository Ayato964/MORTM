"""A3 augment_mode の正しさを実MIDI数曲で検証する。
各modeの permutation_patterns(SYSTEM位置/ブロック順) と deletion_count(k分布) を出力し、
  perm_only : k=0のみ / SYSTEM位置は多様 / CONST位置多様
  del_only  : k∈{0,1,2} / パターンは常に正順(SYSTEM,PAST,CONST,FUTURE)先頭SYSTEM
  meta_first: k∈{0,1,2}+reorder / SYSTEM常に先頭
  full      : 全て多様
を確認する。
"""
import os, re, random, collections
import numpy as np
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker

DATASETS = "/media/takaaki-nagoshi/MIDIdatasets/GMD/training"
PROGRAM = ['PIANO', 'SAX']
N_SONGS = 12

def find_some_midis(root, n):
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                out.append((dp, f))
                if len(out) >= n:
                    return out
    return out

if __name__ == "__main__":
    tok = Tokenizer(get_token_converter_pro(TO_TOKEN))
    midis = find_some_midis(DATASETS, N_SONGS)
    print(f"testing {len(midis)} MIDIs")
    for mode in ("full", "perm_only", "del_only", "meta_first"):
        random.seed(42)
        patt = collections.Counter()
        delk = collections.Counter()
        n_samples = 0
        for d, f in midis:
            con = MIDIConverter(tok, d, f, PROGRAM)
            con.convert()
            if con.is_error:
                continue
            m = FoundationDataMaker(con, 1, 8, disable_block_augment=False, augment_mode=mode)
            m.convert()
            for c in con.expansion_midi():
                m2 = FoundationDataMaker(c, 1, 8, disable_block_augment=False, augment_mode=mode)
                m2.convert()
                for kk, vv in m2.stats["permutation_patterns"].items():
                    patt[kk] += vv
                for kk, vv in m2.stats["deletion_count"].items():
                    delk[kk] += vv
                n_samples += m2.stats["total_samples"]
            for kk, vv in m.stats["permutation_patterns"].items():
                patt[kk] += vv
            for kk, vv in m.stats["deletion_count"].items():
                delk[kk] += vv
            n_samples += m.stats["total_samples"]
        print(f"\n=== mode={mode}  samples={n_samples} ===")
        print(f"  deletion_count(k): {dict(delk)}")
        print(f"  SYSTEM-first ratio: " + (
            f"{sum(v for kk,v in patt.items() if kk.startswith('SYSTEM'))}/{sum(patt.values())}"))
        top = sorted(patt.items(), key=lambda x: -x[1])[:8]
        for kk, vv in top:
            print(f"    {vv:5d}  {kk}")
