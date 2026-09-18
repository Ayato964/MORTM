"""密度5-7の濃い分析評価窓を test 曲(GMD, リークフリー)から新規収集。
凍結TEST-TASKは1曲1窓(最初の窓)で低密度に偏る(76%が密度<=3)ため、
全窓を走査し密度5-7の窓だけを集める(1曲最大3窓で偏り防止)。探索的診断用(凍結TEST-TASKとは別)。
出力: /home/.../data/paper/TEST-TASK-DENSE-GEN/analysis_{key,dense}.jsonl
"""
import json, os, sys
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from mortm.eval.protocol_builder import ProtocolBuilder

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "splits")
GMD = "/media/takaaki-nagoshi/MIDIdatasets/GMD/training"
OUT = "/home/takaaki-nagoshi/data/paper/TEST-TASK-DENSE-GEN"
PROGRAMS = ["PIANO", "SAX"]
DENSE_BINS = {5, 6, 7}
MAX_PER_SONG = 3
TARGET = int(sys.argv[sys.argv.index("--target")+1]) if "--target" in sys.argv else 500


def main():
    tk = Tokenizer(get_token_converter_pro(TO_TOKEN))
    pb = ProtocolBuilder(tk, PROGRAMS)
    rev = {v: k for k, v in tk.tokens.items()}
    klo, khi = tk.get_length_tuple("k")
    def dbin(tid):
        n = rev.get(int(tid), "")
        return int(n.replace("<NOTE_DENSE_", "").replace(">", "")) if n.startswith("<NOTE_DENSE_") else None

    test_songs = sorted(open(os.path.join(SPLIT_DIR, "test_songs.txt")).read().split())
    sset = set(test_songs)
    h2p = {}
    for dp, _, fs in os.walk(GMD):
        for f in fs:
            if f.lower().endswith((".mid", ".midi")):
                s = os.path.splitext(f)[0]
                if s in sset:
                    h2p[s] = os.path.join(dp, f)

    os.makedirs(OUT, exist_ok=True)
    fk = open(os.path.join(OUT, "analysis_key.jsonl"), "w")
    fd = open(os.path.join(OUT, "analysis_dense.jsonl"), "w")
    n = 0; processed = 0; hist = {}
    for song in test_songs:
        if n >= TARGET:
            break
        p = h2p.get(song)
        if not p:
            continue
        d, f = os.path.split(p)
        try:
            con = MIDIConverter(tk, d, f, PROGRAMS); con.convert()
            if con.is_error or con.midi2seq is None:
                continue
            maker = FoundationDataMaker(con, 1, 8)
            if maker.is_error:
                continue
        except Exception:
            continue
        processed += 1
        per_song = 0
        try:
            for w in pb.windows(d, f):
                if per_song >= MAX_PER_SONG or n >= TARGET:
                    break
                gt_dense = {prog: int(d_) for prog, d_ in w.info}
                rep = dbin(gt_dense.get(w.programs[0]))
                if rep not in DENSE_BINS:
                    continue
                try:
                    meta = pb._meta_block(maker, w.info)
                    kc = [int(t) for t in meta if klo <= int(t) < khi]
                    key_tok = kc[0] if kc else None
                    pk, _ = pb.build(w, maker, "analysis_key")
                    pd, _ = pb.build(w, maker, "analysis_dense")
                except Exception:
                    continue
                rec_k = {"song": song, "programs": w.programs, "prompt": [int(x) for x in pk],
                         "ref_const": None, "gt_key": key_tok, "gt_dense": gt_dense}
                rec_d = {"song": song, "programs": w.programs, "prompt": [int(x) for x in pd],
                         "ref_const": None, "gt_key": key_tok, "gt_dense": gt_dense}
                fk.write(json.dumps(rec_k) + "\n"); fd.write(json.dumps(rec_d) + "\n")
                hist[rep] = hist.get(rep, 0) + 1
                n += 1; per_song += 1
        except Exception:
            continue
        if processed % 200 == 0:
            print(f"processed {processed} songs, collected {n} (hist={hist})", flush=True)
    fk.close(); fd.close()
    print(f"[DENSE-GEN] done: {n} 窓 (密度分布={hist}) from {processed} test曲 -> {OUT}")


if __name__ == "__main__":
    main()
