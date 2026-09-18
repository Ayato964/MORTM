"""E3 再測定用: MIDICaps 由来・密度5-7 の高密度分析評価セット (key/dense/genre 3属性)。

凍結 TEST-TASK(GMD由来・低密度)や TEST-TASK-DENSE-GEN(GMD・genre無し)と異なり、
SFT の学習分布(MIDICaps ピアノ&サックス, genre あり)に一致した密度5-7窓を収集する。
リークフリー: docs/splits/test_songs.txt(SFT train から除外済み)からのみ収集。
key GT = SFT が学習した算法ラベル(make_system_prompt の music21 合議キー) = _meta_block 由来。
genre GT = MIDICaps json の第1ジャンル(genre_names_to_ids)。プロンプトは analysis_key と同一
          (音楽 -> <SYSTEM> 打ち切り; eval側が <META><SYSTEM> を再構成して full meta を生成)。

出力: /home/.../data/paper/TEST-TASK-MIDICAPS-DENSE/analysis_{key,dense,genre}.jsonl
"""
import json, os, sys
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker, genre_names_to_ids
from mortm.eval.protocol_builder import ProtocolBuilder

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "splits")
MIDICAPS_ROOT = "/media/takaaki-nagoshi/MIDIdatasets/MIDI_Caps"
MIDICAPS_JSON = MIDICAPS_ROOT + "/json/train.json"
PROGRAMS = ["PIANO", "SAX"]
DENSE_BINS = {5, 6, 7}
MAX_PER_SONG = 8
TARGET = int(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else 500
# 文脈長(分析対象CONSTの小節数)。既定4。8小節版で文脈長レバーを検証(SFTはmax_m=8で学習=分布内)。
CONST_M = int(sys.argv[sys.argv.index("--const_m") + 1]) if "--const_m" in sys.argv else 4
OUT = "/home/takaaki-nagoshi/data/paper/TEST-TASK-MIDICAPS-DENSE" + (f"-{CONST_M}BAR" if CONST_M != 4 else "")


def main():
    tk = Tokenizer(get_token_converter_pro(TO_TOKEN))
    pb = ProtocolBuilder(tk, PROGRAMS)
    rev = {v: k for k, v in tk.tokens.items()}
    klo, khi = tk.get_length_tuple("k")

    def dbin(tid):
        n = rev.get(int(tid), "")
        return int(n.replace("<NOTE_DENSE_", "").replace(">", "")) if n.startswith("<NOTE_DENSE_") else None

    # リークフリー評価プール = test ∪ val (どちらも SFT train から除外済み)。
    # test(640)だけでは密度5-7窓が~200で不足するため val も併用(それでも train非重複)。
    pool = set(open(os.path.join(SPLIT_DIR, "test_songs.txt")).read().split())
    pool |= set(open(os.path.join(SPLIT_DIR, "val_songs.txt")).read().split())
    test_songs = sorted(pool)
    sset = set(test_songs)

    # MIDICaps json: hash(basename) -> (location相対path, [genres])  ※test曲のみ
    hash2loc, hash2genre = {}, {}
    for line in open(MIDICAPS_JSON):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        loc = d.get("location"); g = d.get("genre", [])
        if not loc:
            continue
        h = os.path.splitext(os.path.basename(loc))[0]
        if h in sset:
            hash2loc[h] = loc; hash2genre[h] = g
    print(f"test曲∩MIDICaps(genre付き探索対象): {len(hash2loc)}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    fk = open(os.path.join(OUT, "analysis_key.jsonl"), "w")
    fd = open(os.path.join(OUT, "analysis_dense.jsonl"), "w")
    fg = open(os.path.join(OUT, "analysis_genre.jsonl"), "w")
    n = 0; processed = 0; hist = {}
    for song in test_songs:
        if n >= TARGET:
            break
        loc = hash2loc.get(song)
        if not loc:
            continue
        genres = list(hash2genre.get(song, []))
        genre_tokens = genre_names_to_ids(genres[:1], tk) if genres else []
        genre_tok = int(genre_tokens[0]) if genre_tokens else None
        if genre_tok is None:
            continue  # genre 無し曲は3属性が揃わないため除外
        p = os.path.join(MIDICAPS_ROOT, loc)
        if not os.path.exists(p):
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
            for w in pb.windows(d, f, const_m=CONST_M):
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
                    pk, _ = pb.build(w, maker, "analysis_key")   # = analysis_dense/genre と同一プロンプト
                except Exception:
                    continue
                prm = [int(x) for x in pk]
                fk.write(json.dumps({"song": song, "programs": w.programs, "prompt": prm,
                                     "ref_const": None, "gt_key": key_tok, "gt_dense": gt_dense}) + "\n")
                fd.write(json.dumps({"song": song, "programs": w.programs, "prompt": prm,
                                     "ref_const": None, "gt_key": key_tok, "gt_dense": gt_dense}) + "\n")
                fg.write(json.dumps({"song": song, "programs": w.programs, "prompt": prm,
                                     "ref_const": None, "gt_genre": genre_tok,
                                     "gt_key": None, "gt_dense": None}) + "\n")
                hist[rep] = hist.get(rep, 0) + 1
                n += 1; per_song += 1
        except Exception:
            continue
        if processed % 200 == 0:
            print(f"processed {processed} songs, collected {n} (hist={hist})", flush=True)
    fk.close(); fd.close(); fg.close()
    print(f"[MIDICAPS-DENSE] done: {n} 窓 (密度分布={hist}) from {processed} test曲 -> {OUT}")


if __name__ == "__main__":
    main()
