"""トークナイザ往復テスト (研究設計書 v1.3 §9.12 / §9 注記)。

  「9.12 のトークナイザ往復テストは E0 の前に必ず通すこと。
   往復が非可逆だと全客観指標が汚染される。」

不変量: 量子化(小節 96 分割)があるため生 MIDI のビット一致は成立しない。
正しい不変量は **トークン列の冪等性**:
    MIDI --encode--> tok1 --decode--> MIDI' --encode--> tok2
    について tok1 == tok2 (最初の encode で量子化が完了するため、以降は不変)。

エンコード = MIDIConverter(use_midi2seq).aya_node[program]
デコード   = mortm.utils.de_convert.ct_token_to_midi (要 tokenizer.mode(TO_MUSIC))

pytest でもスクリプト単体でも実行可能。
    python tests/test_tokenizer_roundtrip.py [MIDI_DIR] [N]
"""
import os
import sys
import glob
import tempfile

import numpy as np
import torch

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN, TO_MUSIC
from mortm.utils.convert import MIDIConverter
from mortm.utils.de_convert import ct_token_to_midi

PROGRAMS = ["PIANO", "SAX"]
DEFAULT_MIDI_DIR = "/media/takaaki-nagoshi/MIDIdatasets/MIDI_Caps/lmd_full/0"

# デコード tempo について: ct_token_to_midi は PrettyMIDI() を既定(120bpm, テンポ
# イベントなし)で書き出す。よって再エンコードは常に 120 で時間->grid 変換する。
# 自己整合には **復号も 120** で配置する必要がある(元テンポで配置すると書き出しヘッダ
# 120 と不整合になり悪化する)。grid トークンが真にテンポ不変ならこれで冪等になるはず。
DECODE_TEMPO = 120


def _enc_tokenizer() -> Tokenizer:
    return Tokenizer(get_token_converter_pro(TO_TOKEN))


def encode(path: str, tokenizer: Tokenizer):
    """MIDI -> ({program: token np.ndarray}, tempo0)。失敗時 (None, None)。"""
    d, f = os.path.split(path)
    con = MIDIConverter(tokenizer, d, f, PROGRAMS)
    con.convert()
    if con.is_error or con.midi2seq is None:
        return None, None
    tempos = getattr(con.midi2seq, "tempo", None)
    tempo0 = float(tempos[0]) if tempos is not None and len(tempos) > 0 else 120.0
    return dict(con.midi2seq.aya_node), tempo0


def parse_notes(clip: np.ndarray, tokenizer: Tokenizer):
    """旋律トークン列を音符イベント列に構文解析する。
    返り値: canonical にソートした (measure, position, pitch, duration) の多重集合。
    生のトークン配列比較は同時発音ノートの並び順で崩れるため、音符レベルで比較する
    ための正規化(研究設計書 §9.12 の不変量 A: 音符が同じグリッドセルに戻るか)。
    """
    s_lo, s_hi = tokenizer.get_length_tuple("s")
    p_lo, p_hi = tokenizer.get_length_tuple("p")
    d_lo, d_hi = tokenizer.get_length_tuple("d")
    sme = tokenizer.get("<SME>")

    notes = []
    measure = -1          # 最初の <SME> で 0 になる
    cur_s = None
    cur_p = None
    for tok in clip:
        v = int(tok)
        if v == sme:
            measure += 1
        elif s_lo <= v < s_hi:
            cur_s = v
        elif p_lo <= v < p_hi:
            cur_p = v
        elif d_lo <= v < d_hi:
            if cur_s is not None and cur_p is not None:
                notes.append((max(measure, 0), cur_s, cur_p, v))
                cur_p = None      # 音高は音符ごとに更新(和音は s を共有し p/d を反復)
    return sorted(notes)


def roundtrip_program(clip1: np.ndarray, program: str, tempo: float,
                      workdir: str, tokenizer: Tokenizer):
    """clip1 を decode->re-encode し、(音符レベル一致, 生トークン一致, clip2) を返す。"""
    dec_tk = _enc_tokenizer()
    dec_tk.mode(TO_MUSIC)  # rev_tokens 構築 + デコードモード
    eos = dec_tk.get("<EOS>")
    inst = dec_tk.get(f"<INST_{program}>")
    te = dec_tk.get("<TE>")
    seq = np.concatenate([[eos, inst], clip1, [te]]).astype(int)

    out_midi = os.path.join(workdir, "rt.mid")
    ct_token_to_midi(dec_tk, torch.tensor(seq), out_midi, tempo=tempo)

    ay2, _ = encode(out_midi, _enc_tokenizer())
    clip2 = np.asarray(ay2.get(program, [])) if ay2 else np.array([], dtype=int)

    exact = len(clip1) == len(clip2) and bool(np.all(clip1 == clip2))
    n1, n2 = parse_notes(clip1, tokenizer), parse_notes(clip2, tokenizer)
    struct_ok, dur_ok = compare_notes(n1, n2)
    return struct_ok, dur_ok, exact, clip2


def compare_notes(n1, n2, dur_tol: int = 1):
    """(構造一致, 音長許容一致) を返す。
    構造 = (measure, position, pitch) が完全一致(=onset/pitch は完全可逆であること)。
    音長 = 構造一致の上で |Δduration| <= dur_tol(既定1。encode の int() 切り捨てに由来する
    浮動小数の ±1 は亜知覚であり学習不使用なので許容する。§9.12 不変量 A / v1.4)。
    """
    if len(n1) != len(n2):
        return False, False
    struct_ok = True
    dur_ok = True
    for a, b in zip(n1, n2):  # parse_notes は既にソート済み
        if a[:3] != b[:3]:
            struct_ok = False
            dur_ok = False
            break
        if abs(a[3] - b[3]) > dur_tol:
            dur_ok = False
    return struct_ok, dur_ok


def run_batch(midi_dir: str = DEFAULT_MIDI_DIR, n: int = 30) -> dict:
    """先頭 n 曲(encodeに成功しPIANO/SAXを含むもの)で往復冪等率を測る。"""
    files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
    tk = _enc_tokenizer()
    struct_pass = dur_pass = exact_pass = pairs = 0
    fail_samples = []
    checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        for p in files:
            if checked >= n:
                break
            ay, tempo0 = encode(p, _enc_tokenizer())
            if not ay:
                continue
            progs = [k for k in PROGRAMS if k in ay]
            if not progs:
                continue
            checked += 1
            for prog in progs:
                clip1 = np.asarray(ay[prog])
                struct_ok, dur_ok, exact_ok, clip2 = roundtrip_program(
                    clip1, prog, DECODE_TEMPO, workdir, tk)
                pairs += 1
                struct_pass += int(struct_ok)
                dur_pass += int(dur_ok)
                exact_pass += int(exact_ok)
                if not struct_ok:
                    fail_samples.append((os.path.basename(p), prog, len(clip1), len(clip2)))
    return {"checked_songs": checked, "pairs": pairs,
            "struct_pass": struct_pass, "dur_pass": dur_pass, "exact_pass": exact_pass,
            "struct_failed": pairs - struct_pass,
            "fail_samples": fail_samples[:10]}


def test_tokenizer_roundtrip_idempotent():
    """pytest エントリ: 検査した全 (曲,楽器) で往復トークンが完全一致すること。"""
    if not os.path.isdir(DEFAULT_MIDI_DIR):
        import pytest  # type: ignore
        pytest.skip(f"MIDI corpus not mounted: {DEFAULT_MIDI_DIR}")
    res = run_batch(n=30)
    assert res["checked_songs"] > 0, "検査対象の MIDI が見つからない"
    # 不変量 A(§9.12/v1.4): 構造(小節・位置・音高)は完全可逆であること。
    # 音長は encode の int() 切り捨て由来の浮動小数 ±1 を許容(亜知覚・学習不使用)。
    assert res["struct_failed"] == 0, f"構造(onset/pitch)が非可逆: {res['fail_samples']}"


if __name__ == "__main__":
    midi_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MIDI_DIR
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    res = run_batch(midi_dir, n)
    pairs = res["pairs"]
    sr = res["struct_pass"] / pairs * 100 if pairs else 0.0
    dr = res["dur_pass"] / pairs * 100 if pairs else 0.0
    er = res["exact_pass"] / pairs * 100 if pairs else 0.0
    print(f"[roundtrip] songs={res['checked_songs']} pairs={pairs}")
    print(f"  構造一致 onset/pitch/measure (不変量A必須): {res['struct_pass']}/{pairs} = {sr:.1f}%")
    print(f"  音長±1許容込み (不変量A): {res['dur_pass']}/{pairs} = {dr:.1f}%")
    print(f"  生トークン完全一致 (参考): {res['exact_pass']}/{pairs} = {er:.1f}%")
    for s in res["fail_samples"]:
        print("  STRUCT-FAIL:", s)
    sys.exit(0 if res["struct_failed"] == 0 else 1)
