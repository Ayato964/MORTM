"""トークン列から直接キーを推定する(MIDIファイルI/Oを経由しない高速版)。

正確性の担保:
  music21 の `KeyWeightKeyAnalysis.process()` は Stream を **ピッチクラス分布(音長重み付き)を
  作るためだけ** に使う。したがってトークンから同じ音高・音長を持つ Stream を組み立てれば、
  `converter.parse(midi)` を経由した場合と **同じ解析器・同じ入力** で推定できる。
  合議(K-S / T-K-P / B-B)と投票・閾値・平滑化は `mortm.utils.key` の関数をそのまま再利用する。

省けるのは MIDI 書き出し + `converter.parse` + `makeMeasures` のみ。推定アルゴリズムは不変。
"""

from typing import List, Optional, Tuple

import numpy as np
from music21 import note as m21note, stream as m21stream
from pretty_midi.containers import Note

from ..train.custom_token import ShiftTimeContainer, ChordToken
from ..train.tokenizer import Tokenizer, DURATION_TYPE
from .key import _window_key_vote, _vote_weight, _smooth_spikes


def tokens_to_notes(tokenizer: Tokenizer, seq, tempo: int = 120) -> List[Note]:
    """トークン列 -> pretty_midi.Note のリスト。

    `mortm.utils.de_convert.ct_token_to_midi` と同一の変換器チェーンを使うため、
    音高・開始・終了(秒)は MIDI 書き出し時と一致する。
    """
    seq = list(seq)
    if len(seq) and int(seq[0]) == tokenizer.get("<EOS>"):
        seq = seq[1:]                      # 先頭 <EOS> は捨てる(ct_token_to_midi と同じ)

    notes: List[Note] = []
    back_note = None
    container = ShiftTimeContainer(0, tempo, True)
    cur = Note(pitch=0, velocity=100, start=0, end=0)

    for tid in seq:
        tid = int(tid)
        if tid == tokenizer.get("<TE>"):
            break
        try:
            token = tokenizer.rev_get(tid)
        except (KeyError, IndexError):
            continue        # 語彙外ID(vocab_size の空きスロット)は無視する
        for con in tokenizer.music_token_list:
            if isinstance(con, ChordToken):
                continue
            token_type = con(token=token, note=cur, back_notes=back_note,
                             container=container, tempo=tempo)
            if container.get_inst() is not None:
                container = ShiftTimeContainer(0, tempo, True)   # 楽器切替でリセット
            if token_type == DURATION_TYPE:                      # 音符確定
                back_note = cur
                notes.append(cur)
                cur = Note(pitch=0, velocity=100, start=0, end=0)
    return notes


def notes_to_stream(notes: List[Note], tempo: int = 120) -> m21stream.Stream:
    """pretty_midi.Note 群 -> music21 Stream(音高と quarterLength を保つ)。

    quarterLength = 秒 * (tempo/60)。music21 の pc 分布は quarterLength 重みなので、
    この 2 属性さえ一致すれば推定結果は MIDI 経由と一致する。
    """
    qps = tempo / 60.0                       # quarters per second
    s = m21stream.Stream()
    for n in notes:
        dur = max(1e-6, (n.end - n.start) * qps)
        m = m21note.Note(int(n.pitch))
        m.quarterLength = dur
        s.insert(n.start * qps, m)
    return s


def get_key_from_tokens(tokenizer, seq, tempo: int = 120,
                        window_measures: int = 4, overlap: float = 0.5,
                        corr_threshold: float = 0.2, margin_threshold: float = 0.05,
                        min_segment_windows: int = 2) -> Optional[str]:
    """トークン列の代表キー("C major" 形式)を合議+多数決で返す。推定不能なら None。

    `mortm.utils.key.get_key` と同じ合議(3プロファイル)・同じ閾値・同じ既定窓幅を使う。
    窓は **小節単位** で切る: トークン側には `<SME>` による明示的な小節境界があるので、
    正準版の `makeMeasures` + `window_measures=4` と同じ粒度に揃えられる。
    (quarterLength で切る実装は正準版と一致率 45% しか出なかったため撤回した。)
    """
    from ..train.tokenizer import Tokenizer as _T
    from ..utils.convert import split_sequence_measure

    arr = np.asarray(list(seq), dtype=np.int64)
    if len(arr) and int(arr[0]) == tokenizer.get("<EOS>"):
        arr = arr[1:]
    measures = [np.asarray(m) for m in
                split_sequence_measure(arr, 1, tokenizer.get("<SME>"))]
    measures = [m for m in measures if len(m)]
    M = len(measures)
    if M == 0:
        return None

    step = max(1, int(round(window_measures * (1.0 - overlap))))
    starts = list(range(0, max(M - window_measures, 0) + 1, step)) or [0]

    win_keys: List[Optional[str]] = [None]      # index 0 は未使用(_smooth_spikes の仕様)
    win_weights: List[float] = [0.0]
    for st in starts:
        chunk = np.concatenate(measures[st:st + window_measures])
        notes = tokens_to_notes(tokenizer, chunk, tempo)
        if not notes:
            win_keys.append(None); win_weights.append(0.0); continue
        sub = notes_to_stream(notes, tempo)
        kname, stats = _window_key_vote(sub, corr_threshold, margin_threshold)
        win_keys.append(kname)
        win_weights.append(_vote_weight(stats, corr_threshold, margin_threshold))

    # 窓が少ないと min_segment_windows の平滑化が全票を消してしまうため、
    # 4 窓未満では平滑化を掛けない(短い生成物で推定不能が多発した原因)。
    if len(starts) >= 4:
        win_keys = _smooth_spikes(win_keys, min_segment_windows)

    scores = {}
    for k, w in zip(win_keys[1:], win_weights[1:]):
        if k is not None:
            scores[k] = scores.get(k, 0.0) + w
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def get_key_via_midi(tokenizer, seqs, tempo: int = 120, **kw) -> Optional[str]:
    """トークン -> MIDI 書き出し -> 正準 `mortm.utils.key.get_key` でキー推定する。

    トークン直接版(`get_key_from_tokens`)は正準版と top-1 一致率 38.5% しか出ない。
    差の主因は、正準版が `_choose_rep_part()` で **代表パート1つ** を選ぶのに対し
    直接版が全楽器のピッチクラスを混ぜている点だと考えられる。こちらは MIDI を
    経由するので、窓の切り方・代表パート選択・テンポ処理まで全て正準版と同一になり、
    構造的に一致が保証される(その代わり書き出しと parse の分だけ遅い)。

    Args:
        seqs: {楽器名: トークン列} の dict、または単一のトークン列。
              dict で渡すと **楽器ごとに別トラック** で書き出すので、
              代表パート選択が正準版と同じように働く。
    Returns:
        "C major" 形式の代表キー。推定不能なら None。
    """
    import os
    import tempfile
    from pretty_midi import PrettyMIDI
    from pretty_midi import Instrument
    from .key import get_key

    if not isinstance(seqs, dict):
        seqs = {"_": seqs}

    pm = PrettyMIDI(initial_tempo=tempo)
    for name, s in seqs.items():
        notes = tokens_to_notes(tokenizer, s, tempo)
        if not notes:
            continue
        inst = Instrument(program=0, name=str(name))
        inst.notes = list(notes)
        pm.instruments.append(inst)
    if not pm.instruments:
        return None

    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        pm.write(path)
        return get_key(path, **kw).global_key      # ★ global_key が代表キー
    except Exception:
        return None
    finally:
        if path and os.path.exists(path):
            os.remove(path)
