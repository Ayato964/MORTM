# get_key.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter

from music21 import converter, analysis, stream, tempo, key


# ==============================
# データクラス
# ==============================
@dataclass
class KeySeg:
    start_quarter: float
    end_quarter: float
    start_time: float
    end_time: float
    key_name: str          # 例: "C major" / "Unknown"
    tonic: str             # 例: "C" / ""
    mode: str              # "major"/"minor" / ""
    votes: int             # 本実装では RLE 後の集計なので 0 を基本値に
    profiles: Dict[str, Dict[str, float]]  # 本実装では省略（必要なら後処理で再集計）


@dataclass
class GetKeyResult:
    global_key: Optional[str]           # 全体で最頻のキー（Unknown除外）
    segments: List[KeySeg]              # タイムスタンプ付きキー区間（連続・非重複）
    config: Dict[str, Any]              # 使ったハイパラを記録（再現性用）


# ==============================
# Tempo マップ（四分音符→秒）
# ==============================
def _build_tempo_segments(s: stream.Score) -> List[Tuple[float, float, float, float]]:
    """
    戻り値: [(q_start, q_end, bpm, sec_start), ...]
    """
    marks = sorted(
        s.flat.getElementsByClass(tempo.MetronomeMark),
        key=lambda m: float(m.offset)
    )
    if not marks:
        m = tempo.MetronomeMark(number=120)
        m.offset = 0.0
        marks = [m]

    q_starts = [float(m.offset) for m in marks]
    q_ends = q_starts[1:] + [float(s.highestTime)]

    segments = []
    sec_acc = 0.0
    for q0, q1, m in zip(q_starts, q_ends, marks):
        bpm = m.number if m.number is not None else m.getQuarterBPM()
        bpm = float(bpm if bpm else 120.0)
        spq = 60.0 / bpm
        segments.append((q0, q1, bpm, sec_acc))
        sec_acc += max(0.0, q1 - q0) * spq
    return segments


def _quarters_to_seconds(segments: List[Tuple[float, float, float, float]], q: float) -> float:
    for q0, q1, bpm, sec0 in segments:
        if q <= q1:
            spq = 60.0 / bpm
            return sec0 + max(0.0, q - q0) * spq
    # 末尾を超えた場合の安全策
    q0, q1, bpm, sec0 = segments[-1]
    spq = 60.0 / bpm
    return sec0 + max(0.0, q - q0) * spq


# ==============================
# 小節関連
# ==============================
def _choose_rep_part(s: stream.Score) -> stream.Part | stream.Stream:
    parts = list(s.parts) if s.parts else [s]
    lengths = []
    for p in parts:
        ms = list(p.getElementsByClass(stream.Measure))
        lengths.append(len(ms))
    idx = max(range(len(parts)), key=lambda i: lengths[i])
    return parts[idx]


def _measure_start_quarter(part: stream.Part, mnum: int) -> float:
    """
    1始まりの小節番号 mnum の開始四分音符位置を返す。
    mnum が実在しない（末尾+1）場合は highestTime を返す。
    """
    m = part.measure(mnum)
    return float(m.offset) if m is not None else float(part.highestTime)


def _make_windows(num_measures: int, win_meas: int, overlap: float) -> List[Tuple[int, int]]:
    step = max(1, int(round(win_meas * (1.0 - overlap))))
    windows = []
    i = 1
    while i <= num_measures:
        j = min(num_measures, i + win_meas - 1)
        windows.append((i, j))
        if j == num_measures:
            break
        i += step
    return windows


# ==============================
# 1窓あたりキー推定（3プロファイル合議）
# ==============================
def _profiles() -> Dict[str, analysis.discrete.DiscreteAnalysis]:
    return {
        "KrumhanslSchmuckler": analysis.discrete.KrumhanslSchmuckler(),
        "TemperleyKostkaPayne": analysis.discrete.TemperleyKostkaPayne(),
        "BellmanBudge": analysis.discrete.BellmanBudge(),
    }


def _window_key_vote(sub, corr_thr: float, margin_thr: float):
    """
    その窓のキーを合議で決める（K-S / T-K-P / B-B）。
    返り値: (勝者キー名 or None, プロファイル別スコア dict)
    """
    profs = _profiles()

    stats = {}     # {profile: {"corr": float, "margin": float}}
    votes = []     # 閾値通過したキー名を投票

    for name, proc in profs.items():
        try:
            k_top = proc.getSolution(sub)  # Key オブジェクト
        except Exception:
            continue

        top_corr = float(getattr(k_top, 'correlationCoefficient', 0.0))
        alts = getattr(k_top, 'alternateInterpretations', [])
        if alts:
            second_corr = float(getattr(alts[0], 'correlationCoefficient', top_corr))
        else:
            second_corr = float('-inf')  # 代替が無ければ margin 大きめ扱い

        margin = top_corr - second_corr if second_corr != float('-inf') else top_corr
        kname = f"{k_top.tonic.name} {k_top.mode}"

        stats[name] = {"corr": top_corr, "margin": margin}

        if (top_corr >= corr_thr) and (margin >= margin_thr):
            votes.append(kname)

    if not votes:
        return None, stats

    cnt = Counter(votes).most_common()
    top_freq = cnt[0][1]
    candidates = [k for k, c in cnt if c == top_freq]
    if len(candidates) == 1:
        return candidates[0], stats

    # tie-break: K-S を優先
    try:
        ks = analysis.discrete.KrumhanslSchmuckler().getSolution(sub)
        ks_key = f"{ks.tonic.name} {ks.mode}"
        if ks_key in candidates:
            return ks_key, stats
    except Exception:
        pass

    return candidates[0], stats


# ==============================
# 後処理（方式B: 小節多数決 → RLE）
# ==============================
def _vote_weight(stats: Dict[str, Dict[str, float]],
                 corr_thr: float,
                 margin_thr: float) -> float:
    """
    3プロファイル中、(corr>=thr & margin>=thr) を満たした数を重みとする。
    """
    return float(sum(1 for v in stats.values()
                     if (v["corr"] >= corr_thr and v["margin"] >= margin_thr)))


def _smooth_spikes(keys: List[Optional[str]],
                   min_len: int) -> List[Optional[str]]:
    """
    連続長 < min_len の島を前後に吸収する簡易平滑化。
    keys[1..M] を想定（index 0は未使用）。
    """
    n = len(keys) - 1
    i = 1
    while i <= n:
        j = i
        while j <= n and keys[j] == keys[i]:
            j += 1
        run_len = j - i
        if run_len < min_len:
            left = keys[i - 1] if i - 1 >= 1 else None
            right = keys[j] if j <= n else None
            repl = right if right is not None else left
            for t in range(i, j):
                keys[t] = repl
        i = j
    return keys


# ==============================
# メイン
# ==============================
def get_key(
        midi_path: str,
        window_measures: int = 4,
        overlap: float = 0.5,
        corr_threshold: float = 0.2,
        margin_threshold: float = 0.05,
        min_segment_measures: int = 2,
) -> GetKeyResult:
    """
    方式B:
      1) 小節窓でキー推定
      2) 小節ごとに票（重み）を加点して多数決
      3) 小節系列を RLE 圧縮して区間化
      4) 四分音符・秒に変換（連続かつ非重複。Unknown を明示）
    """
    # --- パース & 代表パート ---
    s = converter.parse(midi_path)
    s.makeMeasures(inPlace=True)
    rep = _choose_rep_part(s)

    # --- 小節数 / 窓の準備 ---
    measures = list(rep.getElementsByClass(stream.Measure))
    M = len(measures)
    windows = _make_windows(M, window_measures, overlap)

    # --- Tempo map ---
    tempo_segments = _build_tempo_segments(s)

    # --- 各窓のキー決定 ---
    win_results = []  # (start_m, end_m, keyName|None, stats)
    for (m0, m1) in windows:
        sub = s.measures(m0, m1)
        kname, stats = _window_key_vote(sub, corr_threshold, margin_threshold)
        win_results.append((m0, m1, kname, stats))

    # --- 小節ごとの票集計 ---
    # measure_votes[m] = { key_name: score }
    measure_votes: List[Dict[str, float]] = [defaultdict(float) for _ in range(M + 1)]  # 1..M
    for (m0, m1, kname, stats) in win_results:
        if kname is None:
            continue
        w = _vote_weight(stats, corr_threshold, margin_threshold)
        if w <= 0:
            continue
        for m in range(m0, m1 + 1):
            measure_votes[m][kname] += w

    # --- 小節単位のキー決定（argmax; 票が無ければ None=Unknown） ---
    per_measure_key: List[Optional[str]] = [None] * (M + 1)  # 1..M
    for m in range(1, M + 1):
        if not measure_votes[m]:
            per_measure_key[m] = None
            continue
        k, _ = max(measure_votes[m].items(), key=lambda kv: kv[1])
        per_measure_key[m] = k

    # --- 短いスパイク平滑化 ---
    if min_segment_measures > 1:
        per_measure_key = _smooth_spikes(per_measure_key, min_len=min_segment_measures)

    # --- RLE で連続区間へ ---
    segments: List[KeySeg] = []
    i = 1
    while i <= M:
        cur_key = per_measure_key[i]
        # Unknown を明示
        label = cur_key if cur_key is not None else "Unknown"
        j = i
        while j <= M:
            nxt = per_measure_key[j]
            if (nxt if nxt is not None else "Unknown") != label:
                break
            j += 1

        # 小節→四分音符→秒
        q0 = _measure_start_quarter(rep, i)
        q1 = _measure_start_quarter(rep, min(j, M + 1))
        t0 = _quarters_to_seconds(tempo_segments, q0)
        t1 = _quarters_to_seconds(tempo_segments, q1)

        if label == "Unknown":
            tonic_str, mode_str = "", ""
        else:
            tonic_str, mode_str = label.split()

        segments.append(
            KeySeg(
                start_quarter=q0, end_quarter=q1,
                start_time=t0, end_time=t1,
                key_name=label,
                tonic=tonic_str,
                mode=(mode_str.lower() if mode_str else ""),
                votes=0,          # 多数決→RLE後なので 0 に固定（必要なら別途再集計）
                profiles={},      # 必要なら窓側の統計をここに戻し入れる処理を追加
            )
        )
        i = j

    # --- グローバルキー（Unknownは無視、区間長で加重最頻） ---
    ctr = Counter()
    for seg in segments:
        if seg.key_name != "Unknown":
            ctr[seg.key_name] += (seg.end_quarter - seg.start_quarter)
    global_key = ctr.most_common(1)[0][0] if ctr else None

    return GetKeyResult(
        global_key=global_key,
        segments=segments,
        config=dict(
            postprocess="measure_vote_RLE",
            window_measures=window_measures,
            overlap=overlap,
            corr_threshold=corr_threshold,
            margin_threshold=margin_threshold,
            min_segment_measures=min_segment_measures,
            profiles=list(_profiles().keys()),
        ),
    )


# ==============================
# 便利ラッパ
# ==============================
def get_key_dict(
        midi_path: str,
        window_measures: int = 4,
        overlap: float = 0.5,
        corr_threshold: float = 0.2,
        margin_threshold: float = 0.05,
        min_segment_measures: int = 2,
) -> Dict[str, Any]:
    res = get_key(
        midi_path,
        window_measures=window_measures,
        overlap=overlap,
        corr_threshold=corr_threshold,
        margin_threshold=margin_threshold,
        min_segment_measures=min_segment_measures,
    )
    return {
        "global_key": res.global_key,
        "segments": [asdict(seg) for seg in res.segments],
        "config": res.config,
    }
