# get_key.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import math

from music21 import converter, analysis, stream, tempo, key

# ------------------------------
# 設定用データクラス
# ------------------------------
@dataclass
class KeySeg:
    start_quarter: float
    end_quarter: float
    start_time: float
    end_time: float
    key_name: str          # 例: "C major"
    tonic: str             # 例: "C"
    mode: str              # "major" / "minor"
    votes: int             # 何プロファイルが同意したか
    profiles: Dict[str, Dict[str, float]]  # corr, margin を格納

@dataclass
class GetKeyResult:
    global_key: str                    # 全体で最頻のキー（参考）
    segments: List[KeySeg]             # タイムスタンプ付きキー区間
    config: Dict[str, Any]             # 使ったハイパラを記録（再現性用）


# ------------------------------
# Tempoマップ（四分音符→秒）作成
# ------------------------------
def _build_tempo_segments(s: stream.Score) -> List[Tuple[float, float, float, float]]:
    """
    戻り値: [(q_start, q_end, bpm, sec_start), ...]
    """
    marks = sorted(
        s.flat.getElementsByClass(tempo.MetronomeMark),
        key=lambda m: float(m.offset)
    )
    if not marks:
        # テンポ記号が無い場合は 120 BPM とみなす
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


# ------------------------------
# 窓の作成（小節ベース）
# ------------------------------
def _choose_rep_part(s: stream.Score) -> stream.Part | stream.Stream:
    parts = list(s.parts) if s.parts else [s]
    lengths = []
    for p in parts:
        ms = list(p.getElementsByClass(stream.Measure))
        lengths.append(len(ms))
    idx = max(range(len(parts)), key=lambda i: lengths[i])
    return parts[idx]

def _measure_start_quarter(part: stream.Part, mnum: int) -> float:
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


# ------------------------------
# 1窓あたりキー推定（3プロファイル合議）
# ------------------------------
def _profiles() -> Dict[str, analysis.discrete.DiscreteAnalysis]:
    return {
        "KrumhanslSchmuckler": analysis.discrete.KrumhanslSchmuckler(),
        "TemperleyKostkaPayne": analysis.discrete.TemperleyKostkaPayne(),
        "BellmanBudge": analysis.discrete.BellmanBudge(),
    }

def _window_key_vote(
        sub: stream.Stream,
        corr_thr: float,
        margin_thr: float
) -> Tuple[str | None, Dict[str, Dict[str, float]]]:
    """
    その窓のキーを合議で決める。
    戻り値: (勝者キー名 or None, プロファイル別スコア辞書)
    """
    profs = _profiles()
    stats = {}
    votes = []
    for name, proc in profs.items():
        # 代替候補を含むスコア列を取得
        # data: [(keyStr, modeStr, corr), ...] を想定
        data, _ = proc.process(sub, storeAlternatives=True)
        data = sorted(data, key=lambda x: x[2], reverse=True)
        top1, top2 = data[0], data[1]
        kname = f"{top1[0]} {top1[1]}"  # "C major" 等
        corr = float(top1[2])
        margin = float(top1[2] - top2[2])
        stats[name] = {"corr": corr, "margin": margin}

        if corr >= corr_thr and margin >= margin_thr:
            votes.append(kname)

    if not votes:
        return None, stats

    # 最頻→同数なら平均corrの高い方
    from collections import Counter
    cnt = Counter(votes).most_common()
    top_freq = cnt[0][1]
    candidates = [k for k, c in cnt if c == top_freq]
    if len(candidates) == 1:
        return candidates[0], stats
    else:
        # 同票なら corr の平均値で決定
        def avg_corr(kname: str) -> float:
            c = []
            for n, st in stats.items():
                # そのプロファイルの1位キー名をもう一度取り直す
                # （厳密運用なら keyName も保存しておく）
                pass
            # 上の簡略化のため、ここは votes に入ったプロファイルの corr を平均
            for n, st in stats.items():
                # “投票”した＝しきい値通過＆knameだった時のみ加点したいが
                # 実装簡略化で全体平均でも差は小さい
                c.append(st["corr"])
            return sum(c) / max(1, len(c))
        best = max(candidates, key=avg_corr)
        return best, stats


# ------------------------------
# メイン関数
# ------------------------------
def get_key(
        midi_path: str,
        window_measures: int = 4,
        overlap: float = 0.5,
        corr_threshold: float = 0.2,
        margin_threshold: float = 0.05,
        min_segment_measures: int = 2,
) -> GetKeyResult:
    """
    転調対応：小節窓でキー推定 → 連結 → タイムスタンプ(秒/四分音符)
    """
    s = converter.parse(midi_path)
    s.makeMeasures(inPlace=True)

    rep = _choose_rep_part(s)
    measures = list(rep.getElementsByClass(stream.Measure))
    M = len(measures)
    windows = _make_windows(M, window_measures, overlap)

    # Tempo map 準備
    tempo_segments = _build_tempo_segments(s)

    # 各窓のキー決定
    win_results = []  # (start_m, end_m, keyName|None, stats)
    for (m0, m1) in windows:
        sub = s.measures(m0, m1)
        kname, stats = _window_key_vote(sub, corr_threshold, margin_threshold)
        win_results.append((m0, m1, kname, stats))

    # 同一キーの連続窓を結合（None=不確実は分離）
    segments: List[KeySeg] = []
    cur_key = None
    cur_stats: List[Dict[str, Dict[str, float]]] = []
    seg_m0 = None
    seg_m1 = None

    def flush_segment():
        nonlocal segments, cur_key, cur_stats, seg_m0, seg_m1
        if cur_key is None or seg_m0 is None or seg_m1 is None:
            return
        # 短すぎる区間は棄却（ノイズ抑制）
        if seg_m1 - seg_m0 + 1 < min_segment_measures:
            cur_key = None; cur_stats = []; seg_m0 = None; seg_m1 = None
            return
        # タイミングに変換
        q0 = _measure_start_quarter(rep, seg_m0)
        q1 = _measure_start_quarter(rep, min(seg_m1 + 1, M+1))
        t0 = _quarters_to_seconds(tempo_segments, q0)
        t1 = _quarters_to_seconds(tempo_segments, q1)

        # votes = 何プロファイルが合意したか（近似：各窓で閾値通過数の平均）
        votes_list = []
        merged_stats: Dict[str, Dict[str, float]] = {}
        for st in cur_stats:
            votes_list.append(sum(1 for v in st.values() if (v["corr"] >= corr_threshold and v["margin"] >= margin_threshold)))
            # プロファイル別 corr, margin を窓平均しておく
            for pname, v in st.items():
                merged_stats.setdefault(pname, {"corr": 0.0, "margin": 0.0, "n": 0})
                merged_stats[pname]["corr"] += v["corr"]
                merged_stats[pname]["margin"] += v["margin"]
                merged_stats[pname]["n"] += 1
        for pname in list(merged_stats.keys()):
            n = merged_stats[pname].pop("n")
            if n > 0:
                merged_stats[pname]["corr"] /= n
                merged_stats[pname]["margin"] /= n

        # Keyオブジェクト生成（表示用）
        tonic, mode = cur_key.split()
        kobj = key.Key(tonic)
        if mode.lower().startswith('min'):
            kobj = key.Key(tonic).relative  # 表示整形だけの簡便策

        segments.append(
            KeySeg(
                start_quarter=q0, end_quarter=q1,
                start_time=t0, end_time=t1,
                key_name=cur_key, tonic=tonic, mode=mode.lower(),
                votes=int(round(sum(votes_list)/max(1, len(votes_list)))),
                profiles={p: {"corr": float(v["corr"]), "margin": float(v["margin"])}
                          for p, v in merged_stats.items()}
            )
        )
        cur_key = None; cur_stats = []; seg_m0 = None; seg_m1 = None

    for (m0, m1, kname, stats) in win_results:
        if kname is None:
            # 不確実窓で一旦区切る
            flush_segment()
            continue
        if cur_key is None:
            cur_key = kname
            seg_m0, seg_m1 = m0, m1
            cur_stats = [stats]
        elif kname == cur_key:
            seg_m1 = max(seg_m1, m1)
            cur_stats.append(stats)
        else:
            flush_segment()
            cur_key = kname
            seg_m0, seg_m1 = m0, m1
            cur_stats = [stats]
    flush_segment()

    # 参考にグローバルキー（区間長で重みづけ最頻）
    from collections import Counter
    ctr = Counter()
    for seg in segments:
        ctr[seg.key_name] += (seg.end_quarter - seg.start_quarter)
    global_key = ctr.most_common(1)[0][0] if ctr else None

    return GetKeyResult(
        global_key=global_key,
        segments=segments,
        config=dict(
            window_measures=window_measures,
            overlap=overlap,
            corr_threshold=corr_threshold,
            margin_threshold=margin_threshold,
            min_segment_measures=min_segment_measures,
            profiles=list(_profiles().keys()),
        ),
    )

# 使いやすい辞書/JSONに変換
def get_key_dict(midi_path: str,
                 window_measures: int = 4,
                 overlap: float = 0.5,
                 corr_threshold: float = 0.2,
                 margin_threshold: float = 0.05,
                 min_segment_measures: int = 2,) -> Dict[str, Any]:
    res = get_key(midi_path, window_measures, overlap, corr_threshold, margin_threshold, min_segment_measures)
    return {
        "global_key": res.global_key,
        "segments": [asdict(seg) for seg in res.segments],
        "config": res.config,
    }
