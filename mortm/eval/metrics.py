"""客観指標（研究設計書 v1.6 §6.2, §9.4）。

トークン列(生成/GT)上で計算できる指標を実装。既存 `utils/eval.py` を将来統合。
音符表現は (measure, position, pitch, duration) に構文解析（§9.12 の parse と整合）。

含む指標:
- 形式妥当性: 文法違反率(範囲外/シフト逆行)、horizon 遵守(生成小節数一致・|Δ|)
- seam(CONST 境界): S1 境界隣接音程分布 W1、S2 境界 IOI 分布 W1、S4 密度段差
  ※ S3(コードトーン適合)は基盤 META にコード不在(E0-7)のため N/A。chord 条件データでのみ有効。
- 分布/多様性: ピッチクラス JS、音価 JS、pitch-interval 3-gram 重複率、8-gram 完全一致率(剽窃)
- 統計: ブートストラップ 95%CI

依存を増やさないため JS/Wasserstein-1 は numpy 実装。
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# 構文解析: トークン列 -> 音符 (measure, position, pitch, duration)
# ----------------------------------------------------------------------
def parse_notes(clip, tokenizer):
    s_lo, s_hi = tokenizer.get_length_tuple("s")
    p_lo, p_hi = tokenizer.get_length_tuple("p")
    d_lo, d_hi = tokenizer.get_length_tuple("d")
    sme = tokenizer.get("<SME>")
    notes, measure, cur_s, cur_p = [], -1, None, None
    for tok in clip:
        v = int(tok)
        if v == sme:
            measure += 1
        elif s_lo <= v < s_hi:
            cur_s = v - s_lo
        elif p_lo <= v < p_hi:
            cur_p = v - p_lo
        elif d_lo <= v < d_hi:
            if cur_s is not None and cur_p is not None:
                notes.append((max(measure, 0), cur_s, cur_p, v - d_lo))
                cur_p = None
    return notes


# ----------------------------------------------------------------------
# 形式妥当性
# ----------------------------------------------------------------------
def grammar_violations(clip, tokenizer):
    """文法違反数: (a) 小節内シフト逆行(position が同一小節内で減少)、(b) 範囲外トークンは
    構文上発生しないため 0。返り値 dict。"""
    s_lo, s_hi = tokenizer.get_length_tuple("s")
    sme = tokenizer.get("<SME>")
    last_s, reversals, steps = -1, 0, 0
    for tok in clip:
        v = int(tok)
        if v == sme:
            last_s = -1
        elif s_lo <= v < s_hi:
            s = v - s_lo
            if s < last_s:
                reversals += 1
            last_s = s
            steps += 1
    return {"shift_reversals": reversals, "shift_steps": steps,
            "reversal_rate": reversals / steps if steps else 0.0}


def n_measures(clip, tokenizer):
    sme = tokenizer.get("<SME>")
    return int(np.sum(np.asarray(clip) == sme))


def horizon_adherence(gen_clip, target_measures, tokenizer):
    g = n_measures(gen_clip, tokenizer)
    return {"gen_measures": g, "target_measures": target_measures,
            "match": g == target_measures, "abs_delta": abs(g - target_measures)}


# ----------------------------------------------------------------------
# 分布ヘルパ (numpy)
# ----------------------------------------------------------------------
def _hist(vals, k):
    h = np.bincount(np.asarray(vals, dtype=int) % k, minlength=k).astype(float)
    return h / h.sum() if h.sum() > 0 else h


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps; q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein1(a, b):
    """1次元経験分布間の W1(= |累積分布の差の積分|)。a,b は値の配列。"""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    widths = np.diff(np.concatenate([grid, grid[-1:]]))
    return float(np.sum(np.abs(ca - cb) * widths))


# ----------------------------------------------------------------------
# 分布/多様性 (群 vs 群)
# ----------------------------------------------------------------------
def pitch_class_js(notes_gen, notes_gt):
    pc = lambda notes: _hist([p for _, _, p, _ in notes], 12)
    return js_divergence(pc(notes_gen), pc(notes_gt))


def duration_js(notes_gen, notes_gt, k=96 * 3 + 1):
    du = lambda notes: _hist([d for _, _, _, d in notes], k)
    return js_divergence(du(notes_gen), du(notes_gt))


def _pitch_interval_ngrams(notes, n=3):
    ps = [p for _, _, p, _ in sorted(notes)]
    iv = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]
    return set(tuple(iv[i:i + n]) for i in range(len(iv) - n + 1))


def ngram_overlap(notes_gen, notes_gt, n=3):
    g, t = _pitch_interval_ngrams(notes_gen, n), _pitch_interval_ngrams(notes_gt, n)
    return len(g & t) / len(g) if g else 0.0


def eight_gram_exact(notes_gen, train_ngrams: set, n=8):
    """剽窃検査: 生成の pitch-interval 8-gram が訓練集合に完全一致する率。"""
    g = _pitch_interval_ngrams(notes_gen, n)
    return len(g & train_ngrams) / len(g) if g else 0.0


# ----------------------------------------------------------------------
# seam 指標 (CONST 境界)
# ----------------------------------------------------------------------
def boundary_intervals(before_notes, after_notes):
    """境界を跨ぐ隣接音程(半音)。before の最後の音と after の最初の音。"""
    if not before_notes or not after_notes:
        return []
    b = sorted(before_notes)[-1][2]
    a = sorted(after_notes)[0][2]
    return [a - b]


def seam_s1(gen_before, gen_after, gt_before, gt_after):
    """S1: 境界隣接音程分布の W1(生成群 vs GT群)。呼び出し側で群を渡す(ここは1対)。"""
    return boundary_intervals(gen_before, gen_after), boundary_intervals(gt_before, gt_after)


def density_step(before_notes, after_notes):
    """S4: 境界前後の音符密度段差 |d_after - d_before|(小節あたり音符数近似)。"""
    def dens(notes):
        if not notes:
            return 0.0
        measures = max(1, len({m for m, _, _, _ in notes}))
        return len(notes) / measures
    return abs(dens(after_notes) - dens(before_notes))


# ----------------------------------------------------------------------
# 統計: ブートストラップ CI
# ----------------------------------------------------------------------
def bootstrap_ci(values, stat=np.mean, n_boot=10000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi)
