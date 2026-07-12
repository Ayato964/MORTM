"""方向別セグメント損失（研究設計書 v1.6 付録 B.3）。

H1〜H5 判定の正式な計測器。系列全体 AR CE の下では 1 サンプルが複数の条件付き方向へ
同時に勾配を与えるため、サンプル単位の単一ラベルではなく **ブロック区間 × prefix 条件**で
損失を層別する。真実源は各サンプルの残存ブロック順序（pattern_key 相当。ここでは系列内の
ブロックマーカーから復元し、npz の pattern_keys と一致する）。

付録 B.3（prefix = 当該区間より前の残存ブロック集合）:
  analysis     : META 区間、prefix に音楽ブロック >=1
  infill       : CONST 区間、prefix ⊇ {PAST, FUTURE}
  continuation : CONST 区間、PAST ∈ prefix ∧ FUTURE ∉ prefix
  anticipation : CONST 区間、FUTURE ∈ prefix ∧ PAST ∉ prefix
  meta_gen     : CONST 区間、prefix = {META}
  uncond       : CONST 区間、prefix = ∅

使い方（val 評価で checkpoint に対し呼ぶ想定）:
    masks = direction_masks(seq_ids, tokenizer)   # {bucket: bool mask over target 位置}
    # 位置別 CE を計算し、mask で層別平均する
"""
from __future__ import annotations

import numpy as np

# 論理ブロック名（マーカートークン名 -> 論理名）
_MARKER_NAMES = {
    "<SYSTEM>": "META",
    "<PAST_M>": "PAST",
    "<CONST_M>": "CONST",
    "<FUTURE_M>": "FUTURE",
}
_MUSIC = {"PAST", "CONST", "FUTURE"}
BUCKETS = ("analysis", "infill", "continuation", "anticipation", "meta_gen", "uncond")


def _marker_ids(tokenizer):
    return {tokenizer.get(m): name for m, name in _MARKER_NAMES.items()}


def block_spans(seq, tokenizer):
    """系列を (block_name, start_idx, end_idx_exclusive) のリストに分割する。
    各ブロックはその開始マーカーから次のブロック開始マーカー直前まで（<TAG_END> 含む）。
    先頭の <EOS> 等マーカー前のトークンはどのブロックにも属さない。"""
    marker = _marker_ids(tokenizer)
    idxs = [(i, marker[int(t)]) for i, t in enumerate(seq) if int(t) in marker]
    spans = []
    for k, (start, name) in enumerate(idxs):
        end = idxs[k + 1][0] if k + 1 < len(idxs) else len(seq)
        spans.append((name, start, end))
    return spans


def direction_masks(seq, tokenizer):
    """target 位置（seq[1:] を予測、位置 j は seq[j] を prefix=seq[:j] から予測）に対する
    付録 B.3 バケツ別 bool マスクを返す。dict{bucket: np.bool_ array 長さ len(seq)}。
    位置 0（<EOS>予測相当）は常に False。"""
    seq = np.asarray(seq)
    n = len(seq)
    spans = block_spans(seq, tokenizer)
    # 各位置 j が属するブロックと、その開始位置(=block start marker idx)
    block_of = [None] * n
    block_start = [None] * n
    for name, s, e in spans:
        for j in range(s, e):
            block_of[j] = name
            block_start[j] = s
    masks = {b: np.zeros(n, dtype=bool) for b in BUCKETS}
    for j in range(1, n):
        blk = block_of[j]
        if blk is None:
            continue
        # prefix = この位置のブロックより前に開始した残存ブロックの集合
        bs = block_start[j]
        prefix = {name for name, s, _ in spans if s < bs}
        music_in_prefix = prefix & _MUSIC
        if blk == "META":
            if music_in_prefix:                      # analysis
                masks["analysis"][j] = True
        elif blk == "CONST":
            has_p, has_f = "PAST" in prefix, "FUTURE" in prefix
            if has_p and has_f:
                masks["infill"][j] = True
            elif has_p and not has_f:
                masks["continuation"][j] = True
            elif has_f and not has_p:
                masks["anticipation"][j] = True
            elif prefix == {"META"}:
                masks["meta_gen"][j] = True
            elif not prefix:
                masks["uncond"][j] = True
        # PAST/FUTURE 区間自体の損失は方向別集計の対象外（文脈側）
    return masks


def bucket_token_counts(seq, tokenizer):
    """デバッグ/監査用: 各バケツの target トークン数。"""
    m = direction_masks(seq, tokenizer)
    return {b: int(v.sum()) for b, v in m.items()}
