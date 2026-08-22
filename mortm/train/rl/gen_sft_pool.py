"""生成SFTデータセット(MORTM4.5D-160M-SFT-gen の学習データ)を CARL の両フェーズで使う。

    /home/takaaki-nagoshi/data/sft/generation/{train,eval}.json  (81,457 ファイル)

なぜ分析SFTではなくこちらを使うか:
  `convert_analysis_sft` が作る分析SFTは **1〜8 小節の単一 CONST** しか持たない。
  一方 MORTM-gem が実際に解くのは PAST/FUTURE を伴う補完タスク群で、そこでは
  「人間が書いた PAST」と「AI が書いた CONST」と「人間が書いた FUTURE」が
  **つなぎ目を持って隣接する**。判別器にこのつなぎ目を見せないと、AI 由来性の
  判定から最も情報量の多い手がかりが抜け落ちる。
  そこで両フェーズとも生成SFTデータを使い、時系列順に畳んだ **最大24小節の
  統一 CONST** を分析対象にする。

サンプル形式(convert_foundation.convert_generation_sft):

    meta        : <EOS> <SYSTEM>..<TAG_END>                          <MGEN> [CoT] CONST <TE>
    meta_past   : <EOS> <SYSTEM>..<TAG_END> <PAST_M>..<TAG_END>      <MGEN> [CoT] CONST <TE>
    meta_future : <EOS> <SYSTEM>..<TAG_END> <FUTURE_M>..<TAG_END>    <MGEN> [CoT] CONST <TE>
    infill      : <EOS> <SYSTEM>..<TAG_END> <PAST_M>.. <FUTURE_M>..  <MGEN> [CoT] CONST <TE>
    inst_comp   : <EOS> <SYSTEM>..<TAG_END> <CONST_M>[他楽器]..      <MGEN> [CoT] CONST <TE>

  [CoT] は約 41% のサンプルに付く `<SYSTEM>..<TAG_END>`(生成前に META を書き下す)。

判別器へ渡す形は分析SFTと同じ:

    <EOS> <CONST_M>[<INST_x> 畳んだ系列 <ESEQ>]* <TAG_END> <META> <SYSTEM>..<TAG_END> <TE>

  違いは CONST が PAST++CONST++FUTURE を連結した最大24小節になる点だけなので、
  MORTM-ana の入出力仕様は変えずに済む。

答え META から GMC は落とす:
  畳んだ入力からは「どの区間が生成対象だったか」が原理的に判別できないため、
  `<GEN_MEASURE_COUNT_k>` は ana にとって不可知なラベルになる。小節数は
  `task_norm.structural_check` が生成物の `<SME>` を数えて厳密に求められるので、
  推論させる価値もない。
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .task_norm import (normalize_to_const, parse_gen_output, parse_melody_block,
                        rev_safe, strip_cot_meta)


TASKS = ("meta", "meta_past", "meta_future", "infill", "inst_comp")

_MARKERS = ("<SYSTEM>", "<PAST_M>", "<CONST_M>", "<FUTURE_M>")


# ----------------------------------------------------------------------
# 1 サンプルの分解
# ----------------------------------------------------------------------
def parse_gen_sft_sample(sample: Sequence[int], tokenizer) -> Optional[dict]:
    """生成SFTサンプル -> 条件・正解・タスク種別に分解する。

    Returns:
        {"task", "prompt", "cond_meta", "past", "future", "other_inst", "gt"}
        `prompt` は `<MGEN>` までを含む gem への入力そのもの。
        `gt` は `<MGEN>` の後ろ(CoT と `<TE>` は除去済み)= 人間が書いた正解 CONST。
        壊れたサンプルは None。
    """
    ids = [int(t) for t in sample]
    MGEN, TAG_END, TE = (tokenizer.get("<MGEN>"), tokenizer.get("<TAG_END>"),
                         tokenizer.get("<TE>"))
    if MGEN not in ids:
        return None
    mg = ids.index(MGEN)
    head, tail = ids[:mg], ids[mg + 1:]

    # head を [SYSTEM, PAST_M, CONST_M, FUTURE_M] のブロックに割る(固定順・各 TAG_END 終端)
    mark_ids = {tokenizer.get(m): m for m in _MARKERS}
    blocks: Dict[str, List[int]] = {}
    cur: Optional[str] = None
    for t in head:
        if t in mark_ids:
            cur = mark_ids[t]
            blocks[cur] = []
            continue
        if t == TAG_END:
            cur = None
            continue
        if cur is not None:
            blocks[cur].append(t)

    if "<SYSTEM>" not in blocks:
        return None

    has_p, has_f = "<PAST_M>" in blocks, "<FUTURE_M>" in blocks
    has_c = "<CONST_M>" in blocks
    task = ("infill" if has_p and has_f else
            "meta_past" if has_p else
            "meta_future" if has_f else
            "inst_comp" if has_c else "meta")

    def _blk(name):
        if name not in blocks:
            return None
        return parse_melody_block(blocks[name] + [TAG_END], tokenizer)

    # 正解フィールド。CoT(生成前に書き下す META)と <TE> は落とす。
    # R_sim の参照に使うので、音楽以外のトークンを混ぜない。
    gt_field = [int(t) for t in strip_cot_meta(tail, tokenizer) if int(t) != TE]

    return {
        "task": task,
        "prompt": np.asarray(ids[:mg + 1], dtype=np.int64),
        # 条件 META は `<SYSTEM>` マーカーを含む生のトークン列にしておく
        # (structural_check / parse_meta がこの形を前提にしている)
        "cond_meta": np.asarray([tokenizer.get("<SYSTEM>")] + blocks["<SYSTEM>"]
                                + [TAG_END], dtype=np.int64),
        "past": _blk("<PAST_M>"),
        "future": _blk("<FUTURE_M>"),
        "other_inst": _blk("<CONST_M>"),
        "gt": np.asarray(gt_field, dtype=np.int64),
    }


# ----------------------------------------------------------------------
# 統一 CONST への畳み込み
# ----------------------------------------------------------------------
def fold_to_const(rec: dict, gen_field: Sequence[int], tokenizer
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """条件側の人間ブロックと生成フィールドを時系列順に畳んで統一 CONST にする。

    `gen_field` に人間の正解(`rec["gt"]`)を渡せば Human サンプル、
    gem の出力を渡せば AI サンプルになる。**同じ関数を通す**ことが重要で、
    畳み方の違いが AI/Human の手がかりになってしまうのを防ぐ。
    """
    return normalize_to_const(rec["task"], gen_field, tokenizer,
                              past=rec.get("past"), future=rec.get("future"),
                              other_inst=rec.get("other_inst"))


def strip_gmc(meta_ids: Sequence[int], tokenizer) -> np.ndarray:
    """答え META から `<GEN_MEASURE_COUNT_k>` を落とす。

    畳んだ CONST からは生成区間の長さが決定できないので、ana に推論させると
    ラベルとして不可知になる(学習を濁らせるだけ)。小節数はルールベースで
    厳密に数えられるため、報酬側は `structural_check` が担当する。
    """
    out = []
    for t in meta_ids:
        name = rev_safe(tokenizer, int(t))
        if isinstance(name, str) and name.startswith("<GEN_MEASURE_COUNT_"):
            continue
        out.append(int(t))
    return np.asarray(out, dtype=np.int64)


def blank_measure_ratio(seqs: Dict[str, np.ndarray], tokenizer) -> float:
    """{楽器: 系列} の **合計小節数** に占める空小節(音符ゼロの小節)の割合。

    `<SME>` は小節の **先頭** に置かれるので、単純に `<SME>` を見た時点で
    直前までの音符を集計すると 1 小節ぶんずれる(最終小節も数え落とす)。
    区間分割は `convert.split_sequence_measure` に任せ、判定条件は
    `_build_melody_block` の「S トークンが1つも無い = 空」と揃える。
    """
    from ...utils.convert import split_sequence_measure

    lo, hi = tokenizer.get_length_tuple("s")
    SME = tokenizer.get("<SME>")
    total = blank = 0
    for seq in seqs.values():
        for m in split_sequence_measure(np.asarray(seq), 1, SME):
            m = np.asarray(m)
            total += 1
            if not np.any((m >= lo) & (m < hi)):
                blank += 1
    return (blank / total) if total else 1.0


# ----------------------------------------------------------------------
# プール構築
# ----------------------------------------------------------------------
def load_gen_sft_pool(manifest: str, tokenizer, limit: int = 20000,
                      max_blank_ratio: float = 0.5,
                      tasks: Optional[Sequence[str]] = None,
                      rng: Optional[np.random.Generator] = None) -> List[dict]:
    """生成SFTの npz 群から、両フェーズで使えるレコードを読み込む。

    各レコードは `parse_gen_sft_sample` の戻りに加えて
      "seq"  : 人間の正解を畳んだ統一 CONST ブロック(判別器の Human 入力)
      "meta" : GMC を除いた答え META(`<SYSTEM>..<TAG_END>`)
    を持つ。偶数ラウンドは "prompt" と past/future を、奇数ラウンドは
    "seq"/"meta" を使う。**同一レコードから両方が作れる**ので、
    条件側の分布が2フェーズ間でずれない。

    max_blank_ratio: 畳んだ正解の空小節が半数以上なら捨てる(人間側の正解と
        入力クエリにのみ掛ける。gem の生成物に `<BLANK>` が出るのは正常)。
    """
    paths = json.load(open(manifest))
    if rng is not None:
        paths = [paths[i] for i in rng.permutation(len(paths))]
    want = set(tasks) if tasks else set(TASKS)

    pool: List[dict] = []
    dropped = {"blank_gt": 0, "blank_query": 0, "parse": 0, "empty": 0}
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                for k in d.files:
                    a = np.asarray(d[k])
                    if a.ndim != 1 or len(a) < 16:
                        continue
                    rec = parse_gen_sft_sample(a, tokenizer)
                    if rec is None or rec["task"] not in want:
                        dropped["parse"] += 1
                        continue
                    const, merged = fold_to_const(rec, rec["gt"], tokenizer)
                    if not merged:
                        dropped["empty"] += 1
                        continue
                    # ★ 空小節フィルタは **入力クエリ** と **人間の正解** を
                    #   それぞれ独立に見る。畳んだ全体で一度に見ると、密な PAST の陰に
                    #   スカスカな正解が隠れて通過してしまう(実測 3.1%)。
                    #   gem の生成物に <BLANK> が出るのは正常なので、そちらは対象外。
                    if blank_measure_ratio(parse_gen_output(rec["gt"], tokenizer),
                                           tokenizer) >= max_blank_ratio:
                        dropped["blank_gt"] += 1
                        continue
                    query = {}
                    for name in ("past", "future", "other_inst"):
                        if rec.get(name):
                            query.update(rec[name])
                    if query and blank_measure_ratio(query, tokenizer) >= max_blank_ratio:
                        dropped["blank_query"] += 1
                        continue
                    rec["seq"] = const
                    rec["meta"] = strip_gmc(rec["cond_meta"], tokenizer)
                    pool.append(rec)
                    if len(pool) >= limit:
                        break
        except Exception:
            continue
        if len(pool) >= limit:
            break

    n = len(pool)
    from collections import Counter
    c = Counter(r["task"] for r in pool)
    print(f"[CARL] 生成SFTプール {n} 件 "
          f"({' / '.join(f'{k} {v}' for k, v in c.most_common())})")
    print(f"[CARL]   除外: 空小節過多(正解 {dropped['blank_gt']} / 入力クエリ "
          f"{dropped['blank_query']}) / 解析不能 {dropped['parse']} / 空 {dropped['empty']} "
          f"(閾値 {max_blank_ratio})")
    return pool[:limit]
