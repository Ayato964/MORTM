"""生成タスクの正規化: あらゆる SFT タスクの出力を **1 本の時系列順 <CONST_M>** に畳む。

MORTM-gem(`MORTM4.5D-160M-SFT-gen`)は Meta2CONST だけでなく、
`convert_foundation.py: convert_generation_sft` の全タスクを解く:

    meta        : [SYSTEM]                                  <MGEN> CONST <TE>
    meta_past   : [SYSTEM] <PAST_M>..<TAG_END>              <MGEN> CONST <TE>
    meta_future : [SYSTEM] <FUTURE_M>..<TAG_END>            <MGEN> CONST <TE>
    infill      : [SYSTEM] <PAST_M>.. <FUTURE_M>..<TAG_END> <MGEN> CONST <TE>
    inst_comp   : [SYSTEM] <CONST_M>[他楽器]..<TAG_END>      <MGEN> [対象楽器CONST] <TE>

MORTM-ana は統一表現しか受け付けないので、どのタスクでも
`<CONST_M> [<INST_x> 時系列順の1本 <ESEQ>]* <TAG_END>` に畳んでから渡す。

**結合は 2 軸ある**:
  - 時間軸 : PAST ++ 生成CONST ++ FUTURE を楽器ごとに連結する。
             これらは元々 `seq_dict[program]` の小節境界で切った **連続スライス** なので、
             単純連結で元のストリームが復元される(`<SME>` が境界を保持している)。
             shift トークンは小節内相対なので、境界さえ保てば連結しても時刻がずれない。
  - 楽器軸 : inst_comp は同一窓の別レイヤーなので、条件側の他楽器と生成側の対象楽器を
             **重ねる**(時間方向には繋がない)。

小節数は PAST/CONST/FUTURE それぞれ可変なので、固定長を仮定しない。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


TASKS = ("meta", "meta_past", "meta_future", "infill", "inst_comp")


def rev_safe(tokenizer, tid):
    """`tokenizer.rev_get` の安全版。未定義IDなら None を返す。

    モデルの vocab_size(700) は tokenizer の語彙数(694)より大きく、ID 694-699 は
    空きスロットである。生成器はこれらを出力できてしまうため、素の rev_get だと
    KeyError で学習が落ちる。生成物は信用できない前提で常にこちらを使う。
    """
    try:
        return tokenizer.rev_get(int(tid))
    except (KeyError, IndexError, TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# ブロックの分解
# ----------------------------------------------------------------------
def parse_melody_block(block: Sequence[int], tokenizer) -> Dict[str, np.ndarray]:
    """`<MARKER> [<INST_x> seq <ESEQ>]* <TAG_END>` -> {楽器名: 系列}。

    先頭マーカーの種類(<PAST_M>/<FUTURE_M>/<CONST_M>)は問わない。
    """
    out: Dict[str, np.ndarray] = {}
    ESEQ, TAG_END = tokenizer.get("<ESEQ>"), tokenizer.get("<TAG_END>")
    cur_name, buf = None, []
    for t in block:
        t = int(t)
        if t == TAG_END:
            break
        name = rev_safe(tokenizer, t)
        if isinstance(name, str) and name.startswith("<INST_"):
            if cur_name is not None:
                out[cur_name] = np.asarray(buf, dtype=np.int64)
            cur_name, buf = name[len("<INST_"):-1], []
            continue
        if t == ESEQ:
            if cur_name is not None:
                out[cur_name] = np.asarray(buf, dtype=np.int64)
            cur_name, buf = None, []
            continue
        if cur_name is not None:
            buf.append(t)
    if cur_name is not None and buf:
        out[cur_name] = np.asarray(buf, dtype=np.int64)
    return out


def strip_cot_meta(gen: Sequence[int], tokenizer) -> np.ndarray:
    """CoT 有効時に `<MGEN>` 直後へ出力されるフル META を剥がす。

    生成フィールドは CoT のとき `[<SYSTEM>..<TAG_END>] [<INST_x> seq <ESEQ>]* <TE>` になる。
    先頭が `<SYSTEM>` なら最初の `<TAG_END>` までを捨てる。
    """
    g = [int(t) for t in gen]
    if not g:
        return np.asarray([], dtype=np.int64)
    if g[0] == tokenizer.get("<SYSTEM>"):
        TAG_END = tokenizer.get("<TAG_END>")
        for i, t in enumerate(g):
            if t == TAG_END:
                return np.asarray(g[i + 1:], dtype=np.int64)
        return np.asarray([], dtype=np.int64)
    return np.asarray(g, dtype=np.int64)


def parse_gen_output(gen: Sequence[int], tokenizer) -> Dict[str, np.ndarray]:
    """gem の生成フィールド -> {楽器名: 系列}。CoT メタと末尾 <TE> を除去する。"""
    g = strip_cot_meta(gen, tokenizer)
    TE = tokenizer.get("<TE>")
    g = [int(t) for t in g if int(t) != TE]
    # 先頭マーカーが無い形式なので、ダミーのマーカーを付けて共通パーサに通す
    return parse_melody_block(list(g) + [tokenizer.get("<TAG_END>")], tokenizer)


# ----------------------------------------------------------------------
# 正規化: 統一 <CONST_M> への畳み込み
# ----------------------------------------------------------------------
def splice_time(parts: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    """時間軸の連結。None/空は飛ばす。小節境界(<SME>)はそのまま保たれる。"""
    keep = [np.asarray(p, dtype=np.int64) for p in parts if p is not None and len(p) > 0]
    return np.concatenate(keep) if keep else np.asarray([], dtype=np.int64)


def normalize_to_const(task: str, gen: Sequence[int], tokenizer,
                       past: Optional[Dict[str, np.ndarray]] = None,
                       future: Optional[Dict[str, np.ndarray]] = None,
                       other_inst: Optional[Dict[str, np.ndarray]] = None
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """任意タスクの生成物を統一 `<CONST_M> ... <TAG_END>` に畳む。

    Args:
        task:       TASKS のいずれか。
        gen:        gem の生成フィールド(<MGEN> の後ろ)。
        past/future:条件に使った PAST/FUTURE ブロックの {楽器: 系列}(該当タスクのみ)。
        other_inst: inst_comp の条件側 {他楽器: 系列}。

    Returns:
        (統一ブロック, {楽器: 時系列順に畳んだ系列})
    """
    gen_seqs = parse_gen_output(gen, tokenizer)

    if task == "inst_comp":
        # 楽器軸の結合: 同一窓なので時間方向には繋がず、レイヤーとして重ねる
        merged = dict(other_inst or {})
        merged.update(gen_seqs)
    else:
        merged = {}
        names = set(gen_seqs) | set(past or {}) | set(future or {})
        for n in names:
            merged[n] = splice_time([
                (past or {}).get(n) if task in ("meta_past", "infill") else None,
                gen_seqs.get(n),
                (future or {}).get(n) if task in ("meta_future", "infill") else None,
            ])

    merged = {k: v for k, v in merged.items() if len(v) > 0}
    return build_const_block(merged, tokenizer), merged


def build_const_block(seqs: Dict[str, np.ndarray], tokenizer) -> np.ndarray:
    """{楽器: 系列} -> `<CONST_M> [<INST_x> seq <ESEQ>]* <TAG_END>`。

    楽器順は名前でソートして決定的にする(分析側に順序の情報を漏らさないため)。
    """
    parts = [np.array([tokenizer.get("<CONST_M>")], dtype=np.int64)]
    for name in sorted(seqs):
        parts.append(np.concatenate([
            np.array([tokenizer.get(f"<INST_{name}>")], dtype=np.int64),
            np.asarray(seqs[name], dtype=np.int64),
            np.array([tokenizer.get("<ESEQ>")], dtype=np.int64),
        ]))
    parts.append(np.array([tokenizer.get("<TAG_END>")], dtype=np.int64))
    return np.concatenate(parts)


# ----------------------------------------------------------------------
# 構造の実測(報酬用): 小節数と楽器
# ----------------------------------------------------------------------
def measure_count_of(seq: Sequence[int], tokenizer) -> int:
    """系列に含まれる小節数を `<SME>` の個数から数える。"""
    SME = tokenizer.get("<SME>")
    return int(sum(1 for t in seq if int(t) == SME))


def structural_check(gen: Sequence[int], cond_meta: Sequence[int], tokenizer
                     ) -> Dict[str, object]:
    """生成物が「指定小節数」「指定楽器」を守っているかを **実測** で検査する。

    ana の推論に頼らず生成物そのものから測るので、判別器の性能に依存しない。

    Returns:
        {"measure_ok": bool, "inst_ok": bool,
         "gen_measures": int, "want_measures": Optional[int],
         "gen_inst": set, "want_inst": set}
    """
    from .carl_odd import inst_names_of

    gen_seqs = parse_gen_output(gen, tokenizer)
    want_inst = set(inst_names_of(cond_meta, tokenizer))
    gen_inst = set(gen_seqs)

    want_m = None
    for t in cond_meta:
        name = rev_safe(tokenizer, t)
        if isinstance(name, str) and name.startswith("<GEN_MEASURE_COUNT_"):
            want_m = int(name[len("<GEN_MEASURE_COUNT_"):-1])
            break

    # 小節数は楽器ごとに同じはずなので最大値で代表する(空楽器に引きずられないため)
    gen_m = max((measure_count_of(s, tokenizer) for s in gen_seqs.values()), default=0)
    return {
        "measure_ok": (want_m is None) or (gen_m == want_m),
        "inst_ok": (not want_inst) or (gen_inst == want_inst),
        "gen_measures": gen_m, "want_measures": want_m,
        "gen_inst": gen_inst, "want_inst": want_inst,
    }
