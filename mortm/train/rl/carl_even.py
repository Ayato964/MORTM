"""CARL 偶数ラウンド: MORTM-gem(生成器)を GRPO で強化学習する。

奇数ラウンドで作った **CoT 付き判別器**(MORTM-ana)を、そのまま報酬源として使う。
「MORTM に MORTM を評価させて成長させる」の実体がこのラウンドである。

1 ステップの流れ:
  1. データセットから条件 META(<SYSTEM>..<TAG_END>)を複数取り出す。
  2. MORTM-gem に 1 つの META あたり G 本を生成させる(GRPO のグループ)。
  3. MORTM-ana に CONST を見せ、META(<TE> まで)を **推論** させる。
  4. その系列をもう一度 MORTM-ana に入れ、PMA + OutHead で AI/Human を判定させる。

報酬(1 系列あたり):
  R = R_disc + R_meta + R_key - R_struct - R_sim
    R_disc  : sigmoid(-disc_logit)。Human 寄りなら 1 点に近づく(0 〜 +1.0)。
    R_meta  : ana の推論 META と条件 META の距離で採点。density/genre/inst の
              3 属性それぞれ 完全一致 +0.25 〜 最遠 -0.25(ゼロ交差は距離 0.5)。
              gmc(小節数)は R_struct が直接実測しているため採点対象外(観測のみ)。
    R_key   : 条件キーが ana の候補順位で 1 位なら +0.25、2 位で 0、
              3 位以降は順位に応じて -0.25 へ連続的に低下。
    R_struct: 生成物から実測した小節数/楽器の遵守度からの **減点のみ**(0 〜 -0.5)。
    R_sim   : 参照(その META の出どころの人間曲)に **似すぎたら減点**(0 〜 -0.5)。
              類似度 0.5 以下は 0。復元不可能なタスクなので加点側は置かない。
  → グループ内で標準化して GRPO のアドバンテージにする。
  → 上限 = disc 1.0 + key 0.25 + メタ3属性 0.75 = 2.00。

目的関数: クリップ付き方策勾配 + β * KL(π ‖ π_base)。π_base は MORTM-gem-base(凍結)。

終了条件(いずれかで次ラウンドへ):
  1. KL が閾値超え **かつ** 判別器報酬も高い → 報酬ハッキングとみなし停止し、
     直前の健全なスナップショットへ巻き戻す(その系列は奇数ラウンドの学習材料になる)。
  2. 平均報酬が reward_stop 超え。
  3. max_steps 到達。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ...utils.key_from_tokens import get_key_from_tokens, tokens_to_notes
from .carl_odd import density_token_of, generate_chunked, inst_names_of
from .task_norm import parse_gen_output, rev_safe


# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
@dataclass
class EvenRoundConfig:
    """偶数ラウンド(GRPO)のハイパーパラメータ。"""
    group_size: int = 16              # 1 つの META あたりのロールアウト本数 G。
                                      # ★ G=8 では優位性の統計量が荒すぎた:
                                      #   グループ標準偏差の相対誤差 26.7%(= グループ
                                      #   ごとに実効学習率が ±27% ばらつく)、
                                      #   グループ平均の SE 0.064 は「gem平均→人間平均」
                                      #   の系統差 0.213 の 30% に相当。G=16 で
                                      #   それぞれ 18.3% / 0.045 に下がる。
    prompts_per_step: int = 4         # 1 step で引くプロンプト本数。
                                      # ロールアウト総数 = prompts_per_step * group_size。
                                      # 以前は group_size//2 に固定されており、G を
                                      # 上げると生成コストが二乗で増えた。
    temperature: float = 1.0
    top_p: float = 0.9
    max_new: int = 1200
    max_measures: int = 8

    lr: float = 2e-6                  # 方策(gem)の学習率。RLHF/GRPO の常用域は 1e-6〜5e-6。
                                      # 5e-5 では 15 ステップで KL が 1.8 まで飛んだ実績あり。
    max_steps: int = 6000             # このラウンドの上限ステップ
    grad_clip: float = 1.0

    clip_eps: float = 0.2             # PPO/GRPO の比クリップ幅
    kl_coef: float = 0.05             # β: KL 正則化の重み。

    # 終了条件
    gen_chunk: int = 16               # 一度に生成するバッチ幅(KVキャッシュのVRAM制約)
    train_chunk: int = 8              # 逆伝播を割るバッチ幅。全語彙KLが (B,S,V) を
                                      # 複数持つため、畳み込みで系列長が伸びた環境では
                                      # 32本一括だとピーク10GB超で OOM する。

    kl_stop: float = 0.3              # 報酬ハッキング判定の KL 側の閾値(0.5 から厳格化)
    disc_stop: float = 0.8            # 同・判別器報酬側の閾値。KL と **両方** 超えて初めて
                                      # ハッキングとみなす。KL 単独では正当な改善と区別できない
                                      # (報酬もMETA一致率も伸びている局面で KL は普通に上がる)。
    stop_window: int = 20             # 停止判定に使う移動平均の窓(生値の単発スパイクで切らない)
    snapshot_every: int = 20          # 健全な重みを退避する間隔(ハッキング時の巻き戻し先)
    reward_stop: float = 0.625        # 平均報酬がこれを超えたら達成として停止(本物の人間の平均値)。
                                      # ★ v2 は加点を全廃したので上限が +1.00 になり、
                                      #   **0.5 = 「ana が平均して五分五分」** を意味する。
                                      #   v1 の 1.25 は上限 2.00 のうち約 0.50 が無条件の
                                      #   加点で構成されており、達成には「ana の META 推論が
                                      #   ほぼ完璧」かつ「判別器が半分騙されている」が同時に
                                      #   必要だった(8ラウンド中 0 回達成)。

    # 報酬の内訳(v2)
    #
    # ★ 設計方針: **駆動項は R_disc ただ一つ。他は全部ガードレール**。
    #
    #   v1 は density / inst / key に加点を置いていたため、全サンプルに無条件で
    #   約 +0.50 が入っていた(実測 density +0.227 / inst +0.245 / key +0.03)。
    #   GRPO はグループ内で報酬を標準化するので **定数項は勾配に一切寄与しない**。
    #   加点は「平均報酬を高く見せて目標 1.25 を近く錯覚させる」だけの働きしかせず、
    #   実際 8 ラウンドで報酬は 0.730 -> 0.643 と一度も改善しなかった。
    #
    #   v2 では加点を全廃し、違反時のみ減点する「ガードレール」に統一する。
    #   **ガードレールは人間の正解が必ず 0 点になるよう定義する**のが原則:
    #     density : 条件ラベルは const_seq から計算されたものなので、生成CONSTから
    #               同じ規則で測れば定義上一致する(人間の正解 2168/2168 = 100% 実測)。
    #     struct  : 小節数・楽器も同様に定義上一致する。
    #     key     : 条件キーは曲レベル(time=0, window_measures=999)なので CONST 区間の
    #               スケール外音比率は 0 にならない(人間実測: 中央 0.000 / 90% 0.251 /
    #               95% 0.411)。固定閾値だと人間まで減点されるので、**同一レコードの
    #               人間の正解を基準にした相対閾値**にする。
    #     sim     : 元から減点のみ。
    #
    #   これにより上限は R_disc の +1.00 になり、0.5 = 「ana が五分五分」という
    #   解釈可能な目盛りになる。
    empty_penalty: float = -1.0       # 音符を1つも生成しなかった場合の報酬(他項目を無視して確定)
    disc_weight: float = 1.0          # 判別器報酬の重み(唯一の駆動項)

    struct_penalty: float = 0.5       # 小節数/楽器の違反(ルールベース実測)
    density_penalty: float = 0.25     # density の違反(ルールベース・CONST区間・楽器別)
    density_tolerance: int = 1        # level 差がこれ以下なら違反ゼロ。
                                      # density は ceil(平均比率*10) の量子化なので、
                                      # 音符 1 個で level が飛ぶ。境界ノイズを吸収する。
    key_penalty: float = 0.25         # スケール外音が人間より過剰な分の減点
    key_tau_floor: float = 0.20       # 人間が完全にダイアトニックでも、これだけは許す
    key_margin: float = 0.10          # 人間の実測比率に上乗せする余裕

    sim_penalty: float = 0.5          # 参照(人間の原曲)に似すぎた場合の減点の上限。
    sim_center: float = 0.75          # sigmoid の中心(この類似度で減点が半分)
    sim_sharpness: float = 12.0       # sigmoid の鋭さ。大きいほど center 付近で切り立つ
    adv_eps: float = 1e-4

    # ana に META を推論させるか。既定 False。
    # ★ disc ロジットは `_pack_music_only` + 因果マスクにより **音楽ブロックだけの関数**
    #   なので、判別に META は要らない。v1 は R_meta / R_key のためだけに 32 本 x 最大
    #   128 トークンの greedy 生成を毎ステップ回していたが、density と key が
    #   ルールベースになった今、残るのは genre だけ(8ラウンドで 0.337 -> 0.345 と不動)。
    #   True にすると critique の戻り値に pred_metas / key_rankings が入るが、
    #   **v2 の compute_reward はそれらを採点しない**(診断用の観測窓として残してある)。
    #   採点に戻すなら compute_reward 側に項を足すこと。
    use_ana_meta: bool = False


# ----------------------------------------------------------------------
# META の分解(条件 META と ana の推論 META を突き合わせるため)
# ----------------------------------------------------------------------
def parse_meta(meta_ids: Sequence[int], tokenizer) -> Dict[str, object]:
    """`<SYSTEM>..<TAG_END>` 相当の並びから key / density / genre / 楽器を取り出す。

    Returns:
        {"key": Optional[int], "density": Tuple[int, ...], "genre": frozenset[int],
         "inst": Tuple[int, ...], "gmc": Optional[int]}
    """
    key_lo, key_hi = tokenizer.get_length_tuple("k")
    key: Optional[int] = None
    gmc: Optional[int] = None
    dens: List[int] = []
    genre: List[int] = []
    inst: List[int] = []
    for t in meta_ids:
        t = int(t)
        if key_lo <= t < key_hi:
            if key is None:
                key = t                                  # 最初の k_ を代表キーとする
            continue
        name = rev_safe(tokenizer, t)
        if not isinstance(name, str):
            continue
        if name.startswith("<NOTE_DENSE_"):
            dens.append(t)
        elif name.startswith("<GENRE_"):
            genre.append(t)
        elif name.startswith("<INST_"):
            inst.append(t)
        elif name.startswith("<GEN_MEASURE_COUNT_"):
            gmc = t
    return {"key": key, "density": tuple(dens), "genre": frozenset(genre),
            "inst": tuple(inst), "gmc": gmc}


def key_rank_reward(cond_key, key_ranking, penalty=0.25, bonus=0.25):
    """条件キーが ana の候補順位のどこにあるかで採点する。

        1 位    -> +bonus
        2 位    ->  0.0
        3 位以降 -> 0 から -penalty へ順位に比例して低下、最下位で -penalty
        圏外    -> -penalty

    完全一致(1位)は積極的に加点する。相対調(C major / A minor)のように構成音が
    同一で top-1 では原理的に当てられない対があるため、2 位は罰しない。
    3 位以降を連続にするのは、一律 -penalty だとグループ全員が外したときに
    分散ゼロ = 学習信号ゼロ になるため。「3 位」と「20 位」は区別する。
    """
    if cond_key is None:
        return 0.0, None
    ranking = list(key_ranking or [])
    if not ranking or cond_key not in ranking:
        return -penalty, None
    rank = ranking.index(cond_key) + 1
    if rank == 1:
        return bonus, 1
    if rank == 2:
        return 0.0, 2
    n = len(ranking)
    if n <= 3:
        return -penalty, rank
    frac = (rank - 2) / (n - 2)          # 3位で最小、最下位で 1.0
    return -penalty * frac, rank


def _dense_level(tid, tokenizer):
    """`<NOTE_DENSE_n>` -> n (1-10)。それ以外は None。"""
    name = rev_safe(tokenizer, tid)
    if isinstance(name, str) and name.startswith("<NOTE_DENSE_"):
        return int(name[len("<NOTE_DENSE_"):-1])
    return None


def _gmc_level(tid, tokenizer):
    """`<GEN_MEASURE_COUNT_k>` -> k。それ以外は None。"""
    name = rev_safe(tokenizer, tid)
    if isinstance(name, str) and name.startswith("<GEN_MEASURE_COUNT_"):
        return int(name[len("<GEN_MEASURE_COUNT_"):-1])
    return None


def _ordinal_dist(pred, target, span):
    """順序尺度の正規化距離 [0,1]。完全一致=0、最遠=1。"""
    if pred is None or target is None or span <= 0:
        return 1.0                      # 予測できていない = 最遠と同じ扱い
    return min(1.0, abs(pred - target) / span)


def _set_dist(pred, target):
    """集合の正規化距離 [0,1] = 1 - Jaccard。完全一致=0、共通ゼロ=1。"""
    p, t = set(pred or ()), set(target or ())
    if not p and not t:
        return 0.0
    u = len(p | t)
    return 1.0 - (len(p & t) / u) if u else 0.0


def _attr_score(d: float, bonus: float, penalty: float) -> float:
    """距離 d[0,1] -> 得点。d=0 で +bonus、d=1 で -penalty、その間は線形。

    順位で採点する key の「1位 +bonus / 2位 0 / 以降 -penalty へ連続」を、
    順位が定義できない属性(集合・順序尺度)へ距離で一般化したもの。
    ゼロ交差は d = bonus/(bonus+penalty)(既定値なら d=0.5)。
    """
    return bonus - (bonus + penalty) * min(1.0, max(0.0, d))


def meta_match_reward(cond_meta: Dict[str, object], pred_meta: Dict[str, object],
                      penalty: float = 0.25, tokenizer=None, bonus: float = 0.25
                      ) -> Tuple[float, Dict[str, float]]:
    """条件 META と ana の推論 META の **距離** に比例した減点を返す。

    従来は属性ごとに全か無か(-penalty か 0)だったが、GRPO はグループ内で優位性を
    標準化するため、**グループ8本が揃って外すと分散ゼロ = 学習信号ゼロ** になっていた
    (ログの density 0.00 はこの状態)。距離に応じた部分点にすると「惜しい」生成が
    区別され、グループ内に勾配が立つ。

    符号を正に振っても GRPO では定数シフトが打ち消されるだけで挙動は変わらないので、
    減点のまま粒度だけを上げる。

      density : 1-10 の順序尺度。|差|/9 に比例(楽器ごとに測り平均)
      gmc     : 小節数の順序尺度。|差|/条件小節数 に比例
      genre   : 集合。1 - Jaccard に比例
      inst    : 集合。1 - Jaccard に比例
      key     : ここでは扱わない(順位ベースの key_rank_reward が担当)

    Returns:
        (得点の合計, 属性ごとの一致度 [0,1] の dict)
    """
    scores: Dict[str, float] = {}
    r = 0.0

    if cond_meta.get("density"):
        if tokenizer is not None:
            ct = [_dense_level(t, tokenizer) for t in cond_meta["density"]]
            pt = [_dense_level(t, tokenizer) for t in (pred_meta.get("density") or ())]
            ds = [_ordinal_dist(pt[i] if i < len(pt) else None, ct[i], 9.0)
                  for i in range(len(ct))]
            d = sum(ds) / max(len(ct), 1)
        else:
            d = 0.0 if tuple(pred_meta.get("density") or ()) == tuple(cond_meta["density"]) else 1.0
        scores["density"] = 1.0 - d
        r += _attr_score(d, bonus, penalty)

    if cond_meta.get("gmc") is not None:
        # ★ gmc は **採点しない**(観測のみ)。
        #   小節数は R_struct が生成物の <SME> を数えて直接実測しており、そちらは
        #   全ステップで遵守度 1.00 = gem は既に完璧である。にもかかわらず ana の
        #   推論一致率は 0.49-0.66 しかない。つまりこの差は gem の失敗ではなく
        #   **ana が小節数を数えられていない**ことの反映で、gem には改善余地が無い。
        #   採点に入れると (a) 取り除けない固定ハンデになり、(b) グループ 8 本が全て
        #   正しい小節数なのに ana の推論だけがばらつくため、その分散が純粋なノイズ
        #   として advantage に混入する。R_struct との二重計上でもある。
        if tokenizer is not None:
            c = _gmc_level(cond_meta["gmc"], tokenizer)
            p = _gmc_level(pred_meta.get("gmc"), tokenizer)
            d = _ordinal_dist(p, c, float(max(c or 1, 1)))
        else:
            d = 0.0 if pred_meta.get("gmc") == cond_meta["gmc"] else 1.0
        scores["gmc"] = 1.0 - d

    if cond_meta.get("genre"):
        d = _set_dist(pred_meta.get("genre"), cond_meta["genre"])
        scores["genre"] = 1.0 - d
        r += _attr_score(d, bonus, penalty)

    if cond_meta.get("inst"):
        d = _set_dist(pred_meta.get("inst"), cond_meta["inst"])
        scores["inst"] = 1.0 - d
        r += _attr_score(d, bonus, penalty)

    return r, scores


# ----------------------------------------------------------------------
# ピアノロール類似度
# ----------------------------------------------------------------------
def structural_reward(gen, cond_meta, tokenizer) -> Tuple[float, Dict[str, float]]:
    """生成物から **実測** した構造遵守度 [0,1] を返す。

    ana の推論を介さないので、判別器を騙しても誤魔化せない。
    小節数は相対誤差、楽器は Jaccard で段階評価する(全か無かだと勾配が消えるため)。
    """
    from .task_norm import structural_check, parse_gen_output

    st = structural_check(gen, cond_meta, tokenizer)
    want_m, got_m = st["want_measures"], st["gen_measures"]
    if want_m:
        m_score = max(0.0, 1.0 - abs(got_m - want_m) / float(max(want_m, 1)))
    else:
        m_score = 1.0
    wi, gi = st["want_inst"], st["gen_inst"]
    if wi:
        u = len(wi | gi)
        i_score = (len(wi & gi) / u) if u else 0.0
    else:
        i_score = 1.0
    return (m_score + i_score) / 2.0, {"measure": m_score, "inst": i_score}


# ----------------------------------------------------------------------
# ルールベースのガードレール(density / key)
# ----------------------------------------------------------------------
_PITCH_CLASS = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
                'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
                'A#': 10, 'Bb': 10, 'B': 11}
_MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10, 11)   # 自然的短音階 + 導音(第7音上げ)


def key_pitch_classes(key_token, tokenizer) -> Optional[frozenset]:
    """`k_CM` 等のキートークン -> ダイアトニックな音class集合。

    短調は導音(第7音上げ)を含める。含めないと、調性内の主要な進行である
    V-i の導音が全部「調外」に数えられてしまう。
    """
    if key_token is None:
        return None
    name = rev_safe(tokenizer, int(key_token))
    if not isinstance(name, str) or not name.startswith("k_"):
        return None
    body = name[2:]
    if len(body) < 2:
        return None
    mode, tonic = body[-1], body[:-1]
    root = _PITCH_CLASS.get(tonic)
    if root is None or mode not in ("M", "m"):
        return None                       # k_Unknown 等
    steps = _MAJOR_STEPS if mode == "M" else _MINOR_STEPS
    return frozenset((root + i) % 12 for i in steps)


def out_of_key_ratio(notes, pcs: Optional[frozenset]) -> Optional[float]:
    """音符列のうち、指定キーのスケールから外れた音の割合 [0,1]。

    キー推定(`get_key_from_tokens`)ではなく **照合** にするのが要点:
      - 1 小節でも測れる(数小節から調は決まらないので推定は当てにならない)
      - 相対調の曖昧さが消える(C major と A minor は同じ音class集合)
      - music21 を呼ばないので実質ゼロコスト(推定は 1 本 97ms かかる)
    """
    if pcs is None or not notes:
        return None
    out = sum(1 for n in notes if (int(n.pitch) % 12) not in pcs)
    return out / len(notes)


def key_violation(gen_notes, ref_notes, pcs, tau_floor: float = 0.20,
                  margin: float = 0.10) -> Tuple[float, Optional[float]]:
    """スケール外音が「同じ窓で人間が書いた量」を超えた分だけを違反とする [0,1]。

    条件キーは曲レベル(time=0)の値なので、CONST 窓が転調区間に当たると人間の
    正解でもスケール外音が大量に出る(実測: 90 パーセンタイルで 0.251、
    95 パーセンタイルで 0.411)。固定閾値にすると人間の正解まで減点されるため、
    **同一レコードの人間の正解を基準**にする。

        tau = max(tau_floor, 人間の比率) + margin
        violation = clip((生成の比率 - tau) / (1 - tau), 0, 1)

    人間の正解を入れれば生成比率 == 人間比率 なので必ず violation = 0 になる。
    """
    g = out_of_key_ratio(gen_notes, pcs)
    if g is None:
        return 0.0, None
    r = out_of_key_ratio(ref_notes, pcs)
    tau = max(tau_floor, r if r is not None else 0.0) + margin
    if tau >= 1.0:
        return 0.0, g
    return float(min(1.0, max(0.0, (g - tau) / (1.0 - tau)))), g


def density_violation(gen, cond_meta_parsed, cond_inst_names, tokenizer,
                      tolerance: int = 1) -> Tuple[float, Optional[bool]]:
    """生成 CONST から楽器ごとに density を実測し、条件ラベルとの差を違反度にする。

    条件の `<NOTE_DENSE_n>` は `convert_foundation.py` が **const_seq のみ** から
    `_calculate_density_token` で作ったものなので、生成 CONST に同じ規則を当てれば
    直接比較できる(人間の正解で 2168/2168 = 100% 一致を実測済み)。

    ana の推論を介さないので、判別器を騙しても誤魔化せない。

    Returns: (違反度 [0,1], 完全一致だったか(測れなければ None))
    """
    levels_cond = [_dense_level(t, tokenizer) for t in (cond_meta_parsed.get("density") or ())]
    if not levels_cond or not cond_inst_names:
        return 0.0, None
    gen_seqs = parse_gen_output(gen, tokenizer)

    viol, ok, n = [], 0, 0
    span = max(1.0, 9.0 - tolerance)
    for name, want in zip(cond_inst_names, levels_cond):
        if want is None:
            continue
        seq = gen_seqs.get(name)
        if seq is None or len(seq) == 0:
            continue                      # 楽器の欠落は R_struct の担当。二重計上しない
        try:
            got_tok = density_token_of(np.asarray(seq, dtype=np.int64), name, tokenizer)
        except Exception:
            got_tok = None
        n += 1
        if got_tok is None:               # 規則上限を超えた = 明確な違反
            viol.append(1.0)
            continue
        got = _dense_level(got_tok, tokenizer)
        if got is None:
            viol.append(1.0)
            continue
        d = abs(got - want)
        if d <= tolerance:
            ok += 1
        viol.append(min(1.0, max(0.0, (d - tolerance) / span)))
    if not viol:
        return 0.0, None
    return float(np.mean(viol)), (ok == n)


def _has_note(seq: Sequence[int], tokenizer) -> bool:
    """生成物に shift(`s_`) トークンが1つでもあるか = 音符があるか。"""
    lo, hi = tokenizer.get_length_tuple("s")
    return any(lo <= int(t) < hi for t in seq)


def to_piano_roll(tokenizer, seq: Sequence[int], fps: float = 16.0,
                  tempo: int = 120) -> np.ndarray:
    """トークン列 -> (128, T) の 2 次元バイナリ・ピアノロール。"""
    return notes_to_roll(tokens_to_notes(tokenizer, seq, tempo), fps)


def notes_to_roll(notes, fps: float = 16.0) -> np.ndarray:
    """音符リスト -> (128, T) のバイナリ・ピアノロール。

    `tokens_to_notes` は安くないので、キー照合と類似度で同じ音符列を使い回せるよう
    トークン列からの変換と分離してある。
    """
    if not notes:
        return np.zeros((128, 1), dtype=bool)
    end = max(n.end for n in notes)
    T = max(1, int(np.ceil(end * fps)))
    roll = np.zeros((128, T), dtype=bool)
    for n in notes:
        p = int(n.pitch)
        if not (0 <= p < 128):
            continue
        s = max(0, int(np.floor(n.start * fps)))
        e = min(T, max(s + 1, int(np.ceil(n.end * fps))))
        roll[p, s:e] = True
    return roll


def roll_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """2 つのバイナリロールの IoU(Jaccard)。長さは短い方に揃える。"""
    T = min(a.shape[1], b.shape[1])
    if T == 0:
        return 0.0
    a, b = a[:, :T], b[:, :T]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _smoothstep(x: float, lo: float, hi: float) -> float:
    """lo で 0、hi で 1 になる滑らかな遷移(C1 連続)。"""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def similarity_reward(sim: float, weight: float = 0.5, center: float = 0.75,
                      sharpness: float = 12.0) -> float:
    """参照(人間の原曲)との類似度 -> **ペナルティ**。0 〜 -weight の連続値。

        sim <= 0.5 : 0        (独自に書けている = 咎めない)
        sim  > 0.5 : 尖らせた sigmoid で連続的に -weight へ

    帯域式(0.5/0/1.0 の3段)をやめた理由:
      参照は「その META の出どころになった、まさにその人間の曲」なので、
      キー・密度・ジャンル・楽器の4項目だけから復元するのは原理的に不可能。
      実測で 302 ステップ中 298 が最下帯に張り付き、R_sim は定数 +0.5 になっていた。
      GRPO の advantage はグループ内で正規化されるため、全員同値の定数は
      打ち消されて **勾配に一切寄与しない**(学習信号ゼロ)。
      そこで「近すぎたら罰する」側にだけ意味を持たせ、上振れ余地を捨てる。

    sigmoid は [0.5, 1.0] を [0, 1] へ写すよう両端で正規化するので、
    sim=0.5 でちょうど 0、sim=1.0 でちょうど -weight になり不連続が出ない。
    sharpness を上げるほど center 付近で切り立つ(既定 12 は center±0.1 で
    おおよそ 0.15 / 0.85 を通る S 字)。
    """
    if sim <= 0.5:
        return 0.0
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    lo, hi = sig(sharpness * (0.5 - center)), sig(sharpness * (1.0 - center))
    frac = (sig(sharpness * (min(1.0, sim) - center)) - lo) / (hi - lo)
    return float(-weight * frac)


def discriminator_reward(disc_logit: float, span: float = 8.0) -> float:
    """判別器ロジット -> 報酬 [0,1]。OutHead は AI=1 / Human=0 なので符号を反転する。

    sigmoid をやめてロジットに線形にしている。判別器が勝っている動作点
    (実測 z≈1.7〜3.1)では sigmoid の傾きが 0.13 まで潰れ、生成物の違いが
    報酬差として出てこないため(GAN の飽和問題)。線形なら同じ範囲でも
    グループ内の差がそのまま残る。

    z=-span/2 で 1.0、z=0 で 0.5、z=+span/2 で 0.0。範囲は従来と同じ [0,1] なので
    reward_stop などの閾値の意味は変わらない。
    """
    return float(min(1.0, max(0.0, 0.5 - disc_logit / span)))


# ----------------------------------------------------------------------
# 系列の対数尤度(GRPO の材料)
# ----------------------------------------------------------------------
def sequence_logprobs_full(model, x: Tensor, padding_mask: Tensor):
    """(B,S) -> 全語彙の対数確率 (B,S-1,V)。厳密KLの計算に使う。"""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward(x, padding_mask=padding_mask, is_causal=True,
                               is_save_cache=False)
    return torch.log_softmax(logits[:, :-1, :].float(), dim=-1)


def exact_kl(logp_new_full: Tensor, logp_base_full: Tensor) -> Tensor:
    """厳密な per-token KL(pi_new || pi_base) を全語彙にわたって計算する (B,S-1)。

    k3 推定量(`exp(d)-d-1`)は方策からのサンプル 1 点で近似するため分散が大きく、
    発散を抑える clamp が **勾配まで消して** しまう(clamp 範囲外は grad=0)。
    その結果、最も強く引き戻すべき「深く離れたトークン」に復元力が働かず、
    KL が 0.5 -> 3.2 と暴走した。

    語彙が 700 しかないので全語彙の総和を直接取れる。推定誤差ゼロ、clamp 不要、
    そして **どれだけ離れても勾配が残る**。
    """
    p_new = logp_new_full.exp()
    return (p_new * (logp_new_full - logp_base_full)).sum(dim=-1)


def sequence_logprobs(model, x: Tensor, padding_mask: Tensor,
                      response_mask: Tensor) -> Tensor:
    """(B,S) の入力に対し、応答区間の各トークンの対数尤度 (B,S-1) を返す。

    response_mask: (B,S) 生成トークン(=学習対象)の位置が True。
    戻り値は response_mask[:,1:] の外を 0 で埋めた行列。
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward(x, padding_mask=padding_mask, is_causal=True,
                               is_save_cache=False)
    logits = logits[:, :-1, :].float()
    tgt = x[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # (B,S-1)
    return tok_logp * response_mask[:, 1:].float()


def grpo_advantages(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    """グループ内標準化: A_i = (r_i - mean_g) / (std_g + eps)。

    rewards: (N,) で N = グループ数 * group_size、同一グループが連続して並ぶ前提。
    価値関数を持たないのが GRPO の要点なので、Critic は使わない。
    """
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)


# ----------------------------------------------------------------------
# 目的関数
# ----------------------------------------------------------------------
class GRPOLoss(nn.Module):
    """クリップ付き方策勾配 + KL(π ‖ π_base)。transformers 非依存の自前実装。

        L = -E[ min(ρ·A, clip(ρ, 1±ε)·A) ] + β · KL_k3
        ρ      = exp(logp_new - logp_old)
        KL_k3  = exp(Δ) - Δ - 1,  Δ = logp_base - logp_new   (非負・低分散な推定量)

    平均は「系列内でトークン平均 → 系列間で平均」の順に取る(長さのバイアスを避ける)。
    """

    def __init__(self, clip_eps: float = 0.2, kl_coef: float = 0.04):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef

    def forward(self, logp_new: Tensor, logp_old: Tensor, logp_base: Tensor,
                advantages: Tensor, mask: Tensor, kl_exact: Optional[Tensor] = None):
        """logp_*: (B,S-1)、advantages: (B,)、mask: (B,S-1) の応答マスク。

        kl_exact を渡した場合はそれを KL として使う(全語彙の厳密値)。
        渡さない場合のみ従来の k3 推定量にフォールバックする。
        """
        mask = mask.float()
        n = mask.sum(dim=1).clamp(min=1.0)

        ratio = torch.exp(logp_new - logp_old)
        adv = advantages.unsqueeze(1)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        pg = -torch.min(unclipped, clipped)

        if kl_exact is not None:
            kl = kl_exact                       # 全語彙の厳密KL(clamp不要・勾配が消えない)
        else:
            delta = torch.clamp(logp_base - logp_new, min=-10.0, max=10.0)
            kl = torch.exp(delta) - delta - 1.0

        pg_seq = (pg * mask).sum(dim=1) / n
        kl_seq = (kl * mask).sum(dim=1) / n
        loss = (pg_seq + self.kl_coef * kl_seq).mean()
        return loss, pg_seq.mean().detach(), kl_seq.mean().detach()


# ----------------------------------------------------------------------
# ロールアウト〜報酬(MORTM が MORTM を評価する本体)
# ----------------------------------------------------------------------
class EvenRound:
    """MORTM-gem を GRPO で回すための、ロールアウトと報酬計算の実装。

    Args:
        gem:      学習対象の生成器(Meta2CONST)。
        ana:      奇数ラウンドで学習した AnalysisMORTM(凍結して評価にのみ使う)。
        base:     MORTM-gem-base(凍結。KL の参照方策)。
        tokenizer / device / cfg
    """

    def __init__(self, gem, ana, base, tokenizer, device, cfg: EvenRoundConfig):
        self.gem, self.ana, self.base = gem, ana, base
        self.tok, self.dev, self.cfg = tokenizer, device, cfg
        self.loss_fn = GRPOLoss(cfg.clip_eps, cfg.kl_coef)
        self._TE = tokenizer.get("<TE>")
        self._META = tokenizer.get("<META>")
        self._CONST_M = tokenizer.get("<CONST_M>")
        self._TAG_END = tokenizer.get("<TAG_END>")
        self._MGEN = tokenizer.get("<MGEN>")
        self._EOS = tokenizer.get("<EOS>")

    # -- 1) データセットの META から生成プロンプトを作る --------------
    # -- 2) GRPO 的生成: 1 レコードあたり G 本 -------------------------
    @torch.no_grad()
    def rollout(self, recs: List[dict], seed: Optional[int] = None) -> Dict:
        """生成SFTレコードごとに group_size 本ずつ一括生成する。

        `recs` は `gen_sft_pool.load_gen_sft_pool` の要素。プロンプトは
        SFT 学習時のものをそのまま使うので、meta / meta_past / meta_future /
        infill / inst_comp のどのタスクでも同じ経路で扱える。

        Returns: {"prompts", "gens", "cond", "cond_raw", "recs", "refs", "group_size"}
            recs: 各ロールアウトに対応する元レコード(ana へ渡す前の畳み込みに使う)
            refs: 人間が書いた正解 CONST(R_sim の参照)
        """
        G = self.cfg.group_size
        prompts, conds, raws, rr, refs = [], [], [], [], []
        for rec in recs:
            p = [int(t) for t in rec["prompt"]]
            cond = parse_meta(rec["cond_meta"], self.tok)
            for _ in range(G):
                prompts.append(p)
                conds.append(cond)
                raws.append(np.asarray(rec["cond_meta"]))   # structural_check 用の生META
                rr.append(rec)
                refs.append(rec["gt"])
        gens = generate_chunked(
            self.gem, prompts, self.tok, self.dev, self.cfg.max_measures,
            chunk=self.cfg.gen_chunk,
            p=self.cfg.top_p, temperature=self.cfg.temperature,
            max_new=self.cfg.max_new, seed=seed)
        return {"prompts": prompts, "gens": gens, "cond": conds, "cond_raw": raws,
                "recs": rr, "refs": refs, "group_size": G}

    # -- 3+4) ana に META を推論させ、その系列で AI/Human を判定させる --
    def to_const_block(self, gen: Sequence[int], rec: Optional[dict] = None) -> List[int]:
        """gem の出力を **統一 CONST** に畳んで分析入力にする。

        `rec` があれば条件側の人間ブロック(PAST/FUTURE/他楽器)と時系列順に連結し、
        最大 24 小節の `<CONST_M> [<INST_x> 系列 <ESEQ>]* <TAG_END>` にする。
        判別器につなぎ目(人間の PAST と AI の CONST の境界)を必ず見せるための処理で、
        Human サンプル側も `gen_sft_pool.fold_to_const` の同じ経路を通るため、
        畳み方の違い自体が AI/Human の手がかりになることはない。
        """
        if rec is not None:
            from .gen_sft_pool import fold_to_const
            const, _ = fold_to_const(rec, gen, self.tok)
            return [int(t) for t in const]
        body = [int(t) for t in gen if int(t) != self._TE]
        return [self._CONST_M] + body + [self._TAG_END]

    @torch.no_grad()
    def critique(self, rollouts: Dict) -> Dict:
        """MORTM-ana に生成物を読ませる。

        既定(`use_ana_meta=False`)では **判別ロジットだけ** を取る forward 1 回で済む。

        ★ なぜ META 推論を回さなくてよいか:
          `forward_with_disc(meta_id=<META>)` は `_pack_music_only` で `<META>` の手前
          までを切り出してプーリングし、デコーダは因果マスクなので、その位置は
          `<META>` 以降のトークンを一切見ない。つまり **disc ロジットは音楽ブロック
          だけの関数**である。v1 は R_meta / R_key のためだけに 32 本 x 最大 128
          トークンの greedy 生成を毎ステップ回していたが、density と key が
          ルールベースになった今、それで得られるのは genre だけ。

        Returns: {"disc": (N,), "pred_metas": list|None, "key_rankings": list|None}
        """
        recs = rollouts.get("recs") or [None] * len(rollouts["gens"])
        analysis_prompts = [
            [self._EOS] + self.to_const_block(g, r) + [self._META]
            for g, r in zip(rollouts["gens"], recs)
        ]

        if not getattr(self.cfg, "use_ana_meta", False):
            x, pad = self._pad_batch(analysis_prompts)
            _, disc = self.ana.forward_with_disc(x, pad, is_causal=True, meta_id=self._META)
            return {"disc": disc.float().cpu(), "pred_metas": None, "key_rankings": None}

        # --- 以下は従来経路(use_ana_meta=True のときだけ) ---
        # パス1: META の推論(<TE> で停止)。greedy で推論させる。サンプリングだと
        # 同一の生成物でも評価のたびに META が変わり報酬が乱数化する。
        pred_tokens = generate_chunked(
            self.ana, analysis_prompts, self.tok, self.dev,
            10 ** 6, chunk=self.cfg.gen_chunk,
            p=1.0, temperature=0.0, max_new=128, seed=None)
        pred_metas = [parse_meta(pt, self.tok) for pt in pred_tokens]

        # パス2: <TE> まで含む完成系列を入れ直し、判別ヘッドを通す
        full = [ap + [int(t) for t in pt] + [self._TE]
                for ap, pt in zip(analysis_prompts, pred_tokens)]
        x, pad = self._pad_batch(full)
        logits, disc = self.ana.forward_with_disc(
            x, pad, is_causal=True, meta_id=self._META)

        klo, khi = self.tok.get_length_tuple("k")
        rankings = []
        for i, seq in enumerate(full):
            pos = next((j for j, t in enumerate(seq) if klo <= int(t) < khi), None)
            if pos is None or pos == 0 or pos - 1 >= logits.size(1):
                rankings.append([])
                continue
            row = logits[i, pos - 1, klo:khi]
            order = torch.argsort(row, descending=True).cpu().numpy()
            rankings.append([int(klo + o) for o in order])
        return {"disc": disc.float().cpu(), "pred_metas": pred_metas,
                "key_rankings": rankings}

    # -- 報酬 ---------------------------------------------------------
    def compute_reward(self, rollouts: Dict, critique_out: Dict
                       ) -> Tuple[Tensor, Dict[str, float]]:
        """R = R_disc - (density + key + 構造 + 類似度 の違反)。

        **駆動項は R_disc ただ一つ。他は全部ガードレール**(違反したときだけ減点)。
        v1 のように density/inst/key へ加点を置くと、全サンプルに同じ定数が乗る。
        GRPO はグループ内で報酬を標準化するので、定数は勾配に一切寄与せず、
        平均報酬を高く見せて目標を近く錯覚させるだけになる。

        ガードレールは **人間の正解が必ず 0 点** になるよう定義してある:
          density : 条件ラベルは const_seq から作られたものなので定義上一致する
          struct  : 小節数・楽器も同様
          key     : 同一レコードの人間の正解を基準にした相対閾値
          sim     : 自分自身との比較なので人間は満額の減点になるが、これは
                    「人間の正解を丸写しするな」という項なので意図どおり

        範囲: -1.0(空生成) 〜 +1.0(ana を完全に騙し、違反ゼロ)
        """
        cfg = self.cfg
        disc = critique_out["disc"]
        rs = []
        parts = {"disc": 0.0, "density": 0.0, "key": 0.0, "struct": 0.0, "sim": 0.0}
        sims, oo_keys = [], []
        n_empty = 0
        struct_m, struct_i = [], []
        dens_ok_n = dens_n = 0
        refs = rollouts.get("refs")

        for i, gen in enumerate(rollouts["gens"]):
            # 音符が1つも無い生成は問答無用で最低評価。空の CONST は違反ゼロに
            # なってしまうため、他項目を無視して確定させる。
            if not _has_note(gen, self.tok):
                rs.append(cfg.empty_penalty)
                n_empty += 1
                struct_m.append(0.0); struct_i.append(0.0)
                continue

            cond = rollouts["cond"][i]
            cond_raw = rollouts["cond_raw"][i]
            ref = refs[i] if refs is not None else None

            # --- 駆動項: 判別器 ---
            r_disc = cfg.disc_weight * discriminator_reward(float(disc[i]))

            # --- ガードレール1: 構造(小節数・楽器) ---
            s_score, sd = structural_reward(gen, cond_raw, self.tok)
            p_struct = -cfg.struct_penalty * (1.0 - s_score)
            struct_m.append(sd["measure"]); struct_i.append(sd["inst"])

            # --- ガードレール2: density(CONST区間・楽器別・ルールベース) ---
            v_dens, all_ok = density_violation(
                gen, cond, inst_names_of(cond_raw, self.tok), self.tok,
                tolerance=cfg.density_tolerance)
            p_dens = -cfg.density_penalty * v_dens
            if all_ok is not None:
                dens_n += 1
                dens_ok_n += int(all_ok)

            # --- 音符列は key と sim で共用する(tokens_to_notes は安くない) ---
            gen_notes = tokens_to_notes(self.tok, gen)
            ref_notes = tokens_to_notes(self.tok, ref) if ref is not None else None

            # --- ガードレール3: キー(スケール外音の過剰分) ---
            pcs = key_pitch_classes(cond.get("key"), self.tok)
            v_key, oo = key_violation(gen_notes, ref_notes, pcs,
                                      cfg.key_tau_floor, cfg.key_margin)
            p_key = -cfg.key_penalty * v_key
            if oo is not None:
                oo_keys.append(oo)

            # --- ガードレール4: 人間の正解に似すぎ ---
            if ref_notes is not None:
                sim = roll_similarity(notes_to_roll(gen_notes), notes_to_roll(ref_notes))
                sims.append(sim)
                p_sim = similarity_reward(sim, cfg.sim_penalty,
                                          cfg.sim_center, cfg.sim_sharpness)
            else:
                p_sim = 0.0

            parts["disc"] += r_disc; parts["density"] += p_dens
            parts["key"] += p_key; parts["struct"] += p_struct; parts["sim"] += p_sim
            rs.append(r_disc + p_dens + p_key + p_struct + p_sim)

        n = max(1, len(rs))
        stats = {k: v / n for k, v in parts.items()}
        stats["sim_mean"] = float(np.mean(sims)) if sims else float("nan")
        stats["oo_key_mean"] = float(np.mean(oo_keys)) if oo_keys else float("nan")
        stats["empty_rate"] = n_empty / n
        stats["struct_measure"] = float(np.mean(struct_m)) if struct_m else float("nan")
        stats["struct_inst"] = float(np.mean(struct_i)) if struct_i else float("nan")
        stats["density_ok"] = (dens_ok_n / dens_n) if dens_n else float("nan")
        return torch.tensor(rs, dtype=torch.float32), stats

    # -- 学習ステップ用のバッチ組み立て --------------------------------
    def build_train_batch(self, rollouts: Dict):
        """`prompt + gen` を連結し、応答区間だけ True のマスク付きでパディングする。"""
        seqs, resp = [], []
        for p, g in zip(rollouts["prompts"], rollouts["gens"]):
            s = list(p) + [int(t) for t in g]
            seqs.append(s)
            resp.append([False] * len(p) + [True] * len(g))
        x, pad = self._pad_batch(seqs)
        m = torch.zeros_like(x, dtype=torch.bool)
        for i, r in enumerate(resp):
            m[i, :len(r)] = torch.tensor(r, device=x.device)
        return x, pad, m

    def _pad_batch(self, seqs: List[List[int]]) -> Tuple[Tensor, Tensor]:
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, dtype=torch.long, device=self.dev)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.dev)
        return out, (out != 0)

    def step_backward(self, rollouts: Dict, rewards: Tensor):
        """1 更新分を **マイクロバッチに割って** 前向き+逆伝播する。

        全語彙の厳密 KL は (B, S, V) のテンソルを何本も同時に持つため、
        畳み込みで系列長が伸びた(最大 1552 トークン)いま、32 本一括だと
        ピーク VRAM が 10GB を超えて OOM する(実測: 系列長 842 で 7.8GB)。
        優位性はグループ内で標準化済みなので **バッチを割っても値は変わらない**。
        各チャンクの損失を本数比で重み付けして即 backward し、勾配だけ貯める。

        併せてチャンクごとにその中の最大長までしかパディングしないので、
        長短が混在するバッチでの無駄も消える。

        Returns: (loss, pg, kl) いずれも float。勾配は呼び出し前に zero_grad 済み前提。
        """
        adv_all = grpo_advantages(rewards.to(self.dev), rollouts["group_size"],
                                  self.cfg.adv_eps)
        n_total = len(rollouts["gens"])
        cs = max(1, int(getattr(self.cfg, "train_chunk", 8)))
        acc = {"loss": 0.0, "pg": 0.0, "kl": 0.0}
        for i in range(0, n_total, cs):
            sl = slice(i, min(i + cs, n_total))
            sub = {"prompts": rollouts["prompts"][sl], "gens": rollouts["gens"][sl]}
            x, pad, mask = self.build_train_batch(sub)
            w = (sl.stop - sl.start) / n_total          # 本数比。全体平均と一致させる
            loss, pg, kl = self._chunk_loss(x, pad, mask, adv_all[sl])
            (loss * w).backward()
            acc["loss"] += float(loss.detach()) * w
            acc["pg"] += float(pg) * w
            acc["kl"] += float(kl) * w
            del loss, x, pad, mask
        return acc["loss"], acc["pg"], acc["kl"]

    def _chunk_loss(self, x, pad, mask, adv):
        """マイクロバッチ 1 個ぶんの損失。step_backward からのみ呼ぶ。"""
        with torch.no_grad():
            logp_base_full = sequence_logprobs_full(self.base, x, pad)
        logp_new_full = sequence_logprobs_full(self.gem, x, pad)
        tgt = x[:, 1:]
        m = mask[:, 1:]
        logp_new = logp_new_full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * m.float()
        logp_old = logp_new.detach()            # 単一ステップGRPOなので ratio ≡ 1
        kl_exact = exact_kl(logp_new_full, logp_base_full)
        return self.loss_fn(logp_new, logp_old, None, adv, m, kl_exact=kl_exact)

    def step_loss(self, rollouts: Dict, rewards: Tensor):
        """一括版(検証用)。本番は step_backward を使うこと。"""
        x, pad, mask = self.build_train_batch(rollouts)
        adv = grpo_advantages(rewards.to(self.dev), rollouts["group_size"],
                              self.cfg.adv_eps)
        with torch.no_grad():
            logp_base_full = sequence_logprobs_full(self.base, x, pad)
        # 方策側は1回のフォワードで全語彙を取り、そこから
        # (a) サンプルトークンの対数尤度(方策勾配用) と (b) 厳密KL を作る。
        logp_new_full = sequence_logprobs_full(self.gem, x, pad)
        tgt = x[:, 1:]
        m = mask[:, 1:]
        logp_new = logp_new_full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * m.float()
        logp_old = logp_new.detach()            # 単一ステップGRPOなので ratio ≡ 1
        kl_exact = exact_kl(logp_new_full, logp_base_full)
        return self.loss_fn(logp_new, logp_old, None, adv, m, kl_exact=kl_exact)

    # -- 終了判定 -----------------------------------------------------
    def should_stop(self, mean_reward: float, mean_kl: float,
                    mean_disc: float = 0.0) -> Optional[str]:
        """次ラウンドへ進むべきかを返す(進むなら理由文字列、続行なら None)。

        引数はいずれも **移動平均**(既定 20 ステップ)を渡すこと。KL は毎ステップ別の
        ロールアウトで測るため生値は激しく揺れ、単発スパイクで打ち切ると健全な学習まで
        止まってしまう。

        報酬ハッキングの判定は「KL が高い **かつ** 判別器報酬が高い」。
        つまり *判別器を騙すために* 基準方策から離れた場合だけを指す。
        KL だけ高いのは単に方策が動いただけで、正当な改善でも普通に起こる。
        """
        if mean_kl > self.cfg.kl_stop and mean_disc > self.cfg.disc_stop:
            return (f"reward_hacking: KL {mean_kl:.3f} > {self.cfg.kl_stop} かつ "
                    f"判別器報酬 {mean_disc:.3f} > {self.cfg.disc_stop}")
        if mean_reward > self.cfg.reward_stop:
            return f"achieved: reward {mean_reward:.3f} > {self.cfg.reward_stop}"
        return None
