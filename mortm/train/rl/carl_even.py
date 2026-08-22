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
from .carl_odd import density_token_of, generate_chunked
from .task_norm import rev_safe


# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
@dataclass
class EvenRoundConfig:
    """偶数ラウンド(GRPO)のハイパーパラメータ。"""
    group_size: int = 8               # 1 つの META あたりのロールアウト本数 G
    temperature: float = 1.0
    top_p: float = 0.9
    max_new: int = 1200
    max_measures: int = 8

    lr: float = 2e-6                  # 方策(gem)の学習率。RLHF/GRPO の常用域は 1e-6〜5e-6。
                                      # 5e-5 では 15 ステップで KL が 1.8 まで飛んだ実績あり。
    max_steps: int = 6000             # このラウンドの上限ステップ
    grad_clip: float = 1.0

    clip_eps: float = 0.2             # PPO/GRPO の比クリップ幅
    kl_coef: float = 0.3              # β: KL 正則化の重み。0.1 では優位性(大きさ~1)に負けるため強化

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
    reward_stop: float = 1.25         # 平均報酬がこれを超えたら達成として停止

    # 報酬の内訳
    #
    # ★ 設計方針: **判別器を報酬の主要な伸びしろにする**。
    #   META/キーは完全一致でのみ加点し、外せば減点する対称な採点にする。
    #   ただし META の加点は ana の推論能力で頭打ちになる(genre 0.31)ため、
    #   実際に残る伸びしろは R_disc 側に偏る。
    #   構造と類似度は加点を置かず減点だけにして、抜け道を塞ぐ。
    empty_penalty: float = -1.0       # 音符を1つも生成しなかった場合の報酬(他項目を無視して確定)
    disc_weight: float = 1.0          # 判別器報酬の重み(報酬の主役)
    struct_penalty: float = 0.5       # 構造遵守(小節数/楽器)の **減点** の重み。
                                      # 生成物から直接実測するので判別器を騙しても
                                      # 誤魔化せない。遵守度 score に対し -w*(1-score)。
    meta_bonus: float = 0.25          # density / genre / inst が完全一致したときの加点
    meta_penalty: float = 0.25        # 同・最も外したときの減点(ゼロ交差は距離 0.5)
    key_bonus: float = 0.25           # キーが 1 位で当たったときの加点(2 位は 0)
    key_penalty: float = 0.25         # キーが 3 位以降のときの減点(順位に応じて連続)
    sim_penalty: float = 0.5          # 参照(人間の原曲)に似すぎた場合の減点の上限。
    sim_center: float = 0.75          # sigmoid の中心(この類似度で減点が半分)
    sim_sharpness: float = 12.0       # sigmoid の鋭さ。大きいほど center 付近で切り立つ
    adv_eps: float = 1e-4


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


def _has_note(seq: Sequence[int], tokenizer) -> bool:
    """生成物に shift(`s_`) トークンが1つでもあるか = 音符があるか。"""
    lo, hi = tokenizer.get_length_tuple("s")
    return any(lo <= int(t) < hi for t in seq)


def to_piano_roll(tokenizer, seq: Sequence[int], fps: float = 16.0,
                  tempo: int = 120) -> np.ndarray:
    """トークン列 -> (128, T) の 2 次元バイナリ・ピアノロール。

    `tokens_to_notes` で秒単位の音符に戻し、fps で量子化して鳴っている枠を 1 にする。
    """
    notes = tokens_to_notes(tokenizer, seq, tempo)
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
    def critique(self, rollouts: Dict) -> Tuple[List[Dict], Tensor]:
        """MORTM-ana に (a) META 推論 と (b) AI/Human 判定 をさせる。

        手順は仕様どおり 2 パス:
          パス1: `<EOS> <CONST_M>..<TAG_END> <META>` から <TE> まで自己回帰で META を生成。
          パス2: 生成し切った系列(<TE> 込み)を丸ごと入れ直し、PMA+OutHead で判定。

        Returns: (推論 META の parse 結果, 判別ロジット (N,))
        """
        recs = rollouts.get("recs") or [None] * len(rollouts["gens"])
        analysis_prompts = [
            [self._EOS] + self.to_const_block(g, r) + [self._META]
            for g, r in zip(rollouts["gens"], recs)
        ]
        # パス1: META の推論(<TE> で停止)
        # ★ greedy で推論させる。サンプリングだと同一の生成物でも評価のたびに
        #   META が変わり、報酬が ±0.2 揺れる(実測: 優位性の分散の 46% が
        #   この乱数だった)。GRPO はグループ内の報酬差を優位性にするので、
        #   その差が乱数だと勾配が毎ステップ別方向を向いて打ち消し合う。
        pred_tokens = generate_chunked(
            self.ana, analysis_prompts, self.tok, self.dev,
            10 ** 6,                                   # 小節では止めない
            chunk=self.cfg.gen_chunk,
            p=1.0, temperature=0.0,
            max_new=128, seed=None)
        pred_metas = [parse_meta(pt, self.tok) for pt in pred_tokens]

        # パス2: <TE> まで含む完成系列を入れ直し、判別ヘッドを通す
        full = [ap + [int(t) for t in pt] + [self._TE]
                for ap, pt in zip(analysis_prompts, pred_tokens)]
        x, pad = self._pad_batch(full)
        # 奇数ラウンドと同じく、判別は音楽ブロックのみで行う
        logits, disc = self.ana.forward_with_disc(
            x, pad, is_causal=True, meta_id=self._META)

        # キー候補の順位: ana が実際にキーを吐いた位置の直前(=そこを予測する位置)の
        # ロジットを見て、キー語彙(k_*)だけを確率順に並べる。
        # 「ana が条件キーをどれだけ上位に置いたか」を測るので、相対調のように
        # 構成音が同一で 1 位を取り切れない対でも 2 位で拾える。
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
        return pred_metas, disc.float().cpu(), rankings

    # -- 報酬 ---------------------------------------------------------
    def compute_reward(self, rollouts: Dict, pred_metas: List[Dict], disc: Tensor,
                       references: Optional[List[Sequence[int]]] = None,
                       key_rankings: Optional[List[List[int]]] = None
                       ) -> Tuple[Tensor, Dict[str, float]]:
        """R = R_meta + R_disc + R_sim を系列ごとに合算する。

        references: 各ロールアウトに対応する人間の CONST(ピアノロール比較の参照)。
                    None のときは R_sim を 0 とする。
        """
        cfg = self.cfg
        rs, parts = [], {"meta": 0.0, "disc": 0.0, "sim": 0.0, "struct": 0.0}
        hit_sum, hit_n, sims, key_ranks = {}, {}, [], []
        n_empty = 0
        struct_m, struct_i = [], []
        for i, gen in enumerate(rollouts["gens"]):
            # 音符が1つも無い生成は問答無用で最低評価にする。
            # 空の CONST は類似度 0 -> similarity_reward(0)=+0.5 が満額入るため、
            # 「何も書かない」が部分的に報われる抜け道になっていた。
            if not _has_note(gen, self.tok):
                rs.append(cfg.empty_penalty)
                n_empty += 1
                # 構造の統計にも 0 として計上する。除外すると「空生成が増えるほど
                # struct_measure が 1.00 に近づく」という逆立ちした表示になり、
                # ログ上は満点なのに実際は何も書けていない状況を見逃す。
                struct_m.append(0.0); struct_i.append(0.0)
                continue
            r_meta, hits = meta_match_reward(rollouts["cond"][i], pred_metas[i],
                                             cfg.meta_penalty, tokenizer=self.tok,
                                             bonus=cfg.meta_bonus)
            # key は順位ベース(1位 +bonus / 2位 0 / 以降 -penalty へ連続)
            rk = (key_rankings[i] if key_rankings and i < len(key_rankings) else [])
            r_key, rank = key_rank_reward(rollouts["cond"][i].get("key"), rk,
                                          cfg.key_penalty, cfg.key_bonus)
            r_meta += r_key
            if rank is not None:
                key_ranks.append(rank)
            for k, v in hits.items():          # v は [0,1] の一致度(部分点)
                hit_sum[k] = hit_sum.get(k, 0.0) + float(v)
                hit_n[k] = hit_n.get(k, 0) + 1
            r_disc = cfg.disc_weight * discriminator_reward(float(disc[i]))
            # 構造は加点しない。遵守度 score から外れた分だけ引く(0 〜 -struct_penalty)。
            s_score, sd = structural_reward(gen, rollouts["cond_raw"][i], self.tok)
            r_struct = -cfg.struct_penalty * (1.0 - s_score)
            struct_m.append(sd["measure"]); struct_i.append(sd["inst"])
            if references is not None and references[i] is not None:
                sim = roll_similarity(to_piano_roll(self.tok, gen),
                                      to_piano_roll(self.tok, references[i]))
                sims.append(sim)
                r_sim = similarity_reward(sim, cfg.sim_penalty,
                                          cfg.sim_center, cfg.sim_sharpness)
            else:
                r_sim = 0.0
            parts["meta"] += r_meta; parts["disc"] += r_disc
            parts["sim"] += r_sim; parts["struct"] += r_struct
            rs.append(r_meta + r_disc + r_sim + r_struct)
        n = max(1, len(rs))
        stats = {k: v / n for k, v in parts.items()}
        # 属性別の一致率(どの指示を守れていないかを見るため)
        stats["hit"] = {k: hit_sum[k] / hit_n[k] for k in hit_sum}
        stats["sim_mean"] = float(np.mean(sims)) if sims else float("nan")
        stats["empty_rate"] = n_empty / n
        stats["struct_measure"] = float(np.mean(struct_m)) if struct_m else float("nan")
        stats["struct_inst"] = float(np.mean(struct_i)) if struct_i else float("nan")
        stats["key_top1"] = float(np.mean([r == 1 for r in key_ranks])) if key_ranks else float("nan")
        stats["key_top3"] = float(np.mean([r <= 3 for r in key_ranks])) if key_ranks else float("nan")
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
