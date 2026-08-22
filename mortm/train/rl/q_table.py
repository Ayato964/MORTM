"""Q-Table: 判別器(MORTM-ana)のための経験リプレイ。CARL の中核機構。

なぜ要るか:
  履歴なしの敵対的学習では、判別器は **常に最新の生成器しか見ない**。すると循環が起きる:
  生成器が「判別器が忘れた領域」へドリフト -> 高報酬 -> 判別器が再学習 -> 生成器が元へ戻る。
  過去に AI として棄却した例を定期的に混ぜ直すことで、この往復を断ち切る。
  結果として判別器は「直近の報酬ハッキング検出器」ではなく、AI 由来性そのものの検出器になる。

  強化学習の replay と違い、**ラベルが腐らない**のが本機構の安全な点である。
  古い gem 系列は何ラウンド経っても "AI" のままで、正解が変質しない。

混合比(奇数ラウンドの 1 バッチ):
      人間 1/2  :  最新 gem 1/4  :  Q-Table 1/4
  AI 側の合計は 1/4 + 1/4 = 1/2 なので、BCE のクラス均衡は保たれる。

更新(奇数ラウンド終了時に一括):
  既存 q_max 本のうちランダムに 1/4 を忘却し、そのラウンドで貯めた new_table の
  1/4 をランダムサンプリングしてマージする。両者が満杯(既定 3000)なら
  3000 - 750 + 750 = 3000 で定常になる。

  この一様忘却により、バッファは「経過ラウンド数に対し (1-forget_ratio)^age で
  減衰する混合分布」へ収束する。既定 1/4 なら実効的な記憶地平は 3〜4 ラウンド程度
  (4 ラウンド後の生存率 ≒ 32%、8 ラウンド後 ≒ 10%)。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ラベル(OutHead は AI=1 / Human=0)
HUMAN, GEM, QTAB = "human", "gem", "qtable"


class QTable:
    """過去ラウンドの MORTM-gem 出力を保持する固定容量バッファ。

    Args:
        q_max:        保持する最大系列数。
        forget_ratio: ラウンド終了時に忘却する既存分の割合。
        merge_ratio:  ラウンド終了時に取り込む新規分の割合。
        rng:          np.random.Generator(再現性のため外から渡す)。
    """

    def __init__(self, q_max: int = 3000, forget_ratio: float = 0.25,
                 merge_ratio: float = 0.25, rng: Optional[np.random.Generator] = None):
        self.q_max = q_max
        self.forget_ratio = forget_ratio
        self.merge_ratio = merge_ratio
        self.rng = rng if rng is not None else np.random.default_rng()
        self.table: List[dict] = []            # 確定済み(過去ラウンド)
        self.new: List[dict] = []              # 今ラウンドの蓄積(未マージ)
        self.round = 0

    def __len__(self) -> int:
        return len(self.table)

    # -- 蓄積 ---------------------------------------------------------
    def add(self, items: Sequence[dict]) -> int:
        """今ラウンドの gem 出力を貯める。q_max に達したらそれ以上は捨てる。

        items は `{"seq": 音楽ブロック, "meta": 答え側META}` の辞書。
        META を一緒に持つのは、Q-Table から引くたびにキー推定(約97ms/本)を
        やり直すと学習が止まるため。生成時に一度だけ計算して以後は使い回す。
        """
        room = self.q_max - len(self.new)
        if room <= 0:
            return 0
        take = list(items)[:room]
        self.new.extend(take)
        return len(take)

    # -- 参照 ---------------------------------------------------------
    def sample(self, n: int) -> List[dict]:
        """確定済みテーブルから非復元でランダムサンプリング。足りなければある分だけ。"""
        if not self.table or n <= 0:
            return []
        k = min(n, len(self.table))
        idx = self.rng.choice(len(self.table), size=k, replace=False)
        return [self.table[i] for i in idx]

    # -- 更新(奇数ラウンド終了時に一括) -------------------------------
    def commit(self) -> Dict[str, int]:
        """1/4 を忘却し、新規の 1/4 をマージする。統計を返す。

        初回(テーブルが空)は忘却する対象が無いので、新規から q_max まで一気に満たす。
        ラウンドが早期終了して new が少ない場合、マージ数もその分少なくなり、
        テーブルは一時的に縮む(戻り値の `size` で観測できる)。
        """
        stats = {"before": len(self.table), "forgot": 0, "merged": 0,
                 "new_pool": len(self.new)}
        if not self.table:
            keep = self.new[:self.q_max]                      # 初回は総取り
            self.table = list(keep)
            stats["merged"] = len(keep)
        else:
            n_forget = int(len(self.table) * self.forget_ratio)
            if n_forget > 0:
                keep_idx = self.rng.choice(len(self.table),
                                           size=len(self.table) - n_forget, replace=False)
                self.table = [self.table[i] for i in sorted(keep_idx)]
                stats["forgot"] = n_forget
            n_merge = min(int(len(self.new) * self.merge_ratio),
                          self.q_max - len(self.table))
            if n_merge > 0:
                idx = self.rng.choice(len(self.new), size=n_merge, replace=False)
                self.table.extend(self.new[i] for i in idx)
                stats["merged"] = n_merge
        self.new = []
        self.round += 1
        stats["size"] = len(self.table)
        return stats


# ----------------------------------------------------------------------
# バッチ混合
# ----------------------------------------------------------------------
def mix_batch(human: Sequence[dict], gem: Sequence[dict],
              qtable: Optional[QTable], batch_size: int,
              rng: Optional[np.random.Generator] = None
              ) -> Tuple[List[dict], np.ndarray, List[str]]:
    """人間 1/2 : 最新 gem 1/4 : Q-Table 1/4 で 1 バッチを組む。

    各プールの要素は `{"seq": 音楽ブロック, "meta": 答え側META}` の辞書。

    Q-Table が空のとき(初回ラウンド)は、その 1/4 枠を最新 gem で埋める。
    したがって初回は従来どおり 人間 1/2 : gem 1/2 になり、AI 側の比率は常に 1/2。

    Returns:
        items:  サンプル辞書のリスト
        is_ai:  (N,) float32。人間=0 / gem=1 / qtable=1
        source: 各要素の出所ラベル("human"/"gem"/"qtable")。
                **停止判定を出所別に取るために必要**(下記 accuracy_by_source を参照)。
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_human = batch_size // 2
    n_q = batch_size // 4
    n_gem = batch_size - n_human - n_q

    q_items = qtable.sample(n_q) if qtable is not None else []
    if len(q_items) < n_q:                      # Q-Table 不足分は gem で補填
        n_gem += (n_q - len(q_items))

    def _pick(pool, n):
        if n <= 0 or len(pool) == 0:
            return []
        idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
        return [pool[i] for i in np.atleast_1d(idx)]

    h = _pick(human, n_human)
    g = _pick(gem, n_gem)

    items = list(h) + list(g) + list(q_items)
    is_ai = np.array([0.0] * len(h) + [1.0] * (len(g) + len(q_items)), dtype=np.float32)
    source = [HUMAN] * len(h) + [GEM] * len(g) + [QTAB] * len(q_items)

    perm = rng.permutation(len(items))          # 出所が順番で漏れないようシャッフル
    return ([items[i] for i in perm], is_ai[perm], [source[i] for i in perm])


# ----------------------------------------------------------------------
# 停止判定(出所別)
# ----------------------------------------------------------------------
def accuracy_by_source(disc_logit, is_ai, source) -> Dict[str, float]:
    """判別精度を出所別に出す。`overall` と `human`/`gem`/`qtable` を返す。"""
    import torch
    p = (torch.as_tensor(disc_logit).float() > 0).cpu().numpy().astype(np.float32)
    y = np.asarray(is_ai, dtype=np.float32)
    correct = (p == y).astype(np.float32)
    out = {"overall": float(correct.mean()) if len(correct) else 0.0}
    src = np.asarray(source)
    for s in (HUMAN, GEM, QTAB):
        m = (src == s)
        if m.any():
            out[s] = float(correct[m].mean())
    return out


def should_stop_odd(acc: Dict[str, float], threshold: float = 0.85) -> Optional[str]:
    """奇数ラウンドの終了判定。**最新 gem 分画の精度**で判断する。

    Q-Table の系列は「過去の弱い gem」なので検出が容易であり、これを含めた全体精度で
    判定すると、最新 gem を捕まえられないまま 90% に到達しうる。その判別器を偶数ラウンドへ
    渡すと報酬 R_disc が機能しないため、判定は現行 gem に対する精度に限定する。
    (人間側の精度も併せて見ないと「全部 AI と答える」退化を見逃すので同時に要求する。)
    """
    gem_acc = acc.get(GEM)
    hum_acc = acc.get(HUMAN)
    if gem_acc is None or hum_acc is None:
        return None
    if gem_acc >= threshold and hum_acc >= threshold:
        return (f"discriminator ready: gem {gem_acc:.3f} / human {hum_acc:.3f} "
                f">= {threshold} (overall {acc.get('overall', float('nan')):.3f})")
    return None
