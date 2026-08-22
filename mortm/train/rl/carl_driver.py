"""CARL ドライバ: 奇数(判別器の学習)と偶数(生成器の GRPO)を交互に回す。

    ラウンド1(奇)  MORTM-ana を学習。gem 出力を Q-Table に蓄積し、終了時に commit。
    ラウンド2(偶)  MORTM-gem を GRPO で学習。KL 超過(報酬ハッキング)か報酬到達で終了。
    ラウンド3(奇)  Q-Table から過去の AI 例を 1/4 混ぜて再学習 -> 以降くり返し。

報酬ハッキングした系列がそのまま次の奇数ラウンドの学習材料になるので、
「ハッキング -> 検出 -> 対策」の閉ループが自然に生まれる。
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch

from . import carl_paths as P
from .carl_even import EvenRound, EvenRoundConfig
from .carl_odd import AnalysisMORTM, OddRound, OddRoundConfig, split_analysis_sample
from .q_table import QTable


# `_save` がアンマージ状態で書いたことを示す目印。state_dict に載せるが、
# モデルのパラメータではないので読み込み時に必ず取り除く。
_UNMERGED_MARK = "_carl_unmerged"


def load_gem(config: str, ckpt: str, device, train: bool = False):
    """生成器(または KL 参照)をロードする。"""
    from ...models.mortm import MORTM
    from ...models.modules.config import MORTMArgs
    from ..train import _DefaultLearningProgress

    prog = _DefaultLearningProgress()
    try:
        prog.set_device(device)
    except Exception:
        pass
    m = MORTM(MORTMArgs(config), prog).to(device)
    sd = torch.load(ckpt, map_location=device)
    # ★ CARL が保存した gem は `eval()`(マージ済み)中の state_dict なので、weight に
    #   既に B@A*scaling が焼き込まれている。このまま読んで eval() すると **2回目の
    #   マージ**が走り、デルタが 2 倍になった別モデルになる(実測: 小節数遵守 100%→17.7%、
    #   空生成 0%→74%)。マージ済みかどうかは lora_B が全ゼロかで判別する
    #   (`fix_merged_ckpt` を通した修復版は B=0 なので二重マージが起きない)。
    if _is_legacy_merged(ckpt, sd):
        raise RuntimeError(
            f"マージ済みで保存された旧形式のチェックポイントです: {ckpt}\n"
            f"`python -m mortm.train.rl.fix_merged_ckpt {ckpt}` で修復してから使ってください。")
    sd.pop(_UNMERGED_MARK, None)
    m.load_state_dict(sd)
    if train:
        m.train()
    else:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return m


def unfreeze_attention_only(model) -> tuple:
    """全パラメータを凍結し、**Attention の本体重みだけ**を解放する。

    KV 連想記憶の見立てに基づく設計。FFN が音楽的知識(key-value メモリ)を保持して
    いるなら、それを書き換える必要はない。書き換えるべきは「何を引くか」を決める
    Attention 側であり、問題を生成品質の改善から **知識探索の改善** に読み替える。

    解放するのは `self_attention.qkv_block.{qkv_weight,W_o}.weight` のみ。
    LoRA(`lora_A`/`lora_B`)は Attention のものも含めて凍結する。
    なお gem は eval() で回すため loralib がマージ状態になり、LoRA は計算グラフから
    そもそも外れる。requires_grad=False は二重の担保。

    Returns:
        (解放したパラメータ数, 全パラメータ数)
    """
    n_train = n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        is_attn = ".self_attention." in name and name.endswith(".weight")
        is_lora = name.endswith(("lora_A", "lora_B"))
        p.requires_grad = bool(is_attn and not is_lora)
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total


def _is_legacy_merged(ckpt: str, sd: dict) -> bool:
    """`_save` 修正前に書かれたマージ済みファイルを検出する。

    マージ済みかを重みだけから一般に見分ける方法は無い(元の base が要る)ので、
    「CARL の出力ディレクトリにあり、かつ lora_B が非ゼロ」を条件にする。
    修正後の `_save` は train() 状態で保存するので lora_B が非ゼロでも安全だが、
    そちらは `.pth` にマーカーを入れて区別する(下記 `_UNMERGED_MARK`)。
    """
    if _UNMERGED_MARK in sd:
        return False
    if os.path.dirname(os.path.abspath(ckpt)) != os.path.abspath(P.OUT_DIR):
        return False
    return any(k.endswith("lora_B") and float(v.abs().sum()) > 0 for k, v in sd.items())


def load_ana(config: str, ckpt: str, device):
    """分析器をロードする。判別ヘッド(PMA/OutHead)は新規なので strict=False。"""
    from ...models.modules.config import MORTMArgs
    from ..train import _DefaultLearningProgress

    prog = _DefaultLearningProgress()
    try:
        prog.set_device(device)
    except Exception:
        pass
    m = AnalysisMORTM(MORTMArgs(config), prog).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd.pop(_UNMERGED_MARK, None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    new = [k for k in missing if k.startswith(("pma", "out_head"))]
    other = [k for k in missing if not k.startswith(("pma", "out_head"))]
    print(f"[CARL] ana ロード: 新規ヘッド {len(new)} / 欠損 {len(other)} / 余剰 {len(unexpected)}")
    if other or unexpected:
        raise RuntimeError(f"ana の重みが噛み合っていない: missing={other[:5]} unexpected={unexpected[:5]}")
    return m


def blank_measure_ratio(music_block, tokenizer) -> float:
    """音楽ブロックの **合計小節数** に占める空小節(BLANK 小節)の割合。

    空小節 = その小節に shift(`s_`) が 1 つも無い小節。`_build_melody_block` が
    「S トークンが1つもない(BLANK のみ)」を空と扱うのと同じ判定にする。
    複数楽器あるときは全楽器の小節を合算して数える(= 合計小節数)。
    """
    from ...utils.convert import split_sequence_measure
    from .task_norm import parse_melody_block

    s_lo, s_hi = tokenizer.get_length_tuple("s")
    SME = tokenizer.get("<SME>")
    total = blank = 0
    for seq in parse_melody_block(music_block, tokenizer).values():
        for m in split_sequence_measure(np.asarray(seq), 1, SME):
            m = np.asarray(m)
            total += 1
            if not np.any((m >= s_lo) & (m < s_hi)):
                blank += 1
    return (blank / total) if total else 1.0


def load_human_pool(manifest: str, tokenizer, limit: int = 20000,
                    max_blank_ratio: float = 0.5) -> List[dict]:
    """分析 SFT 形式の npz から人間サンプルを読み、(音楽ブロック, META) に割る。

    max_blank_ratio: 合計小節数の **半分以上** が空小節なら捨てる(既定 0.5)。
        スカスカな正解を判別器に見せると「空小節が多い = 人間」という
        音楽的に無意味な手がかりを学んでしまい、AI/Human の判別信号が濁る。
    """
    paths = json.load(open(manifest))
    pool, dropped = [], 0
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                i = 1
                while f"array{i}" in d.files:
                    music, meta = split_analysis_sample(np.asarray(d[f"array{i}"]), tokenizer)
                    i += 1
                    if music is None or len(music) <= 4 or len(meta) <= 2:
                        continue
                    if blank_measure_ratio(music, tokenizer) >= max_blank_ratio:
                        dropped += 1
                        continue
                    pool.append({"seq": music, "meta": meta})
        except Exception:
            continue
        if len(pool) >= limit:
            break
    if dropped:
        print(f"[CARL] 人間サンプル: 空小節が半数以上のため {dropped} 件を除外 "
              f"(閾値 {max_blank_ratio})")
    return pool[:limit]


def _init_wandb(enabled: bool, run_name: Optional[str], config: dict):
    """wandb を初期化する。未認証や wandb 未導入なら黙って無効化する。"""
    if not enabled:
        return None
    try:
        import wandb
        wandb.init(project="MORTM-CARL", name=run_name, config=config,
                   dir=os.path.dirname(P.OUT_DIR))
        return wandb
    except Exception as e:
        print(f"[CARL] wandb 無効化: {e}")
        return None


class CARLDriver:
    """奇数/偶数ラウンドを交互に回す最上位ループ。"""

    def __init__(self, device=None, q_max: int = 3000, seed: int = 42,
                 odd_cfg: Optional[OddRoundConfig] = None,
                 even_cfg: Optional[EvenRoundConfig] = None,
                 max_blank_ratio: float = 0.5, keep_round_ckpts: bool = False,
                 use_wandb: bool = True, run_name: Optional[str] = None,
                 ana_resume: Optional[str] = None, q_resume: Optional[str] = None,
                 pool_limit: int = 40000):
        from ..tokenizer import Tokenizer, get_token_converter_pro, TO_MUSIC

        self.dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
        self.tok.mode(TO_MUSIC)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        print("[CARL] モデルをロードしています...")
        self.gem = load_gem(P.GEM_CONFIG, P.resolve(P.GEM_CKPT), self.dev, train=True)
        self.base = load_gem(P.BASE_CONFIG, P.resolve(P.BASE_CKPT), self.dev, train=False)
        self.ana = load_ana(P.ANA_CONFIG, P.resolve(P.ANA_INIT_CKPT), self.dev)
        if ana_resume:
            # 奇数ラウンドで学習済みの LoRA + PMA/OutHead を上乗せする(backbone は上で復元済み)。
            if q_resume and os.path.exists(q_resume):
                q = torch.load(q_resume, weights_only=False)
                self.qtable.table = q["table"]
                self.qtable.round = q.get("round", 0)
                print(f"[CARL] Q-Table 復元: {len(self.qtable)} 本 "
                      f"(ラウンド {self.qtable.round} 時点)")
            extra = torch.load(P.resolve(ana_resume), map_location=self.dev)
            extra.pop(_UNMERGED_MARK, None)
            miss, unexp = self.ana.load_state_dict(extra, strict=False)
            if unexp:
                raise RuntimeError(f"ana 再開ckptに未知のキー: {unexp[:5]}")
            print(f"[CARL] ana を再開: {ana_resume} ({len(extra)} テンソル復元)")

        self.qtable = QTable(q_max=q_max, rng=self.rng)
        self.odd_cfg = odd_cfg or OddRoundConfig()
        self.even_cfg = even_cfg or EvenRoundConfig()
        self.keep_round_ckpts = keep_round_ckpts
        # ★ 生成SFTデータ(MORTM4.5D-160M-SFT-gen の学習データ)を両フェーズで共有する。
        #   奇数は「人間の PAST++CONST++FUTURE を畳んだ最大24小節の CONST」を Human 例に、
        #   偶数は同じレコードのプロンプトで gem に生成させ、同じ畳み方で ana に渡す。
        #   条件分布が2フェーズ間で一致するので、判別器が条件の違いで楽をできない。
        from .gen_sft_pool import load_gen_sft_pool
        pool = load_gen_sft_pool(P.resolve(P.GEN_SFT_MANIFEST), self.tok,
                                 limit=pool_limit, max_blank_ratio=max_blank_ratio,
                                 rng=self.rng)
        # ★ レコード単位で分割する。共有すると ana は「gem が偶数ラウンドで解く
        #   のと同じプロンプトの人間正解」を教師として暗記してしまい、判別が
        #   「AI らしさの検出」ではなく「暗記した正解との一致判定」に化ける。
        #   gem は暗記された正解と一字一句同じものを書けないので原理的に騙せず、
        #   判別器の 0.85 到達が 2380->340->100 step と加速する一方、判別器報酬は
        #   0.15 前後でトレンドを持たない、という観測になっていた。
        cut = len(pool) // 2
        self.human_ana = pool[:cut]          # 奇数ラウンド: 判別器の教師データ
        self.human_gem = pool[cut:]          # 偶数ラウンド: GRPO のプロンプト
        self.human = self.human_ana          # 後方互換(既存コードの参照先)
        print(f"[CARL] データ分割: ana用 {len(self.human_ana)} 件 / "
              f"gem用 {len(self.human_gem)} 件 (レコード単位で排他) / device={self.dev}")
        os.makedirs(P.OUT_DIR, exist_ok=True)
        self.history = []
        self.step_global = 0                 # ラウンドを跨いで単調増加する x 軸
        self.wandb = _init_wandb(use_wandb, run_name, {
            "q_max": q_max, "seed": seed, "max_blank_ratio": max_blank_ratio,
            "odd": vars(self.odd_cfg), "even": vars(self.even_cfg),
            "gem_ckpt": P.GEM_CKPT, "n_ana": len(self.human_ana),
            "n_gem": len(self.human_gem),
        })

    def _log(self, d: dict):
        """wandb へ 1 点記録する(無効なら何もしない)。"""
        self.step_global += 1
        if self.wandb is not None:
            self.wandb.log(d, step=self.step_global)

    # -- 奇数: 判別器を鍛える -----------------------------------------
    def run_odd(self, rnd: int):
        print(f"\n{'='*72}\n[CARL] ラウンド {rnd} (奇数): MORTM-ana(分析器 + AI/人間判別器)を学習\n"
              f"  目的: gem の出力を「AI が書いた」と見抜けるようにする。\n"
              f"        判別正解率が gem・人間の両方で {self.odd_cfg.acc_threshold:.2f} を超えたら次へ。\n{'='*72}")
        for p in self.gem.parameters():
            p.requires_grad = False
        self.gem.eval()
        r = OddRound(self.gem, self.ana, self.tok, self.dev, self.qtable,
                     self.odd_cfg, rng=self.rng)
        r.round_no = rnd
        out = r.run(self.human_ana, on_step=lambda d: self._log({**d, "round": rnd}))
        self._save("ana", rnd)
        return out

    # -- 偶数: 生成器を GRPO で鍛える ---------------------------------
    def ensure_qtable(self):
        """次の奇数ラウンドまでに Q-Table が空でないことを保証する。"""
        if len(self.qtable) == 0:
            self.prefill_qtable()

    def run_even(self, rnd: int, steps_per_check: int = 10):
        print(f"\n{'='*72}\n[CARL] ラウンド {rnd} (偶数): MORTM-gem(生成器)を GRPO で学習\n"
              f"  目的: 直前ラウンドで鍛えた判別器を「人間が書いた」と誤認させる。\n"
              f"        平均報酬が {self.even_cfg.reward_stop:.2f} を超えるか、{self.even_cfg.max_steps} step で次へ。\n{'='*72}")
        # ★ 充填は **凍結解除より前** に済ませること。prefill_qtable は内部で
        #   OddRound を作り、その __init__ が gem を全凍結するので、後に置くと
        #   unfreeze_attention_only の結果が消えて optimizer が空になる。
        #   この時点の gem は「1つ前の奇数ラウンドの gem」なので、ここで充填すれば
        #   本来 Q-Table に入るはずだった分布を忠実に再構成できる。
        self.ensure_qtable()

        self.ana.eval()
        for p in self.ana.parameters():
            p.requires_grad = False
        n_train, n_total = unfreeze_attention_only(self.gem)
        print(f"[EvenRound] 学習対象: Attention のみ {n_train/1e6:.1f}M / 全 {n_total/1e6:.1f}M "
              f"({100*n_train/n_total:.1f}%) — FFN・埋め込み・LoRA は凍結")
        # ★ eval() で回すこと。理由は 2 つある。
        #   1. dropout を切る。train() だと logp_new だけにノイズが乗り、凍結 base(eval)
        #      との差が KL に化けて、重みが動く前から KL が 3 を超えてしまう
        #      (ratio = exp(logp_new - logp_old) も同一入力で 1 にならず勾配が壊れる)。
        #      config の dropout=0.2 は attention.py が self.training を見るようになった
        #      ので、eval() なら実際に 0 が渡る。
        #   2. loralib がマージ状態になり LoRA が計算グラフから外れる = Attention の
        #      LoRA も含めて完全に凍結される(unfreeze_attention_only の意図と一致)。
        #   requires_grad は eval() でも生きているので Attention 本体の学習はできる。
        self.gem.eval()

        er = EvenRound(self.gem, self.ana, self.base, self.tok, self.dev, self.even_cfg)
        opt = torch.optim.AdamW([p for p in self.gem.parameters() if p.requires_grad],
                                lr=self.even_cfg.lr)
        print(f"[EvenRound] lr={self.even_cfg.lr} / group_size={self.even_cfg.group_size} "
              f"/ max_steps={self.even_cfg.max_steps} / kl_stop={self.even_cfg.kl_stop} "
              f"/ reward_stop={self.even_cfg.reward_stop}")
        from collections import deque
        W = self.even_cfg.stop_window
        mv = {k: deque(maxlen=W) for k in ("reward", "kl", "disc")}
        # 健全な重みのスナップショット。ハッキング検出時はここへ巻き戻す。
        # 「汚染系列を次の奇数ラウンドの教材にする」意図は Q-Table 側で保持されるので、
        # **生成器の重みまで汚染したまま次ラウンドへ持ち越す必要はない**。
        # 実際、巻き戻さずに進めた結果 KL が 0.5->3.2 と累積劣化し、
        # ラウンド14で生成能力が完全に失われた(gem pool 0/64, 音符なし)。
        snap = {k: v.detach().clone() for k, v in self.gem.state_dict().items()}
        snap_step = 0
        SNAP_EVERY = getattr(self.even_cfg, "snapshot_every", 20)

        reason, step = None, 0
        while reason is None and step < self.even_cfg_max_steps():
            step += 1
            idx = self.rng.choice(len(self.human_gem),
                                  size=max(1, self.even_cfg.group_size // 2), replace=False)
            recs = [self.human_gem[i] for i in np.atleast_1d(idx)]

            roll = er.rollout(recs)
            refs = roll["refs"]        # 人間が書いた正解 CONST(同一プロンプトの真値)
            pred_metas, disc, key_rank = er.critique(roll)
            rewards, parts = er.compute_reward(roll, pred_metas, disc, references=refs,
                                              key_rankings=key_rank)
            # マイクロバッチに割って逆伝播まで済ませる(勾配は貯まった状態で返る)
            loss, pg, kl = er.step_backward(roll, rewards)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.gem.parameters() if p.requires_grad],
                self.even_cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

            mr, mkl = float(rewards.mean()), float(kl)
            loss = torch.tensor(loss)          # 以降の表示・記録は float 扱いに揃える
            mv["reward"].append(mr); mv["kl"].append(mkl)
            mv["disc"].append(parts["disc"])
            # 停止判定は移動平均で行う(生値は毎ステップ別ロールアウトなので揺れが大きい)
            avg = {k: float(np.mean(v)) for k, v in mv.items()}
            self._log({"even/reward": mr, "even/reward_meta": parts["meta"],
                       "even/reward_disc": parts["disc"], "even/reward_sim": parts["sim"],
                       "even/kl": mkl, "even/pg_loss": float(pg),
                       "even/loss": float(loss.detach()), "even/step": step,
                       "even/reward_min": float(rewards.min()),
                       "even/reward_max": float(rewards.max()),
                       "even/reward_sd": float(rewards.std()),
                       "even/similarity": parts.get("sim_mean", float("nan")),
                       "even/reward_ma": avg["reward"], "even/kl_ma": avg["kl"],
                       "even/disc_ma": avg["disc"],
                       **{f"even/hit_{k}": v for k, v in parts.get("hit", {}).items()},
                       "even/empty_rate": parts.get("empty_rate", 0.0),
                       "even/reward_struct": parts.get("struct", 0.0),
                       "even/struct_measure": parts.get("struct_measure", float("nan")),
                       "even/struct_inst": parts.get("struct_inst", float("nan")),
                       "even/key_top1": parts.get("key_top1", float("nan")),
                       "even/key_top3": parts.get("key_top3", float("nan")),
                       "round": rnd})
            if step % steps_per_check == 0:
                print(self._format_even_log(rnd, step, mr, rewards, parts, avg,
                                            mkl, float(loss.detach())))
            reason = (er.should_stop(avg['reward'], avg['kl'], avg['disc'])
                      if len(mv['kl']) == W else None)

            # 健全なうちだけスナップショットを更新する(汚染後に上書きしないこと)
            if reason is None and step % SNAP_EVERY == 0 and avg['kl'] <= self.even_cfg.kl_stop:
                snap = {k: v.detach().clone() for k, v in self.gem.state_dict().items()}
                snap_step = step

        rolled_back = False
        if reason and reason.startswith("reward_hacking"):
            self.gem.load_state_dict(snap)          # ★ 直前の健全な重みへ戻す
            rolled_back = True
            print(f"[EvenRound] ロールバック: step {snap_step} 時点の重みへ復元 "
                  f"(汚染系列は Q-Table 経由で判別器の教材として残る)")
        print(f"[EvenRound] 終了: {reason or 'max_steps'}")
        self._save("gem", rnd)
        return {"steps": step, "reason": reason, "rolled_back": rolled_back,
                "snap_step": snap_step}

    def even_cfg_max_steps(self):
        return self.even_cfg.max_steps

    # -- 交互ループ ---------------------------------------------------
    def run(self, max_total_steps: int = 100000, target_hits: int = 3,
            start_round: int = 1, max_rounds: int = 10000):
        """奇数/偶数を交互に回す。ラウンド数ではなく **成果** で打ち切る。

        終了条件(いずれか):
          1. 全ラウンドの累計ステップが max_total_steps に到達。
          2. 偶数ラウンドが「報酬目標達成(reward_stop 超え)」で終わった回数が
             target_hits に到達。KL 超過(報酬ハッキング)での終了は数えない
             — あれは達成ではなく失敗なので、カウントすると早期に打ち切ってしまう。
        """
        total_steps, hits, rnd = 0, 0, start_round
        while rnd < start_round + max_rounds:
            out = self.run_odd(rnd) if rnd % 2 == 1 else self.run_even(rnd)
            kind = "odd" if rnd % 2 else "even"
            total_steps += int(out.get("steps", 0))
            reason = out.get("reason") or ""
            if kind == "even" and reason.startswith("achieved"):
                hits += 1
                print(f"[CARL] ★ 報酬目標の達成 {hits}/{target_hits} 回目")

            rec = {"round": rnd, "kind": kind, "total_steps": total_steps,
                   "hits": hits, **out}
            self.history.append(rec)
            self._log({"round_end/round": rnd, "round_end/steps": out.get("steps", 0),
                       "round_end/total_steps": total_steps, "round_end/hits": hits,
                       "round_end/qtable_size": len(self.qtable)})

            if hits >= target_hits:
                print(f"\n[CARL] 終了: 報酬目標を {hits} 回達成 (累計 {total_steps} steps)")
                break
            if total_steps >= max_total_steps:
                print(f"\n[CARL] 終了: 累計ステップ {total_steps} >= {max_total_steps} "
                      f"(報酬目標の達成 {hits}/{target_hits} 回)")
                break
            rnd += 1

        if self.wandb is not None:
            self.wandb.finish()
        return self.history

    def prefill_qtable(self, n: Optional[int] = None):
        """Q-Table が空のまま奇数ラウンドへ入らないよう、現行 gem の出力で満たす。

        途中再開したとき、Q-Table はメモリ上にしか無いので空から始まってしまう。
        すると次の奇数ラウンドは「最新 gem だけ」で判別器を鍛えることになり、
        直近専用の検出器へ退化する経路を塞げない(Q-Table を置いた意味が消える)。

        注意: ここで詰めるのは **現行 gem の出力** なので、本来の
        「経過ラウンドに対し (1-forget_ratio)^age で減衰する混合分布」にはならず、
        直近 1 世代ぶんの再構成になる。保存済みテーブル(`q_resume`)があれば
        そちらが優先で、これはあくまで無いときのフォールバック。
        偶数ラウンドの直前(= gem がまだそのラウンドで更新されていない時点)なら、
        現行 gem は 1 つ前の奇数ラウンドの gem と同一なので忠実に復元できる。
        """
        n = n or self.qtable.q_max
        if len(self.qtable) >= n:
            return
        print(f"[CARL] Q-Table を現行 gem の出力で充填します (目標 {n} 本)")
        r = OddRound(self.gem, self.ana, self.tok, self.dev, self.qtable,
                     self.odd_cfg, rng=self.rng)
        r.round_no = 0
        r.quiet_pool = True                       # 数十回ぶんのプール更新ログを抑える
        last = 0
        while len(self.qtable.new) < n:
            got = self.qtable.add(r.refresh_gem_pool(self.human_ana))
            if got == 0:                          # これ以上増えない(生成が全滅)なら打ち切る
                break
            if len(self.qtable.new) - last >= 500:
                last = len(self.qtable.new)
                print(f"  充填中 {last}/{n} 本", flush=True)
        st = self.qtable.commit()
        qp = os.path.join(P.OUT_DIR, "qtable.prefill.pt")
        torch.save({"table": self.qtable.table, "round": self.qtable.round}, qp)
        print(f"[CARL] Q-Table 充填完了: {st['size']} 本 -> {qp} に保存")

    def _format_even_log(self, rnd, step, mr, rewards, parts, avg, kl, loss) -> str:
        """偶数ラウンドの1ブロック。各行が「何の指標か / どちらが良いか / 目標」を持つ。

        値だけ並べると、加点なのか減点なのか・1.00 が良いのか悪いのかが読めない。
        符号と到達目標を必ず併記する。
        """
        cfg = self.even_cfg
        nan = float("nan")
        hit = parts.get("hit", {})
        # ana が推論しない項目(gmc)は常に 0.00 になるだけなので出さない
        hit = {k: v for k, v in hit.items() if k != "gmc"}

        hack_kl = avg["kl"] > cfg.kl_stop
        hack_disc = avg["disc"] > cfg.disc_stop
        verdict = ("★該当 → 巻き戻し" if (hack_kl and hack_disc) else "該当なし")

        def bar(v, lo, hi, w=20):
            """[lo,hi] を w 文字のゲージにする。進捗を目で追えるようにするため。"""
            t = 0.0 if hi <= lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))
            return "█" * int(round(t * w)) + "·" * (w - int(round(t * w)))

        L = []
        L.append(f"[R{rnd} 偶数/GRPO] step {step:5d}/{cfg.max_steps}")
        L.append(f"  総合報酬   今回 {mr:+.3f}   直近{cfg.stop_window}平均 {avg['reward']:+.3f} "
                 f"/ 目標 {cfg.reward_stop:+.2f}  {bar(avg['reward'], 0, cfg.reward_stop)}")
        L.append(f"             ばらつき sd {float(rewards.std()):.3f}  "
                 f"範囲 {float(rewards.min()):+.2f}〜{float(rewards.max()):+.2f}")
        L.append(f"  ├加点 判別器 {parts['disc']:+.3f}  [0〜+1.0] "
                 f"高いほど ana を「人間が書いた」と誤認させている")
        L.append(f"  ├減点 META   {parts['meta']:+.3f}  [-1.0〜+0.75] "
                 f"ana の推論 META が条件からずれた分(完全一致なら加点)")
        L.append(f"  ├減点 構造   {parts.get('struct', 0.0):+.3f}  [-0.5〜0] "
                 f"小節数・楽器の指示違反(生成物から実測)")
        L.append(f"  └減点 類似度 {parts['sim']:+.3f}  [-0.5〜0] "
                 f"人間の正解に似すぎた分(実測 {parts.get('sim_mean', nan):.3f}、0.5超で減点開始)")
        L.append(f"  指示追従   小節 {100*parts.get('struct_measure', nan):5.1f}%  "
                 f"楽器 {100*parts.get('struct_inst', nan):5.1f}%   ← ルールベース実測(高いほど良い)")
        L.append(f"             ana推論の一致度 " +
                 "  ".join(f"{k} {v:.2f}" for k, v in sorted(hit.items())) +
                 f"  key(1位) {parts.get('key_top1', nan):.2f} (3位以内 {parts.get('key_top3', nan):.2f})")
        L.append(f"  生成       空生成率 {100*parts.get('empty_rate', 0.0):.1f}%  ← 0% が正常")
        L.append(f"  方策の乖離 KL {kl:.4f}   直近{cfg.stop_window}平均 {avg['kl']:.4f} "
                 f"/ 上限 {cfg.kl_stop:.2f}  {bar(avg['kl'], 0, cfg.kl_stop)}  低いほど base に近い")
        L.append(f"  損失       {loss:+.4f} (= -優位性 + {cfg.kl_coef}×KL。"
                 f"GRPO は優位性をグループ内で標準化するので平均は 0 になる)")
        L.append(f"  報酬ハッキング判定  KL {avg['kl']:.3f} {'>' if hack_kl else '≦'} "
                 f"{cfg.kl_stop:.2f}  かつ  判別器 {avg['disc']:.3f} "
                 f"{'>' if hack_disc else '≦'} {cfg.disc_stop:.2f}  → {verdict}")
        return "\n".join(L)

    def _save(self, which: str, rnd: int):
        """チェックポイント保存。既定では最新のみ上書きしてディスクを食い潰さない。

        ana は LoRA + PMA + OutHead だけが学習対象なので、その分だけ保存すれば足りる
        (162M 全部で約650MB に対し約13MB)。backbone は元の重みから復元できる。

        ★ 必ず `train()`(= loralib アンマージ)状態で state_dict を取ること。
          `eval()` 中は weight に B@A*scaling が焼き込まれており、それを保存すると
          読み直し時の eval() で二重マージになる。通常の SFT 学習(train.py)は
          train() 状態で保存しているので、保存形式もそちらに揃うことになる。
        """
        m = self.ana if which == "ana" else self.gem
        tag = f".r{rnd}" if self.keep_round_ckpts else ""
        path = os.path.join(P.OUT_DIR, f"MORTM-{which}.carl{tag}.pth")
        if which == "ana":
            # Q-Table はメモリ上にしか無く、クラッシュのたびに失われていた
            # (実際 2 回失った)。ラウンド境界で必ず落としておく。
            qp = os.path.join(P.OUT_DIR, f"qtable{tag}.pt")
            torch.save({"table": self.qtable.table, "round": self.qtable.round}, qp)
            print(f"[CARL] Q-Table 保存: {qp} ({len(self.qtable)} 本)")
        was_training = m.training
        m.train()                                   # アンマージしてから取り出す
        try:
            if which == "ana":
                sd = {k: v.detach().clone() for k, v in m.state_dict().items()
                      if ("lora_" in k) or k.startswith(("pma", "out_head"))}
            else:
                sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        finally:
            if not was_training:
                m.eval()                            # 呼び出し元の状態へ戻す
        sd[_UNMERGED_MARK] = torch.tensor(1)         # アンマージ保存済みの目印
        torch.save(sd, path)
        mb = os.path.getsize(path) / 1e6
        print(f"[CARL] 保存: {path} ({mb:.1f}MB)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--q_max", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    CARLDriver(q_max=a.q_max, seed=a.seed).run(rounds=a.rounds)
