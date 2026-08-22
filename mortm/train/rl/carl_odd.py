"""CARL 奇数ラウンド: MORTM-ana(分析器/判別器)の教師あり学習。

役割:
  - **MORTM-gem**: 生成器(Meta2CONST)。凍結。データセットの META から CONST を生成する。
  - **MORTM-ana**: 学習対象。基盤 MORTM に LoRA を付与し、
      (a) CONST -> META(<TE> まで)  の分析、
      (b) CONST が AI 由来か人間由来か の判別(PMA + OutHead)
    を同時に学習する。

1 ステップの流れ(オンザフライ):
  1. MORTM-gem がデータセットの META から CONST-MIDI(トークン列)を生成する。
  2. 生成物から key / density を **実測して** META を作り直す(条件 META の流用ではない)。
  3. 人間の CONST と AI の CONST を **半々** で混ぜる。
  4. MORTM-ana に CONST を見せ、`<META> <SYSTEM>..<TAG_END> <TE>` を予測させる。
     系列規則は `mortm/utils/convert_foundation.py: convert_analysis_sft` に従う。
  5. デコーダ出力を PMA でプーリングし、1 次元に落として AI/Human を判別する。
  6. META の MaskedCrossEntropy + 判別の BCE を目的関数として教師あり学習する。
  7. 判別器の正解率が閾値(既定 0.90)を超えたら奇数ラウンドを終了する。
"""

import math
import os
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from loralib import mark_only_lora_as_trainable
from flash_attn.bert_padding import unpad_input

from ...models.mortm import MORTM
from ...models.modules.config import MORTMArgs
from ...models.modules.layers import PMA
from ...utils.convert_foundation import INSTRUMENT_RULES
from ...utils.convert import split_sequence_measure  # 小節分割(density算出に使用)
from .task_norm import rev_safe


# ----------------------------------------------------------------------
# MORTM-ana: 分析 + AI/Human 判別
# ----------------------------------------------------------------------
def _pack_music_only(out: Tensor, cu_seqlens: Tensor, x: Tensor,
                     padding_mask: Tensor, meta_id: int):
    """packed varlen の出力から、各系列の `<META>` より前だけを詰め直す。

    `unpad_input` はパディングを除いた順序で詰めるので、系列 i の先頭は
    `cu_seqlens[i]`。パディングは右側にしか無いため、先頭から `<META>` の
    位置までがそのまま音楽ブロックに対応する。

    Returns: (音楽だけを詰めた出力, 新しい cu_seqlens)
    """
    is_meta = (x == meta_id) & padding_mask
    has = is_meta.any(dim=1)
    pos = torch.where(has, is_meta.float().argmax(dim=1),
                      padding_mask.sum(dim=1))          # 見つからなければ全長
    pos = pos.to(torch.long).clamp(min=1)               # 最低1トークンは残す
    starts = cu_seqlens[:-1].to(torch.long)
    idx = torch.cat([torch.arange(int(s), int(s) + int(l), device=out.device)
                     for s, l in zip(starts, pos)])
    new_cu = torch.cat([torch.zeros(1, dtype=cu_seqlens.dtype, device=out.device),
                        pos.to(cu_seqlens.dtype).cumsum(0)])
    return out[idx], new_cu


class AnalysisMORTM(MORTM):
    """基盤 MORTM に LoRA を付与し、AI/Human 判別ヘッド(PMA + OutHead)を足したモデル。

    - 言語ヘッド(Wout)はそのまま使い、CONST -> META の自己回帰予測を行う。
    - 併せてデコーダ出力を PMA でプーリングし、OutHead で 1 次元に落として判別する。
    """

    def __init__(self, args: MORTMArgs, progress, pma_out_dim: Optional[int] = None):
        super().__init__(args, progress)
        pod = pma_out_dim if pma_out_dim is not None else self.d_model
        self.pma_out_dim = pod
        self.pma = PMA(self.d_model, pod)
        self.out_head = nn.Sequential(          # AI(1) / Human(0) の 1 次元ロジット
            nn.Linear(pod, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
        )

    def forward_with_disc(self, x: Tensor, padding_mask: Tensor, is_causal: bool = True,
                          meta_id: Optional[int] = None):
        """言語ロジットと判別ロジットを同時に返す。

        Args:
            meta_id: `<META>` のトークンID。渡すと **判別のプーリングを
                `<META>` より前(= 音楽ブロックのみ)に限定** する。

        ★ なぜ限定が要るか:
          入力は `<EOS> [音楽] <META> [META] <TE>` で、PMA は attention プーリング
          なので何も指定しないと **META ブロックも判定材料に使う**。ところがその
          META は人間側とAI側で別の手続きで作られており(人間=データセットの条件META
          そのまま、AI=生成物から実測して組み直し)、書式が体系的に違う。実測では
          「density と key が揃っていて <thinking> が無ければ AI」という音楽を一切
          見ない規則だけで AI 100% / 人間 65% / 全体 82.5% に達する。
          判別器はこの近道を先に覚え、停止閾値 0.85 をそれだけで超えてしまう。
          しかも偶数ラウンドでは META を ana 自身が greedy 生成するため常に「AI 形」
          に固定され、gem は何を書いてもこのチャネルを動かせない。
          報酬時点の判別精度を実測すると 56%(人間の正解の 78% が「AI」と誤判定)で、
          R_disc は報酬として機能していなかった。

        Returns:
            logits: (B, S, V)  次トークン予測の分布
            disc:   (B,)       AI/Human 判別のロジット(sigmoid 前)
        """
        from flash_attn.bert_padding import pad_input

        # 重みは fp32 のままなので autocast 必須(kv_generate と同じ扱い)。
        # ここで包んでおかないと呼び出し側が忘れて bf16/fp32 の不一致で落ちる。
        dev = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            emb = self.embedding(x).to(dtype=torch.bfloat16)
            b, s, _ = emb.size()
            xu, indices, cu_seqlens, max_s, _ = unpad_input(emb, padding_mask)
            out = self.decoder(tgt=xu, tgt_is_causal=is_causal, cu_seqlens=cu_seqlens,
                               max_seqlen=max_s, batch_size=b, indices=indices,
                               is_save_cache=False)
            # 判別: 音楽ブロックだけを packed のまま切り出してプーリングする
            p_out, p_cu = (out, cu_seqlens) if meta_id is None else \
                _pack_music_only(out, cu_seqlens, x, padding_mask, meta_id)
            disc = self.out_head(self.pma(p_out, p_cu)).squeeze(-1)
            # 言語: pad を戻して語彙へ射影
            logits = self.Wout(pad_input(out, indices, b, s))
        return logits.float(), disc.squeeze(-1).float() if disc.dim() > 1 else disc.float()

    def mark_trainable(self):
        """LoRA + PMA/OutHead のみ学習可能にする(backbone 本体は凍結)。"""
        mark_only_lora_as_trainable(self)
        for name, p in self.named_parameters():
            if name.startswith("pma") or name.startswith("out_head"):
                p.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# META の再計算(生成物の実測ラベル)
# ----------------------------------------------------------------------
def density_token_of(seq: np.ndarray, program_name: str, tokenizer) -> Optional[int]:
    """convert_foundation._calculate_density_token と同一規則で密度トークンを返す。"""
    rule = INSTRUMENT_RULES.get(program_name)
    if rule is None:
        raise ValueError(f"Undefined instrument rule: {program_name}")
    max_notes = 45 if rule.is_polyphonic else 24
    s_min, s_max = tokenizer.get_length_tuple("s")
    measures = split_sequence_measure(seq, 1, tokenizer.get("<SME>"))

    total, valid = 0.0, 0
    for m in measures:
        n = int(np.sum((m >= s_min) & (m <= s_max)))
        if n > max_notes:
            return None                      # 閾値超過は破棄(規則どおり)
        total += n / max_notes
        valid += 1
    if valid == 0:
        return tokenizer.get("<NOTE_DENSE_1>")
    score = max(1, min(10, math.ceil((total / valid) * 10))) if total > 0 else 1
    return tokenizer.get(f"<NOTE_DENSE_{score}>")


_ENHARMONIC = {'C#': 'Db', 'Db': 'C#', 'D#': 'Eb', 'Eb': 'D#', 'F#': 'Gb',
               'Gb': 'F#', 'G#': 'Ab', 'Ab': 'G#', 'A#': 'Bb', 'Bb': 'A#'}


def key_str_to_token_name(key_str: str, tokenizer) -> str:
    """music21 形式("C major" / "e- minor") -> 語彙のキー名("CM" / "Ebm")。

    語彙は `k_CM` / `k_Cm` 形式なので、そのまま繋ぐと `<PAD>` に落ちる。
    変換規則と異名同音の救済は `convert.py: _get_key_str` と同一に揃える。
    """
    if not key_str:
        return "Unknown"
    parts = str(key_str).split()
    tonic = parts[0].replace('-', 'b')
    tonic = tonic[0].upper() + tonic[1:]
    mode = 'M' if (len(parts) > 1 and parts[1].lower().startswith('major')) else 'm'
    cand = f"{tonic}{mode}"
    if f"k_{cand}" in tokenizer.tokens:
        return cand
    alt = _ENHARMONIC.get(tonic)
    if alt and f"k_{alt}{mode}" in tokenizer.tokens:
        return f"{alt}{mode}"
    return "Unknown"


def build_meta(active_program_info: List[Tuple[str, int]], key_str: str,
               measure_count: Optional[int], tokenizer, genre_tokens=None) -> np.ndarray:
    """`<SYSTEM> [<INST_x> <NOTE_DENSE_n>]* [<GEN_MEASURE_COUNT_k>] [genre] k_key <TAG_END>` を作る。

    convert.make_system_prompt と同一並び(先頭の <EOS> は含めない)。

    measure_count に None を渡すと `<GEN_MEASURE_COUNT_k>` を省く。CARL では
    PAST/CONST/FUTURE を畳んだ最大24小節を分析対象にするため、
      - 生成区間の長さは畳んだ入力から決定できず ana にとって不可知
      - 語彙側も 1〜8 しか持たない(畳んだ小節数は 8 を超える)
    の二重の理由で GMC を扱わない。小節数はルールベース(`structural_check`)が担当する。
    """
    p = [tokenizer.get("<SYSTEM>")]
    for name, dens in active_program_info:
        p.append(tokenizer.get(f"<INST_{name}>"))
        if dens is not None:
            p.append(dens)
    if measure_count is not None:
        p.append(tokenizer.get(f"<GEN_MEASURE_COUNT_{measure_count}>"))
    if genre_tokens:
        p.extend(genre_tokens)
    p.append(tokenizer.get(f"k_{key_str_to_token_name(key_str, tokenizer)}"))
    p.append(tokenizer.get("<TAG_END>"))
    return np.array(p, dtype=int)


def genre_tokens_of(meta_ids, tokenizer) -> List[int]:
    """META 並びからジャンルトークン(`<GENRE_x>`)だけを拾う。"""
    out = []
    for t in meta_ids:
        name = rev_safe(tokenizer, t)
        if isinstance(name, str) and name.startswith("<GENRE_"):
            out.append(int(t))
    return out


def rebuild_meta_from_generated(seq: np.ndarray, active_program_info, tokenizer,
                                cond_meta=None, tempo: int = 120) -> Optional[np.ndarray]:
    """生成 CONST から分析ラベル(答え側 META)を作り直す。

    key と density は生成物から **実測** する(`get_key_from_tokens` / `density_token_of`)。
    ジャンルは音符から測れないので、生成条件に使った META から **引き継ぐ**
    (SFT の分析データはジャンル込みなので、ここで欠かすと形式が食い違う)。
    人間サンプルの場合は cond_meta にそのサンプル自身の META を渡せばよい。

    Returns: `<SYSTEM>..<TAG_END>` の配列。密度が規則外なら None(破棄)。
    """
    from ...utils.key_from_tokens import get_key_from_tokens

    key_str = get_key_from_tokens(tokenizer, seq, tempo=tempo)
    if key_str is None:
        return None
    info = []
    for name, _ in active_program_info:
        dens = density_token_of(seq, name, tokenizer)
        if dens is None:
            return None                       # 密度閾値超過は規則どおり破棄
        info.append((name, dens))
    measures = len(split_sequence_measure(seq, 1, tokenizer.get("<SME>")))
    genre = genre_tokens_of(cond_meta, tokenizer) if cond_meta is not None else None
    return build_meta(info, key_str, measures, tokenizer, genre_tokens=genre)


def build_analysis_sample(music_block: np.ndarray, meta: np.ndarray, tokenizer) -> np.ndarray:
    """`<EOS> <CONST_M>[音楽]<TAG_END> <META> <SYSTEM>[meta]<TAG_END> <TE>`(分析SFTと同一規則)。"""
    return np.concatenate([
        np.array([tokenizer.get("<EOS>")], dtype=int),
        music_block,
        np.array([tokenizer.get("<META>")], dtype=int),
        meta,
        np.array([tokenizer.get("<TE>")], dtype=int),
    ])


# ----------------------------------------------------------------------
# 損失: META の MaskedCE + AI/Human の BCE
# ----------------------------------------------------------------------
class OddRoundLoss(nn.Module):
    """L = CE(META 区間のみ) + w * BCE(AI/Human)。

    META 区間 = `<META>` トリガ以降(答え側)。CONST(問題側)には損失を掛けない。
    """

    def __init__(self, meta_trigger_id: int, disc_weight: float = 1.0, ignore_index: int = 0):
        super().__init__()
        self.meta_trigger_id = meta_trigger_id
        self.disc_weight = disc_weight
        self.ignore_index = ignore_index

    def build_meta_mask(self, x: Tensor) -> Tensor:
        """`<META>` 以降を True にするマスクを作る(x: (B,S) 入力トークン列)。"""
        is_trig = (x == self.meta_trigger_id)
        return is_trig.cumsum(dim=1) > 0

    def forward(self, logits: Tensor, target: Tensor, meta_mask: Tensor,
                disc_logit: Tensor, is_ai: Tensor):
        # 言語: META 区間だけを残し、他は ignore_index に落とす
        tgt = target.clone()
        tgt[~meta_mask] = self.ignore_index
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                             tgt.reshape(-1).long(), ignore_index=self.ignore_index)
        bce = F.binary_cross_entropy_with_logits(disc_logit.float(), is_ai.float())
        return ce + self.disc_weight * bce, ce.detach(), bce.detach()


# ----------------------------------------------------------------------
# 奇数ラウンドの外側ループ
# ----------------------------------------------------------------------
def split_analysis_sample(sample, tokenizer):
    """分析SFTサンプルを (音楽ブロック, 答え側META) に割る。

    形式: `<EOS> <CONST_M>..<TAG_END> <META> <SYSTEM>..<TAG_END> <TE>`
    人間サンプルは META を作り直す必要が無いので、ここで切り出して使い回す。
    """
    s = np.asarray(sample, dtype=np.int64)
    hit = np.where(s == tokenizer.get("<META>"))[0]
    if len(hit) == 0:
        return None, None
    i = int(hit[0])
    music = s[1:i] if int(s[0]) == tokenizer.get("<EOS>") else s[:i]
    meta = s[i + 1:]
    if len(meta) and int(meta[-1]) == tokenizer.get("<TE>"):
        meta = meta[:-1]
    return music, meta


def inst_names_of(meta_ids, tokenizer):
    """META から `<INST_x>` を拾って楽器名(x)のリストにする。"""
    out = []
    for t in meta_ids:
        name = rev_safe(tokenizer, t)
        if isinstance(name, str) and name.startswith("<INST_"):
            out.append(name[len("<INST_"):-1])
    return out


class OddRoundConfig:
    """奇数ラウンドのハイパーパラメータ。"""

    def __init__(self, batch_size=16, pool_size=256, lr=1e-4, disc_weight=1.0,
                 acc_threshold=0.85, eval_window=100, max_steps=100000,
                 max_measures=8, top_p=0.9, temperature=1.0, max_new=1200,
                 grad_clip=1.0, gen_chunk=16):
        self.batch_size = batch_size
        self.pool_size = pool_size          # gem プールの本数(生成とキー推定を償却する単位)
        self.lr = lr
        self.disc_weight = disc_weight
        self.acc_threshold = acc_threshold
        self.eval_window = eval_window      # 停止判定に使う直近バッチ数(移動平均の窓)
        self.max_steps = max_steps
        self.max_measures = max_measures
        self.top_p = top_p
        self.temperature = temperature
        self.max_new = max_new
        self.grad_clip = grad_clip
        self.gen_chunk = gen_chunk      # 一度に生成するバッチ幅(VRAM 制約)


class OddRound:
    """MORTM-ana を教師あり学習し、判別精度が閾値を超えたら次ラウンドへ渡す。

    gem は凍結。ana(LoRA + PMA + OutHead)だけを更新する。
    Q-Table により「過去に AI と判定した例」を混ぜ、判別器が直近の
    報酬ハッキング検出器へ退化するのを防ぐ。
    """

    def __init__(self, gem, ana, tokenizer, device, qtable, cfg: OddRoundConfig,
                 optimizer=None, rng=None):
        from .q_table import mix_batch, accuracy_by_source, should_stop_odd
        self._mix_batch = mix_batch
        self._acc_by_source = accuracy_by_source
        self._should_stop = should_stop_odd

        self.gem, self.ana = gem, ana
        self.tok, self.dev, self.q, self.cfg = tokenizer, device, qtable, cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.loss_fn = OddRoundLoss(tokenizer.get("<META>"), cfg.disc_weight)

        self.gem.eval()
        for p in self.gem.parameters():
            p.requires_grad = False
        n_train = self.ana.mark_trainable()
        self.opt = optimizer or torch.optim.AdamW(
            [p for p in self.ana.parameters() if p.requires_grad], lr=cfg.lr)
        self.round_no = 0          # 表示用。ドライバが各ラウンドで設定する
        self.quiet_pool = False    # True でプール更新ログを抑制(充填時に使う)
        print(f"[OddRound] 学習対象パラメータ: {n_train:,} (LoRA + PMA + 判別ヘッド。backbone は凍結)")

    # -- gem プールの生成(生成 + キー/密度の実測をまとめて償却) --------
    @torch.no_grad()
    def refresh_gem_pool(self, human_pool):
        """人間サンプルの META を条件に gem へ生成させ、答え側 META を作り直す。

        キー推定は 1 本あたり約 97ms かかるため、毎バッチではなく pool_size 本まとめて
        行い、以後の複数ステップで使い回す。
        """
        idx = self.rng.choice(len(human_pool), size=min(self.cfg.pool_size,
                                                        len(human_pool)), replace=False)
        recs = [human_pool[i] for i in np.atleast_1d(idx)]
        # プロンプトは SFT 学習時のものをそのまま使う(PAST/FUTURE/他楽器を含む)。
        # 人間側と同じ条件から生成させるので、条件分布のずれが手がかりにならない。
        conds = [r["cond_meta"] for r in recs]
        prompts = [[int(t) for t in r["prompt"]] for r in recs]
        gens = generate_chunked(
            self.gem, prompts, self.tok, self.dev, self.cfg.max_measures,
            chunk=self.cfg.gen_chunk,
            p=self.cfg.top_p, temperature=self.cfg.temperature, max_new=self.cfg.max_new)

        from ...utils.key_from_tokens import get_key_from_tokens

        CONST_M, TAG_END, TE = (self.tok.get("<CONST_M>"), self.tok.get("<TAG_END>"),
                                self.tok.get("<TE>"))
        pool = []
        # 破棄理由の内訳。生成物に <BLANK> が含まれるのは正常なので弾かない。
        # 弾くのは「分析ラベルが作れないもの」だけ:
        #   empty   : 音符が1つも無い(生成失敗)
        #   no_key  : キー推定不能(音符が少なすぎる等)
        #   density : 密度が規則上限超過(convert_foundation と同一規則)
        why = {"empty": 0, "no_key": 0, "density": 0}
        from .gen_sft_pool import fold_to_const

        for rec, cond, g in zip(recs, conds, gens):
            body = [int(t) for t in g if int(t) != TE]
            if not body:
                why["empty"] += 1
                continue
            # ★ 人間側と同じ経路で統一 CONST に畳む。条件側の人間ブロック
            #   (PAST/FUTURE/他楽器)と連結されるので、判別器はつなぎ目を見る。
            const, merged = fold_to_const(rec, body, self.tok)
            if not merged:
                why["empty"] += 1
                continue
            # 以降は `[<INST_x> 系列 <ESEQ>]*` の形が要る(density_token_of が
            # 楽器マーカーで区間を切るため)。畳んだブロックから前後のマーカーを外す。
            seq = np.asarray([int(t) for t in const[1:-1]], dtype=np.int64)
            names = inst_names_of(cond, self.tok)
            key_str = get_key_from_tokens(self.tok, seq)
            if key_str is None:
                why["no_key"] += 1
                continue
            info, bad = [], False
            for n in names:
                try:
                    # ★ 楽器ごとの系列を渡す。畳んだブロック全体を渡すと
                    #   全楽器の小節が合算され、多楽器レコードで全楽器が
                    #   同じ密度トークンになってしまう(人間側は楽器別に算出
                    #   されているので、そこも非対称なラベルになる)。
                    if n not in merged:
                        bad = True
                        break
                    dens = density_token_of(np.asarray(merged[n], dtype=np.int64),
                                            n, self.tok)
                except Exception:
                    dens = None
                if dens is None:
                    bad = True
                    break
                info.append((n, dens))
            if bad:
                why["density"] += 1
                continue
            # GMC は付けない。畳んだ CONST からは生成区間の長さが決定できず、
            # ana にとって不可知なラベルになるため(人間側 `strip_gmc` と対称)。
            meta = build_meta(info, key_str, None, self.tok,
                              genre_tokens=genre_tokens_of(cond, self.tok))
            pool.append({"seq": np.asarray(const, dtype=np.int64), "meta": meta})
        drop = sum(why.values())
        if getattr(self, "quiet_pool", False):
            self.last_pool_stats = {"kept": len(pool), "total": len(gens), **why}
            return pool
        print(f"  [gemプール更新] {len(pool)}/{len(gens)} 本を AI 例として採用"
              f"  (破棄 {drop}: 音符なし {why['empty']} / キー推定不能 {why['no_key']} / "
              f"密度が規則上限超過 {why['density']})")
        self.last_pool_stats = {"kept": len(pool), "total": len(gens), **why}
        return pool

    # -- 1 バッチ -----------------------------------------------------
    def _pad(self, seqs):
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long, device=self.dev)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(np.asarray(s), dtype=torch.long, device=self.dev)
        return x, (x != 0)

    def train_step(self, human_pool, gem_pool):
        items, is_ai, source = self._mix_batch(human_pool, gem_pool, self.q,
                                               self.cfg.batch_size, self.rng)
        seqs = [build_analysis_sample(it["seq"], it["meta"], self.tok) for it in items]
        x, pad = self._pad(seqs)
        y = torch.tensor(is_ai, dtype=torch.float32, device=self.dev)

        # 判別は音楽ブロックのみ。META ブロックは人間/AI で書式が非対称なので、
        # そこを見せると音楽を見ずに分類できてしまう。
        logits, disc = self.ana.forward_with_disc(
            x, pad, is_causal=True, meta_id=self.tok.get("<META>"))
        mask = self.loss_fn.build_meta_mask(x)
        loss, ce, bce = self.loss_fn(logits[:, :-1], x[:, 1:], mask[:, 1:], disc, y)

        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.ana.parameters() if p.requires_grad], self.cfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        acc = self._acc_by_source(disc.detach(), is_ai, source)
        return {"loss": float(loss.detach()), "ce": float(ce), "bce": float(bce), **acc}

    # -- ラウンド本体 -------------------------------------------------
    def run(self, human_pool, log_every=20, on_step=None):
        """判別精度が閾値を超えるまで回す。終了時に Q-Table を一括更新する。

        on_step: 1 ステップごとに統計 dict を受け取るコールバック(wandb 連携用)。

        戻り値: {"steps", "reason", "qtable"}
        """
        from collections import deque
        cfg = self.cfg
        self.ana.train()
        gem_pool = self.refresh_gem_pool(human_pool)
        if not gem_pool:
            # gem が音符を出せない = 生成器が壊れている。この状態で続けると
            # バッチに gem サンプルが入らず acc:gem=nan になり、停止判定が
            # 永久に成立せず max_steps まで空回りする(実際に6000step無駄にした)。
            raise RuntimeError(
                "OddRound: gem プールが空です。生成器が音符を出力できていません "
                f"(直近の内訳: {getattr(self, 'last_pool_stats', {})})。"
                "偶数ラウンドで重みが破壊された可能性があります。")
        self.q.add(gem_pool)                    # 今ラウンド分を蓄積(commit は最後)
        hist = {k: deque(maxlen=cfg.eval_window)
                for k in ("gem", "human", "qtable", "overall")}
        reason, step = None, 0

        for step in range(1, cfg.max_steps + 1):
            if step % max(1, cfg.pool_size // cfg.batch_size) == 0:
                fresh = self.refresh_gem_pool(human_pool)      # プールを入れ替え
                if not fresh:
                    raise RuntimeError(
                        f"OddRound: step {step} で gem プールが空になりました "
                        f"({getattr(self, 'last_pool_stats', {})})。生成器が壊れています。")
                gem_pool = fresh
                self.q.add(gem_pool)
            st = self.train_step(human_pool, gem_pool)
            if on_step is not None:
                on_step({f"odd/{k}": v for k, v in st.items()} |
                        {"odd/qtable_size": len(self.q), "odd/step": step})
            for k in hist:
                if k in st:
                    hist[k].append(st[k])
            if step % log_every == 0:
                avg_now = {k: (float(np.mean(v)) if len(v) else float("nan"))
                           for k, v in hist.items()}
                nan = float("nan")
                w = len(hist["gem"])
                th = cfg.acc_threshold
                ok = (avg_now["gem"] >= th and avg_now["human"] >= th)

                def bar(v, width=20):
                    t = 0.0 if v != v else min(1.0, max(0.0, v))
                    return "█" * int(round(t * width)) + "·" * (width - int(round(t * width)))

                print(
                    f"[R{self.round_no} 奇数/判別器] step {step:6d}/{cfg.max_steps}\n"
                    f"  損失       合計 {st['loss']:.4f}"
                    f"  = META推論 {st['ce']:.4f} (交差エントロピー)"
                    f" + AI/人間判別 {st['bce']:.4f} (BCE)  ← 下がるほど良い\n"
                    f"  判別正解率 今回   gem {st.get('gem', nan):.3f}  "
                    f"人間 {st.get('human', nan):.3f}  "
                    f"Q-Table {st.get('qtable', nan):.3f}  "
                    f"全体 {st.get('overall', nan):.3f}\n"
                    f"             直近{w:3d}回平均  "
                    f"gem {avg_now['gem']:.3f} {bar(avg_now['gem'])}\n"
                    f"                          "
                    f"人間 {avg_now['human']:.3f} {bar(avg_now['human'])}\n"
                    f"  ラウンド終了条件  gem と 人間 の両方が直近{cfg.eval_window}回平均で "
                    f"{th:.2f} 以上  → {'★達成' if ok else '未達'}\n"
                    f"  Q-Table    確定 {len(self.q)} 本 / 今ラウンド蓄積 {len(self.q.new)} 本"
                    f"  (過去の AI 例を 1/4 混ぜて、直近の gem 専用検出器への退化を防ぐ)")
            if len(hist["gem"]) == cfg.eval_window and len(hist["human"]) == cfg.eval_window:
                avg = {k: float(np.mean(v)) for k, v in hist.items() if len(v)}
                reason = self._should_stop(avg, cfg.acc_threshold)
                if reason:
                    break

        qstats = self.q.commit()                # ★ラウンド終了時に一括更新
        print(f"[R{self.round_no} 奇数/判別器] 終了理由: {reason or 'max_steps 到達'}\n"
              f"  Q-Table 更新: {qstats['before']} 本 → 忘却 {qstats['forgot']} "
              f"→ 新規マージ {qstats['merged']} → 確定 {qstats['size']} 本 "
              f"(今ラウンドの蓄積 {qstats['new_pool']} 本)")
        return {"steps": step, "reason": reason, "qtable": qstats}


def generate_chunked(model, prompts, tokenizer, device, max_measures, chunk: int = 16,
                     **kw):
    """`kv_generate_batch` をチャンク分割して呼ぶ。

    KV キャッシュは `position_length + 512` の **固定長** で確保されるため、
    消費 VRAM は系列長ではなくバッチ幅に比例する:

        batch * (position_length+512) * d_model * 2byte * 2(K,V) * n_layer

    160M(768dim/17層/pos5000)では **バッチ1本あたり約288MB**。64本一括で約18GBとなり
    16GB級のGPUでは確保に失敗する(NVML不整合環境では OOM が NVML アサートとして出る)。
    既定 16 本(約4.6GB)ずつに割って回す。
    """
    from ...eval.kv_generate import kv_generate_batch

    out = []
    for i in range(0, len(prompts), chunk):
        part = prompts[i:i + chunk]
        out.extend(kv_generate_batch(
            model, part, tokenizer, device,
            max_measures_list=[max_measures] * len(part), **kw))
        for m in model.modules():                 # チャンク間でキャッシュを解放する
            if hasattr(m, "kv_cache"):
                m.kv_cache = None
            if hasattr(m, "cache_seqlens"):
                m.cache_seqlens = None
        torch.cuda.empty_cache()
    return out
