"""TEST-TASK プロンプト構築（研究設計書 v1.6 §6.1, §9.3）。

5 タスクの評価プロンプトを MIDI から**決定論的**に構築する単一モジュール。全評価・全モデルが
これを import して同一プロンプトを使う（モデル間公平性 §8.4）。

形式（§6.1 v1.2 凍結）:
- **P 形式**: 事前学習ブロックタグのみ。文脈ブロック列 + 標的ブロックの開始タグで打ち切り、
  続きを生成させる（無条件は標的開始タグのみ）。零 SFT 評価(E1 の A1/A2, E9)に用いる。
- **G 形式**: SFT テンプレート(<MGEN>/分析トリガー)。SFT 後モデル(E2,E3)に用いる。

タスク（§6.1、v1.6 + 裁定A 反映: 条件付け/分析の属性は **key + density のみ**。
genre/length(GMC)/chord は基盤 META 目録に無いため除外 = E0-7 恒久ルール）:
  infill        : PAST=前6小節, FUTURE=後6小節 → CONST=中央4小節 (META 有/無 両方)
  continuation  : META+PAST=前8小節 → 8小節
  condgen       : META(key,density) のみ → 8小節
  uncond        : なし → 8小節
  analysis_key  : 音楽16小節 → META(キー抽出)
  analysis_dense: 音楽16小節 → META(密度抽出)

トークン化は既存 MIDIConverter に一元化（往復健全性は §9.12 で pitch 完全可逆を確認済）。
ブロック組み立ては FoundationDataMaker のヘルパを再利用（文字列直書き禁止 §記法規約）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker

TASKS = ("infill", "continuation", "condgen", "uncond", "analysis_key", "analysis_dense")


@dataclass
class Window:
    """1 窓の per-program 区間トークンと META 情報。measure 単位で切り出す。"""
    programs: List[str]
    past: dict          # program -> np.ndarray (前文脈)
    const: dict         # program -> np.ndarray (生成標的/分析対象)
    future: dict        # program -> np.ndarray (後文脈)
    info: list          # [(program, density_token_id)]  (make_system_prompt 用)


class ProtocolBuilder:
    def __init__(self, tokenizer, programs=("PIANO", "SAX")):
        self.tk = tokenizer
        self.programs = list(programs)
        self.EOS = tokenizer.get("<EOS>")

    # --- 窓の列挙 (決定論的: measure グリッドを前から) ---
    def windows(self, directory, file_name, past_m=6, const_m=4, future_m=6, key=None):
        con = MIDIConverter(self.tk, directory, file_name, self.programs, key=key)
        con.convert()
        if con.is_error or con.midi2seq is None:
            return
        maker = FoundationDataMaker(con, min_measure=1, max_measure=8)  # ヘルパ再利用
        if maker.is_error:
            return
        seq_dict = maker.seq_dict
        sme = self.tk.get("<SME>")
        s_e = self.tk.get_length_tuple("s")
        valid = [p for p in con.program_list if p in seq_dict]
        inds = {p: np.where(seq_dict[p] == sme)[0] for p in valid}

        total = past_m + const_m + future_m
        now = 0
        while True:
            past, const, future, info = {}, {}, {}, []
            active = []
            enough = False
            for p in valid:
                ip = inds[p]
                if now + total >= len(ip):
                    continue
                enough = True
                a = seq_dict[p]
                ps, pe = ip[now], ip[now + past_m]
                ce = ip[now + past_m + const_m]
                fe = ip[now + past_m + const_m + future_m]
                cseq = a[pe:ce]
                if not np.any(np.isin(cseq, np.arange(s_e[0], s_e[1]))):
                    continue  # CONST に音符が無い楽器は除外
                dens = maker._calculate_density_token(cseq, p)
                if dens is None:
                    continue
                past[p], const[p], future[p] = a[ps:pe], cseq, a[ce:fe]
                info.append((p, dens))
                active.append(p)
            if not enough:
                break
            if active:
                yield Window(active, past, const, future, info)
            now += past_m

    # --- META ブロック(P形式: <SYSTEM>..k_key..<TAG_END>) ---
    def _meta_block(self, con_maker, info):
        # make_system_prompt は先頭 <EOS> を含む。[1:] が <SYSTEM>..<TAG_END>。
        sp = con_maker.converter.make_system_prompt(0, info)
        return np.asarray(sp[1:], dtype=int)

    def _marker(self, name):
        return np.array([self.tk.get(name)], dtype=int)

    def _framed(self, marker, seqs, programs):
        """<MARKER>[<INST>seq<ESEQ>]*<TAG_END> を組む(_build_melody_block 相当だが決定論: 楽器順固定)。"""
        s_range = self.tk.get_length_tuple("s")
        parts = [self._marker(marker)]
        for prog in programs:
            seq = seqs[prog]
            if not np.any(np.isin(seq, np.arange(s_range[0], s_range[1]))):
                continue
            parts.append(np.concatenate([
                self._marker(f"<INST_{prog}>"), seq, self._marker("<ESEQ>")]))
        parts.append(self._marker("<TAG_END>"))
        return np.concatenate(parts)

    def build(self, window: Window, maker, task: str, include_meta=True):
        """P 形式プロンプトを返す。生成/予測は打ち切り位置以降(モデルが継続する)。
        戻り値 (prompt_ids, ref_const_dict|None)。ref は補完/生成系の参照 CONST。"""
        progs = window.programs
        meta = self._meta_block(maker, window.info)
        eos = np.array([self.EOS], dtype=int)

        def cat(*xs):
            return np.concatenate([x for x in xs if x is not None and len(x)])

        if task == "infill":
            past_b = self._framed("<PAST_M>", window.past, progs)
            fut_b = self._framed("<FUTURE_M>", window.future, progs)
            head = meta if include_meta else None
            # 文脈 + 標的開始タグ <CONST_M> で打ち切り
            prompt = cat(eos, head, past_b, fut_b, self._marker("<CONST_M>"))
            return prompt, window.const
        if task == "continuation":
            past_b = self._framed("<PAST_M>", window.past, progs)
            prompt = cat(eos, meta, past_b, self._marker("<CONST_M>"))
            return prompt, window.const
        if task == "condgen":
            prompt = cat(eos, meta, self._marker("<CONST_M>"))
            return prompt, window.const
        if task == "uncond":
            prompt = cat(eos, self._marker("<CONST_M>"))
            return prompt, window.const
        if task in ("analysis_key", "analysis_dense"):
            # 音楽(CONST を分析対象)を提示し、META 開始タグ <SYSTEM> で打ち切り → META を生成させる
            music = self._framed("<CONST_M>", window.const, progs)
            prompt = cat(eos, music, self._marker("<SYSTEM>"))
            return prompt, None
        raise ValueError(f"unknown task: {task}")
