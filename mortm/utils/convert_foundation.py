import hashlib
import json
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np

from mortm.utils.convert import (
    _AbstractConverter,
    MIDIConverter,
    INSTRUMENT_RULES,
    split_sequence_measure,
)

# --- ジャンルトークン (MIDICaps 用) ---------------------------------------
# ジャンルトークンは custom_token.Genre として tokenizer 末尾に正式登録済み
# (<GENRE_xxx>, ID 647.. 既存IDは不変)。ID は必ず tokenizer.get で解決する。
def genre_names_to_ids(genres, tokenizer):
    """ジャンル名リスト -> トークンIDリスト。tokenizer 経由で解決(未登録は無視)。"""
    ids = []
    for g in genres:
        try:
            ids.append(tokenizer.get(f"<GENRE_{g}>"))
        except KeyError:
            pass
    return ids


class FoundationDataMaker(_AbstractConverter):
    """
    分析と生成が両方できる音楽基盤モデル向けの事前学習データ生成クラス。

    楽曲を Past / Const / Future の 3 区間に分割し、各区間をブロックとして
    並び替え・削除することで、任意のブロック集合を条件として残りを予測する
    能力を獲得させる。
    """

    def __init__(self, converter: MIDIConverter, min_measure=1, max_measure=8, additional_prompt=None,
                 disable_block_augment: bool = False, meta_drop_prob: float = 0.0):
        super().__init__(FoundationDataMaker, None, None)
        self.converter = converter
        self.tokenizer = converter.tokenizer
        self.aya_node = [0]
        # §9.2: aya_node と同順(index 揃え)の per-sample pattern_key(方向タグ真実源)。
        # index 0 は aya_node[0]=0 のダミーに対応する空文字。convert() でのみ追記する。
        self.aya_pattern = [""]
        self.min_measure = min_measure
        self.max_measure = max_measure
        self.additional_prompt = additional_prompt
        # True のとき、シーケンスブロックの入れ替え（並び替え）と削除を一切行わない。
        # ブロックは固定順 [SYSTEM, PAST_M, CONST_M, FUTURE_M] で 3 ブロック全て残す（決定的）。
        # ※ _build_melody_block 内の楽器順シャッフルはブロック内の別レイヤーなので影響しない。
        self.disable_block_augment = disable_block_augment
        # META(SYSTEM) 独立 Bernoulli ドロップ確率。
        # 【v1.4 裁定】A1 は p_drop=0(既定)に復帰。理由: META が prefix に無く後置される
        # サンプルでも、CONST 区間の学習目標は全確率の法則により Σ_m p(CONST|prefix,m)p(m|prefix)
        # = 真の meta-free 条件付き/周辺分布に厳密一致する。よって位置的自由だけで meta-free 方向は
        # 被覆済みで、ドロップは分布情報を足さない(v1.3 の CFG 同型根拠は数学的誤りとして撤回)。
        # 本フラグは E5 の ablation 変数(自由連続生成の終端率・自発 META 率への効果測定)専用。
        # disable_block_augment(A2)では常に無効。
        self.meta_drop_prob = meta_drop_prob
        self.stats = {
            "total_samples": 0,
            "deletion_count": {0: 0, 1: 0, 2: 0},
            "block_presence": {
                "PAST_M": 0,
                "CONST_M": 0,
                "FUTURE_M": 0,
            },
            "permutation_patterns": {},
            "skipped_windows": 0,
            "instrument_finish_reasons": {},
            "meta_dropped": 0,   # §9.0: META をドロップしたサンプル数(E0-2 のドロップ率算出用)
        }

        if converter.is_error:
            self.is_error = True
            self.error_reason = f"MIDIConverter でエラーが発生しています: {converter.error_reason}"
            self.seq_dict = {}
            return

        if not hasattr(converter, 'midi2seq') or converter.midi2seq is None:
            self.is_error = True
            self.error_reason = "converter.midi2seq が存在しません。MIDIConverter を use_midi2seq=True で生成してください。"
            self.seq_dict = {}
            return

        self.seq_dict = converter.midi2seq.aya_node.copy()

    # ---------------------------------------------------------
    # 保存
    # ---------------------------------------------------------

    def _get_out_dir(self, save_directory: str) -> str:
        subdir = hashlib.md5(self.converter.file_name.encode()).hexdigest()[0]
        return os.path.join(save_directory, subdir)

    def save(self, save_directory: str, save_stats: bool = False) -> Tuple[bool, str]:
        if not self.is_error:
            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                # §9.2: pattern_key を index 揃えで同梱(ローダは array{i} のみ読むため無害)
                if len(self.aya_pattern) == len(self.aya_node):
                    array_dict['pattern_keys'] = np.array(self.aya_pattern)
                out_dir = self._get_out_dir(save_directory)
                np.savez(os.path.join(out_dir, self.converter.file_name), **array_dict)
                if save_stats:
                    stats_serializable = {
                        "total_samples": self.stats["total_samples"],
                        "deletion_count": {str(k): v for k, v in self.stats["deletion_count"].items()},
                        "block_presence": self.stats["block_presence"],
                        "permutation_patterns": self.stats["permutation_patterns"],
                        "skipped_windows": self.stats["skipped_windows"],
                        "instrument_finish_reasons": self.stats["instrument_finish_reasons"],
                    }
                    with open(os.path.join(out_dir, self.converter.file_name + "_stats.json"), "w") as f:
                        json.dump(stats_serializable, f, indent=2)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

    def prepare_write_task(
        self, save_directory: str, save_stats: bool = False
    ) -> Optional[Tuple[str, str, dict, Optional[dict]]]:
        """
        書き込みに必要なデータを返す（実際の I/O は行わない）。
        単一ライタープロセスへのキュー投入用。
        戻り値: (out_dir, filename, array_dict, stats_dict_or_None)
                何も保存するものがなければ None。
        """
        if self.is_error:
            return None
        array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
        if len(array_dict) <= 1:
            return None
        # §9.2: pattern_key を index 揃えで同梱(ローダは array{i} のみ読むため無害)
        if len(self.aya_pattern) == len(self.aya_node):
            array_dict['pattern_keys'] = np.array(self.aya_pattern)
        out_dir = self._get_out_dir(save_directory)
        stats_data = None
        if save_stats and self.stats:
            stats_data = {
                "total_samples": self.stats["total_samples"],
                "deletion_count": {str(k): v for k, v in self.stats["deletion_count"].items()},
                "block_presence": self.stats["block_presence"],
                "permutation_patterns": self.stats["permutation_patterns"],
                "skipped_windows": self.stats["skipped_windows"],
                "instrument_finish_reasons": self.stats["instrument_finish_reasons"],
            }
        return out_dir, self.converter.file_name, array_dict, stats_data

    # ---------------------------------------------------------
    # ヘルパー: 制約チェック & 密度計算 (PreTrainDataMaker から流用)
    # ---------------------------------------------------------

    def _is_monophonic_sequence(self, sequence: np.ndarray) -> bool:
        """
        絶対位置(Position)ベースのトークン列に対して、単旋律チェックを行う。
        """
        s_range = self.tokenizer.get_length_tuple("s")
        p_range = self.tokenizer.get_length_tuple("p")
        sme_id = self.tokenizer.get("<SME>")

        current_notes_in_step = 0
        last_s_id = -1

        for token_id in sequence:
            if token_id == sme_id:
                current_notes_in_step = 0
                last_s_id = -1
            elif s_range[0] <= token_id <= s_range[1]:
                if token_id != last_s_id:
                    current_notes_in_step = 0
                    last_s_id = token_id
            elif p_range[0] <= token_id <= p_range[1]:
                current_notes_in_step += 1
                if current_notes_in_step > 1:
                    return False

        return True

    def _calculate_density_token(self, sequence: np.ndarray, program_name: str) -> Optional[int]:
        """
        シーケンスの音密度を計算し、NoteDenseトークンのIDを返す。
        閾値を超えた場合は None を返す。
        """
        rule = INSTRUMENT_RULES.get(program_name)
        if rule is None:
            raise ValueError(f"Undefined instrument rule: {program_name}")

        max_notes_per_measure = 45 if rule.is_polyphonic else 24

        sme_id = self.tokenizer.get("<SME>")
        measures = split_sequence_measure(sequence, 1, sme_id)

        s_range = self.tokenizer.get_length_tuple("s")
        s_min, s_max = s_range

        total_density_score = 0.0
        valid_measures_count = 0

        for measure in measures:
            note_count = np.sum((measure >= s_min) & (measure <= s_max))
            if note_count > max_notes_per_measure:
                return None

            density = note_count / max_notes_per_measure
            total_density_score += density
            valid_measures_count += 1

        if valid_measures_count == 0:
            return self.tokenizer.get("<NOTE_DENSE_1>")

        avg_density = total_density_score / valid_measures_count

        if avg_density <= 0:
            score = 1
        else:
            score = math.ceil(avg_density * 10)
            if score > 10:
                score = 10
            if score < 1:
                score = 1

        return self.tokenizer.get(f"<NOTE_DENSE_{score}>")

    def _check_instrument_constraint(self, program_name: str, sequence: np.ndarray) -> bool:
        """
        楽器ごとの制約を確認する。
        """
        if program_name not in INSTRUMENT_RULES:
            raise ValueError(
                f"Undefined instrument rule for program: {program_name}. "
                "Please update INSTRUMENT_RULES."
            )

        rule = INSTRUMENT_RULES[program_name]
        if rule.is_polyphonic:
            return True
        else:
            return self._is_monophonic_sequence(sequence)

    # ---------------------------------------------------------
    # ヘルパー: ブロック構築
    # ---------------------------------------------------------

    def _build_melody_block(
        self,
        marker_token: str,
        seqs: dict,
        programs: List[str],
    ) -> np.ndarray:
        """
        <MARKER> [<INST_X> ...seq... <ESEQ>]* <TAG_END> の形式でブロックを構築する。
        楽器順は呼び出しごとに独立にランダムシャッフルする。
        """
        s_range = self.tokenizer.get_length_tuple("s")
        shuffled = programs.copy()
        random.shuffle(shuffled)

        block = np.array([self.tokenizer.get(marker_token)], dtype=int)
        for prog in shuffled:
            seq = seqs[prog]
            # S トークンが1つもない（BLANK のみ）楽器はこのブロックから除外
            if not np.any(np.isin(seq, np.arange(s_range[0], s_range[1]))):
                continue
            framed = np.concatenate([
                np.array([self.tokenizer.get(f"<INST_{prog}>")], dtype=int),
                seq,
                np.array([self.tokenizer.get("<ESEQ>")], dtype=int),
            ])
            block = np.concatenate([block, framed])

        block = np.concatenate([block, np.array([self.tokenizer.get("<TAG_END>")], dtype=int)])
        return block

    def _increment_finish_reason(self, reason: str):
        """楽器終了の原因別カウンタを更新する。"""
        self.stats["instrument_finish_reasons"][reason] = (
            self.stats["instrument_finish_reasons"].get(reason, 0) + 1
        )

    # ---------------------------------------------------------
    # メイン変換ロジック
    # ---------------------------------------------------------

    def convert(self, *args, **kwargs):
        if self.is_error:
            return self.stats

        measure_token_id = self.tokenizer.get("<SME>")
        s_e = self.tokenizer.get_length_tuple("s")

        # seq_dict にキーが存在する有効なプログラムのみを抽出
        valid_programs = [p for p in self.converter.program_list if p in self.seq_dict]
        if not valid_programs:
            return

        # 各プログラムの <SME> インデックス配列を構築
        seq_inds = {}
        for program in valid_programs:
            seq = self.seq_dict[program]
            seq_inds[program] = np.where(seq == measure_token_id)[0]

        inst_finish_dict = {program: False for program in valid_programs}

        now_measure = 0

        while not all(inst_finish_dict.values()):
            # 3 区間の長さは常に max_measure 固定
            past_len = self.max_measure
            const_len = self.max_measure
            future_len = self.max_measure

            past_seqs = {}
            const_seqs = {}
            future_seqs = {}
            active_program_info = []  # (program_name, density_token_id)
            is_skip_window = False

            for program in valid_programs:
                if inst_finish_dict[program]:
                    continue

                inds = seq_inds[program]
                total_needed = past_len + const_len + future_len

                # 必要な小節数が足りなければこの楽器を終了
                if now_measure + total_needed >= len(inds):
                    inst_finish_dict[program] = True
                    self._increment_finish_reason("length_exceeded")
                    continue

                past_start = inds[now_measure]
                past_end = inds[now_measure + past_len]
                const_end = inds[now_measure + past_len + const_len]
                future_end = inds[now_measure + past_len + const_len + future_len]

                past_seq = self.seq_dict[program][past_start:past_end]
                const_seq = self.seq_dict[program][past_end:const_end]
                future_seq = self.seq_dict[program][const_end:future_end]

                # CONST 区間に音符がなければこの楽器をスキップ
                has_notes_in_const = np.any(np.isin(const_seq, np.arange(s_e[0], s_e[1])))
                if not has_notes_in_const:
                    continue

                # 3 区間すべてで単旋律制約チェック（違反 → この楽器のみスキップ）
                if not (
                    self._check_instrument_constraint(program, past_seq)
                    and self._check_instrument_constraint(program, const_seq)
                    and self._check_instrument_constraint(program, future_seq)
                ):
                    continue

                # 3 区間すべてで密度チェック（超過 → ウィンドウ全体スキップ）
                if self._calculate_density_token(past_seq, program) is None:
                    is_skip_window = True
                    break
                density_token = self._calculate_density_token(const_seq, program)
                if density_token is None:
                    is_skip_window = True
                    break
                if self._calculate_density_token(future_seq, program) is None:
                    is_skip_window = True
                    break

                past_seqs[program] = past_seq
                const_seqs[program] = const_seq
                future_seqs[program] = future_seq
                active_program_info.append((program, density_token))

            # スキップ判定
            if is_skip_window or len(active_program_info) == 0:
                self.stats["skipped_windows"] += 1
                if all(inst_finish_dict.values()):
                    break
                now_measure += past_len
                continue

            active_programs = [prog for prog, _ in active_program_info]

            # --- 各ブロックを構築 (楽器順は各ブロックで独立にシャッフル) ---
            past_block = self._build_melody_block("<PAST_M>", past_seqs, active_programs)
            const_block = self._build_melody_block("<CONST_M>", const_seqs, active_programs)
            future_block = self._build_melody_block("<FUTURE_M>", future_seqs, active_programs)

            # --- SYSTEM ブロックを構築（密度トークンあり）---
            # make_system_prompt の先頭トークンは <EOS>。
            # EOS は常に系列の先頭に固定するため、ここで分離する。
            system_prompt = self.converter.make_system_prompt(0, active_program_info)
            eos_token = np.array([system_prompt[0]], dtype=int)
            system_block = np.array(system_prompt[1:], dtype=int)

            music_block_names = ["PAST_M", "CONST_M", "FUTURE_M"]
            music_blocks_map = {
                "PAST_M": past_block,
                "CONST_M": const_block,
                "FUTURE_M": future_block,
            }

            if self.disable_block_augment:
                # --- 入れ替え・削除なしモード（決定的）---
                # 全ブロックを残し、固定順 [SYSTEM, PAST_M, CONST_M, FUTURE_M] で並べる。
                k = 0
                remaining_blocks = [
                    ("SYSTEM", system_block),
                    ("PAST_M", music_blocks_map["PAST_M"]),
                    ("CONST_M", music_blocks_map["CONST_M"]),
                    ("FUTURE_M", music_blocks_map["FUTURE_M"]),
                ]
            else:
                # --- 削除処理: k ∈ {0, 1, 2} 個のブロックを一様にランダム削除 ---
                k = random.choice([0, 1, 2])
                delete_names = random.sample(music_block_names, k)
                remaining_music = {
                    name: blk for name, blk in music_blocks_map.items()
                    if name not in delete_names
                }

                # --- 並び替え ---
                # PAST → FUTURE の時系列順を固定し、CONST だけ 3 択でランダム挿入する。
                # 3 択: PASTより前 / PASTとFUTUREの間 / FUTUREより後
                ordered_music = []
                if "PAST_M" in remaining_music:
                    ordered_music.append(("PAST_M", remaining_music["PAST_M"]))
                if "FUTURE_M" in remaining_music:
                    ordered_music.append(("FUTURE_M", remaining_music["FUTURE_M"]))
                if "CONST_M" in remaining_music:
                    # 0 〜 len(ordered_music) のいずれかに挿入
                    pos = random.randint(0, len(ordered_music))
                    ordered_music.insert(pos, ("CONST_M", remaining_music["CONST_M"]))

                # --- META(SYSTEM) 独立ドロップ (§9.0) ---
                # 音楽ブロック削除の「後」に Bernoulli(meta_drop_prob) で META を落とす。
                # 残す場合の挿入位置は従来どおり全体一様。音楽側の削除分布 k は不変。
                # 音楽ブロックは k<=2 のため必ず 1 つ以上残り、META を落としても空系列にならない。
                remaining_blocks = ordered_music.copy()
                if random.random() < self.meta_drop_prob:
                    self.stats["meta_dropped"] += 1
                else:
                    system_pos = random.randint(0, len(ordered_music))
                    remaining_blocks.insert(system_pos, ("SYSTEM", system_block))

            # --- 統計更新 ---
            self.stats["total_samples"] += 1
            self.stats["deletion_count"][k] += 1

            remaining_names = [name for name, _ in remaining_blocks]
            for block_name in ["PAST_M", "CONST_M", "FUTURE_M"]:
                if block_name in remaining_names:
                    self.stats["block_presence"][block_name] += 1

            pattern_key = ",".join(remaining_names)
            self.stats["permutation_patterns"][pattern_key] = (
                self.stats["permutation_patterns"].get(pattern_key, 0) + 1
            )

            # --- 連結して保存（EOS は必ず先頭）---
            sample = np.concatenate([eos_token] + [blk for _, blk in remaining_blocks])
            self.aya_node.append(sample)
            self.aya_pattern.append(pattern_key)  # §9.2: 方向タグ真実源(index 揃え)

            now_measure += past_len

        total_tokens = int(sum(len(arr) for arr in self.aya_node[1:]))
        self.stats["total_tokens"] = total_tokens
        avg_tokens = total_tokens // self.stats["total_samples"] if self.stats["total_samples"] > 0 else 0

        print(f"[FoundationDataMaker] 変換完了: {self.stats['total_samples']} サンプル生成")
        print(f"  総トークン数: {total_tokens}  平均トークン数/サンプル: {avg_tokens}")
        print(f"  削除分布: { {k: v for k, v in self.stats['deletion_count'].items()} }")
        print(f"  ブロック存在数: {self.stats['block_presence']}")
        print(f"  スキップウィンドウ数: {self.stats['skipped_windows']}")
        print(f"  並び替えパターン数: {len(self.stats['permutation_patterns'])}")
        return self.stats



class GenerationDataMaker(FoundationDataMaker):
    """生成SFTタスク (meta / meta_past / meta_future -> CONST) のシーケンス生成器。

    FoundationDataMaker(基盤事前学習用) の窓分割・ブロック構築ヘルパーを継承して使うが、
    責務は SFT 用の生成タスク列生成であり、事前学習(convert)とは分離する。
    """

    COT_PROB = 0.5  # meta_past/meta_future で CoT(<thinking>+フルmeta予測)を有効にする確率

    def _build_gen_field(self, seqs: dict, programs: List[str], cot_meta=None) -> np.ndarray:
        """生成ターゲット: <MGEN> [cot_meta] [<INST_X> seq <ESEQ>]* <TE> を構築する。
        loss_mask が <MGEN> 以降のみ損失計算するため、生成対象(CONST)はここに置く。
        cot_meta: CoT時、<MGEN>直後に予測させるフルmeta(<SYSTEM>..<TAG_END>)のトークン配列。"""
        s_range = self.tokenizer.get_length_tuple("s")
        parts = [np.array([self.tokenizer.get("<MGEN>")], dtype=int)]
        if cot_meta is not None:
            parts.append(np.asarray(cot_meta, dtype=int))  # CoT: フルmetaを先に予測してから生成
        for prog in programs:
            seq = seqs[prog]
            if not np.any(np.isin(seq, np.arange(s_range[0], s_range[1]))):
                continue
            parts.append(np.concatenate([
                np.array([self.tokenizer.get(f"<INST_{prog}>")], dtype=int),
                seq,
                np.array([self.tokenizer.get("<ESEQ>")], dtype=int),
            ]))
        parts.append(np.array([self.tokenizer.get("<TE>")], dtype=int))
        return np.concatenate(parts)

    def _build_partial_system(self, active_program_info, gmc_tok, genre_tokens, key_tok,
                              inc_density, inc_length, inc_genre, inc_key, thinking):
        """条件付きsystemタグ(部分meta)を構築。指定属性のみ含める。
        順序は make_system_prompt 準拠: <SYSTEM> (<INST>[<DENSE>])* [GMC] [genre] [key] [<thinking>] <TAG_END>"""
        blk = [self.tokenizer.get("<SYSTEM>")]
        for prog, dens in active_program_info:
            blk.append(self.tokenizer.get(f"<INST_{prog}>"))   # 楽器は構造上常に含める
            if inc_density and dens is not None:
                blk.append(int(dens))
        if inc_length and gmc_tok is not None:
            blk.append(int(gmc_tok))
        if inc_genre and genre_tokens:
            blk.extend(int(g) for g in genre_tokens)
        if inc_key and key_tok is not None:
            blk.append(int(key_tok))
        if thinking:
            blk.append(self.tokenizer.get("<thinking>"))       # systemタグ末尾にCoTトリガー
        blk.append(self.tokenizer.get("<TAG_END>"))
        return np.array(blk, dtype=int)

    def convert_generation_sft(self, tasks=("meta", "meta_past", "meta_future"),
                               genre_tokens=None, *args, **kwargs):
        """生成 SFT タスクのシーケンスを作る。各窓で指定タスクを emit する。
        genre_tokens: ジャンルのトークンIDリスト(任意)。メタの key トークンの直前に差し込む。

        - "meta"        : [SYSTEM(meta)]            <MGEN> CONST <TE>   (メタのみ -> CONST)
        - "meta_past"   : [SYSTEM] <PAST_M>..<TAG_END>   <MGEN> CONST <TE>
        - "meta_future" : [SYSTEM] <FUTURE_M>..<TAG_END> <MGEN> CONST <TE>
        - "infill"      : [SYSTEM] <PAST_M>..<TAG_END> <FUTURE_M>..<TAG_END> <MGEN> CONST <TE>
                          (PAST と FUTURE の両方を条件にその間の CONST を補完)
        - "inst_comp"   : [SYSTEM(target)] <CONST_M>[他楽器]..<TAG_END> <MGEN> [対象楽器のCONST] <TE>
                          (同一CONST窓の他楽器の旋律から、指定した1楽器の旋律を生成。≥2楽器の窓のみ)

        EOS は系列先頭固定。CONST(=生成対象)は <MGEN> 以降に置く(loss_mask対応)。
        ブロックの並び替え・削除は行わない(SFTは決定的)。
        infill / inst_comp も meta_past/future と同様に CoT(<thinking>+フルmeta予測)を確率的に付与する。
        """
        if self.is_error:
            return self.stats

        measure_token_id = self.tokenizer.get("<SME>")
        s_e = self.tokenizer.get_length_tuple("s")

        valid_programs = [p for p in self.converter.program_list if p in self.seq_dict]
        if not valid_programs:
            return self.stats

        seq_inds = {}
        for program in valid_programs:
            seq_inds[program] = np.where(self.seq_dict[program] == measure_token_id)[0]

        inst_finish_dict = {program: False for program in valid_programs}
        counts = {t: 0 for t in tasks}
        now_measure = 0

        while not all(inst_finish_dict.values()):
            # ★可変長: 過去/中央(生成対象)/未来 を各窓で独立にランダム化 (min..max 小節)。
            #   これにより GEN_MEASURE_COUNT(出力長) と 入力長(PAST/FUTURE) が 1..max で多様化し、
            #   「1小節だけ入力→N小節生成」のような可変入出力を学習できる。
            past_len = random.randint(self.min_measure, self.max_measure)
            const_len = random.randint(self.min_measure, self.max_measure)
            future_len = random.randint(self.min_measure, self.max_measure)
            past_seqs, const_seqs, future_seqs = {}, {}, {}
            active_program_info = []
            is_skip_window = False

            for program in valid_programs:
                if inst_finish_dict[program]:
                    continue
                inds = seq_inds[program]
                if now_measure + past_len + const_len + future_len >= len(inds):
                    inst_finish_dict[program] = True
                    continue

                past_start = inds[now_measure]
                past_end = inds[now_measure + past_len]
                const_end = inds[now_measure + past_len + const_len]
                future_end = inds[now_measure + past_len + const_len + future_len]
                past_seq = self.seq_dict[program][past_start:past_end]
                const_seq = self.seq_dict[program][past_end:const_end]
                future_seq = self.seq_dict[program][const_end:future_end]

                # CONST は生成対象なので音符必須
                if not np.any(np.isin(const_seq, np.arange(s_e[0], s_e[1]))):
                    continue
                if not (
                    self._check_instrument_constraint(program, past_seq)
                    and self._check_instrument_constraint(program, const_seq)
                    and self._check_instrument_constraint(program, future_seq)
                ):
                    continue
                if (self._calculate_density_token(past_seq, program) is None
                        or self._calculate_density_token(future_seq, program) is None):
                    is_skip_window = True
                    break
                density_token = self._calculate_density_token(const_seq, program)
                if density_token is None:
                    is_skip_window = True
                    break

                past_seqs[program] = past_seq
                const_seqs[program] = const_seq
                future_seqs[program] = future_seq
                active_program_info.append((program, density_token))

            if is_skip_window or len(active_program_info) == 0:
                if all(inst_finish_dict.values()):
                    break
                now_measure += past_len
                continue

            active_programs = [prog for prog, _ in active_program_info]

            # SYSTEM プロンプト(メタ)。生成小節数トークン + ジャンルを key の直前に付与。
            # (make_system_prompt は call_function の追加分を k_<key> の直前に置く)
            def _append_measure_count(p: list):
                p.append(self.tokenizer.get(f"<GEN_MEASURE_COUNT_{const_len}>"))
                if genre_tokens:
                    p.extend(genre_tokens)  # ジャンルを key の直前に差し込む

            system_prompt = self.converter.make_system_prompt(
                0, active_program_info, call_function=_append_measure_count)
            eos_token = np.array([system_prompt[0]], dtype=int)
            full_system_block = np.array(system_prompt[1:], dtype=int)  # フル <SYSTEM>..<TAG_END>

            # 部分meta構築用のパーツ(key/GMC)を抽出
            gmc_tok = self.tokenizer.get(f"<GEN_MEASURE_COUNT_{const_len}>")
            key_lo, key_hi = self.tokenizer.get_length_tuple("k")
            key_cands = [int(t) for t in full_system_block if key_lo <= int(t) < key_hi]
            key_tok = key_cands[0] if key_cands else None

            for task in tasks:
                if task == "meta":
                    # メタ→MIDI: フルmeta入力、CoTなし(従来通り)
                    sample = np.concatenate([
                        eos_token, full_system_block,
                        self._build_gen_field(const_seqs, active_programs),
                    ])
                elif task in ("meta_past", "meta_future"):
                    marker = "<PAST_M>" if task == "meta_past" else "<FUTURE_M>"
                    seqs = past_seqs if task == "meta_past" else future_seqs
                    cond_block = self._build_melody_block(marker, seqs, active_programs)
                    # CoT有無をランダム決定
                    cot = random.random() < self.COT_PROB
                    # 部分meta: 各属性をランダムに含める(楽器は常に含む)
                    partial_system = self._build_partial_system(
                        active_program_info, gmc_tok, genre_tokens, key_tok,
                        inc_density=random.random() < 0.5,
                        inc_length=random.random() < 0.5,
                        inc_genre=random.random() < 0.5,
                        inc_key=random.random() < 0.5,
                        thinking=cot,
                    )
                    # CoT時のみ <MGEN> 直後にフルmeta(=full_system_block)を予測させる
                    cot_meta = full_system_block if cot else None
                    gen_field = self._build_gen_field(const_seqs, active_programs, cot_meta=cot_meta)
                    sample = np.concatenate([eos_token, partial_system, cond_block, gen_field])
                    if cot:
                        counts["_cot"] = counts.get("_cot", 0) + 1
                elif task == "infill":
                    # PAST + FUTURE を条件に、その間の CONST を補完(全楽器)。
                    cond_past = self._build_melody_block("<PAST_M>", past_seqs, active_programs)
                    cond_future = self._build_melody_block("<FUTURE_M>", future_seqs, active_programs)
                    cot = random.random() < self.COT_PROB
                    partial_system = self._build_partial_system(
                        active_program_info, gmc_tok, genre_tokens, key_tok,
                        inc_density=random.random() < 0.5,
                        inc_length=random.random() < 0.5,
                        inc_genre=random.random() < 0.5,
                        inc_key=random.random() < 0.5,
                        thinking=cot,
                    )
                    cot_meta = full_system_block if cot else None
                    gen_field = self._build_gen_field(const_seqs, active_programs, cot_meta=cot_meta)
                    # 時系列順: PAST -> FUTURE
                    sample = np.concatenate([eos_token, partial_system, cond_past, cond_future, gen_field])
                    if cot:
                        counts["_cot"] = counts.get("_cot", 0) + 1
                elif task == "inst_comp":
                    # 同一CONST窓の「他楽器の旋律」を条件に、対象1楽器の旋律を生成。
                    # 2楽器以上ある窓でのみ成立(piano+sax なら 1↔1)。
                    if len(active_programs) < 2:
                        continue
                    target = random.choice(active_programs)
                    cond_programs = [p for p in active_programs if p != target]
                    target_info = [(p, d) for (p, d) in active_program_info if p == target]
                    cond_block = self._build_melody_block("<CONST_M>", const_seqs, cond_programs)
                    cot = random.random() < self.COT_PROB
                    partial_system = self._build_partial_system(
                        target_info, gmc_tok, genre_tokens, key_tok,
                        inc_density=random.random() < 0.5,
                        inc_length=random.random() < 0.5,
                        inc_genre=random.random() < 0.5,
                        inc_key=random.random() < 0.5,
                        thinking=cot,
                    )
                    # CoT時は対象楽器のフルmeta(全属性)を <MGEN> 直後に予測させる
                    cot_meta = self._build_partial_system(
                        target_info, gmc_tok, genre_tokens, key_tok,
                        inc_density=True, inc_length=True, inc_genre=True, inc_key=True,
                        thinking=False,
                    ) if cot else None
                    gen_field = self._build_gen_field(const_seqs, [target], cot_meta=cot_meta)
                    sample = np.concatenate([eos_token, partial_system, cond_block, gen_field])
                    if cot:
                        counts["_cot"] = counts.get("_cot", 0) + 1
                else:
                    continue
                self.aya_node.append(sample)
                counts[task] += 1

            now_measure += past_len

        self.stats["sft_generation_counts"] = counts
        total = sum(counts.values())
        total_tokens = int(sum(len(arr) for arr in self.aya_node[1:]))
        self.stats["total_samples"] = total
        self.stats["total_tokens"] = total_tokens
        print(f"[FoundationDataMaker.SFT] 生成タスク完了: {total} サンプル ({counts})  総トークン {total_tokens}")
        return self.stats


class AnalysisDataMaker(FoundationDataMaker):
    """分析SFTタスク (MIDI -> メタ予測。生成の逆) のシーケンス生成器。

    構造: <EOS> <CONST_M>[旋律]<TAG_END> <TRIGGER> [予測対象(答え)] <TE>
      <META>  : フルmeta(楽器別密度/長さ/ジャンル/キー)
      <KEY>   : k_<key>
      <DENCE> : <INST_x><NOTE_DENSE_x> (楽器別)
      <GENRE> : <GENRE_xxx>
      <LENGTH>: <GEN_MEASURE_COUNT_n>
    loss_mask が <TRIGGER> 以降のみ損失計算するため、予測対象はトリガー後に置く。
    """

    def convert_analysis_sft(self, tasks=("meta", "key", "dence", "genre", "length"),
                             genre_tokens=None, *args, **kwargs):
        if self.is_error:
            return self.stats
        measure_token_id = self.tokenizer.get("<SME>")
        s_e = self.tokenizer.get_length_tuple("s")
        valid_programs = [p for p in self.converter.program_list if p in self.seq_dict]
        if not valid_programs:
            return self.stats
        seq_inds = {p: np.where(self.seq_dict[p] == measure_token_id)[0] for p in valid_programs}
        inst_finish_dict = {p: False for p in valid_programs}
        counts = {t: 0 for t in tasks}
        now_measure = 0

        while not all(inst_finish_dict.values()):
            const_len = random.randint(self.min_measure, self.max_measure)  # 分析対象の長さも可変
            const_seqs = {}
            active_program_info = []
            is_skip = False
            for program in valid_programs:
                if inst_finish_dict[program]:
                    continue
                inds = seq_inds[program]
                if now_measure + const_len >= len(inds):
                    inst_finish_dict[program] = True
                    continue
                cs = self.seq_dict[program][inds[now_measure]:inds[now_measure + const_len]]
                if not np.any(np.isin(cs, np.arange(s_e[0], s_e[1]))):
                    continue
                if not self._check_instrument_constraint(program, cs):
                    continue
                dt = self._calculate_density_token(cs, program)
                if dt is None:
                    is_skip = True
                    break
                const_seqs[program] = cs
                active_program_info.append((program, dt))

            if is_skip or len(active_program_info) == 0:
                if all(inst_finish_dict.values()):
                    break
                now_measure += const_len
                continue

            active_programs = [p for p, _ in active_program_info]
            eos = np.array([self.tokenizer.get("<EOS>")], dtype=int)
            # 分析対象の旋律ブロック: <CONST_M>[<INST> seq <ESEQ>]*<TAG_END>
            music_block = self._build_melody_block("<CONST_M>", const_seqs, active_programs)

            # フルmeta(make_system_promptのSYSTEM..TAG_END中身)からkey/各トークンを取得
            def _amc(p):
                p.append(self.tokenizer.get(f"<GEN_MEASURE_COUNT_{const_len}>"))
                if genre_tokens:
                    p.extend(genre_tokens)
            full_sys = self.converter.make_system_prompt(0, active_program_info, call_function=_amc)
            full_meta = np.array(full_sys[1:], dtype=int)  # <SYSTEM>..<TAG_END>(=メタ分析の答え)
            key_lo, key_hi = self.tokenizer.get_length_tuple("k")
            key_toks = [int(t) for t in full_meta if key_lo <= int(t) < key_hi]

            # 各分析タスクの (トリガー, 予測対象) を構築
            def target_for(task):
                if task == "meta":
                    return self.tokenizer.get("<META>"), list(full_meta)
                if task == "key":
                    return self.tokenizer.get("<KEY>"), key_toks
                if task == "dence":
                    tgt = []
                    for prog, dt in active_program_info:
                        tgt += [self.tokenizer.get(f"<INST_{prog}>"), int(dt)]
                    return self.tokenizer.get("<DENCE>"), tgt
                if task == "genre":
                    return self.tokenizer.get("<GENRE>"), list(genre_tokens or [])
                if task == "length":
                    return self.tokenizer.get("<LENGTH>"), [self.tokenizer.get(f"<GEN_MEASURE_COUNT_{const_len}>")]
                return None, None

            te = np.array([self.tokenizer.get("<TE>")], dtype=int)
            for task in tasks:
                trig, tgt = target_for(task)
                if trig is None or not tgt:
                    continue
                sample = np.concatenate([eos, music_block,
                                         np.array([trig] + list(tgt), dtype=int), te])
                self.aya_node.append(sample)
                counts[task] += 1

            now_measure += const_len

        self.stats["sft_analysis_counts"] = counts
        total = sum(counts.values())
        total_tokens = int(sum(len(arr) for arr in self.aya_node[1:]))
        self.stats["total_samples"] = total
        self.stats["total_tokens"] = total_tokens
        print(f"[AnalysisDataMaker] 分析タスク完了: {total} サンプル ({counts})  総トークン {total_tokens}")
        return self.stats
