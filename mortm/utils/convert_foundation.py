import json
import math
import random
from typing import List, Optional, Tuple

import numpy as np

from mortm.utils.convert import (
    _AbstractConverter,
    MIDIConverter,
    INSTRUMENT_RULES,
    split_sequence_measure,
)


class FoundationDataMaker(_AbstractConverter):
    """
    分析と生成が両方できる音楽基盤モデル向けの事前学習データ生成クラス。

    楽曲を Past / Const / Future の 3 区間に分割し、各区間をブロックとして
    並び替え・削除することで、任意のブロック集合を条件として残りを予測する
    能力を獲得させる。
    """

    def __init__(self, converter: MIDIConverter, min_measure=1, max_measure=8, additional_prompt=None):
        super().__init__(FoundationDataMaker, None, None)
        self.converter = converter
        self.tokenizer = converter.tokenizer
        self.aya_node = [0]
        self.min_measure = min_measure
        self.max_measure = max_measure
        self.additional_prompt = additional_prompt
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

    def save(self, save_directory: str) -> Tuple[bool, str]:
        if not self.is_error:
            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                np.savez(save_directory + "/" + self.converter.file_name, **array_dict)
                # JSON は整数キーを文字列に変換して保存
                stats_serializable = {
                    "total_samples": self.stats["total_samples"],
                    "deletion_count": {str(k): v for k, v in self.stats["deletion_count"].items()},
                    "block_presence": self.stats["block_presence"],
                    "permutation_patterns": self.stats["permutation_patterns"],
                    "skipped_windows": self.stats["skipped_windows"],
                    "instrument_finish_reasons": self.stats["instrument_finish_reasons"],
                }
                with open(save_directory + "/" + self.converter.file_name + "_stats.json", "w") as f:
                    json.dump(stats_serializable, f, indent=2)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

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

            # --- 削除処理: k ∈ {0, 1, 2} 個のブロックを一様にランダム削除 ---
            k = random.choice([0, 1, 2])
            music_block_names = ["PAST_M", "CONST_M", "FUTURE_M"]
            music_blocks_map = {
                "PAST_M": past_block,
                "CONST_M": const_block,
                "FUTURE_M": future_block,
            }
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

            # SYSTEM は全ブロックの中でランダムな位置に挿入
            system_pos = random.randint(0, len(ordered_music))
            remaining_blocks = ordered_music.copy()
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
