import os
import random
from typing import List, Any, Optional, Tuple, Dict

import numpy as np
import torch
import torchaudio
from fontTools.ttLib.tables.S__i_l_f import instre
from pretty_midi.pretty_midi import PrettyMIDI, Instrument, Note, TimeSignature
from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from midi2audio import FluidSynth
import soundfile as sf

from mortm.train.custom_token import Human
from mortm.train.custom_token import Token, ShiftTimeContainer, ChordToken, MeasureToken, Blank
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, TO_TOKEN
from mortm.train.utils.chord_midi import ChordMidi, Chord
from mortm.utils.key import get_key_dict
from mortm.utils.tag import extract_tagged_sequences_batch
T = TypeVar("T")


def split_sequence_measure(sequence: np.ndarray, measure_length: int, measure_token_id: int) -> List[np.ndarray]:
    """
    文脈Mの法則に基づき、指定された小節数ごとにシーケンスを分割する関数。

    Args:
        sequence (np.ndarray): <SME>を含むトークンIDの配列 (文脈M)。
        measure_length (int): 分割する単位となる小節数 (例: 4小節ごとに分割)。
        measure_token_id (int): <SME> (小節境界) のトークンID。

    Returns:
        List[np.ndarray]: 指定小節数ごとに分割されたシーケンスのリスト。
    """

    # 1. シーケンス内の <SME> トークンのインデックスを全て取得する
    # これが各小節の開始位置に相当します
    sme_indices = np.where(sequence == measure_token_id)[0]

    # <SME> が存在しない場合は分割できないため、元のシーケンスをリストに入れて返す
    if len(sme_indices) == 0:
        return [sequence]

    chunks = []

    # 2. measure_length ごとにインデックスをステップさせて分割を行う
    # range(開始位置, 終了位置, ステップ数)
    for i in range(0, len(sme_indices), measure_length):

        # 現在のチャンクの開始位置 (<SME>のインデックス)
        start_index = sme_indices[i]

        # 次のチャンクの開始位置を計算 (現在のインデックス + 指定小節数)
        next_chunk_sme_index_pos = i + measure_length

        if next_chunk_sme_index_pos < len(sme_indices):
            # 次の区切りとなる <SME> が存在する場合
            # 現在の <SME> から、次の区切りの <SME> の直前までを取得する
            end_index = sme_indices[next_chunk_sme_index_pos]
            chunk = sequence[start_index : end_index]
        else:
            # 次の区切りが存在しない場合 (最後のチャンク)
            # 現在の <SME> からシーケンスの最後までを取得する
            chunk = sequence[start_index : ]

        chunks.append(chunk)

    return chunks


def convert_str_int_program(program_list: List[str]) -> List[Tuple[int]]:
    pl = []
    for p in program_list:
        if p == "SAX":
            pl.append((65, 66, 67, 68))
        elif p == "PIANO":
            pl.append((0, 1,2,3,4,5,6))

    return pl

def conv_spectro(waveform, sample_rate, n_fft, hop_length, n_mels):
    """
    Converts a waveform into a mel spectrogram.

    Args:
        waveform (Tensor): Input waveform tensor of shape [1, time].
        sample_rate (int): Sampling rate of the waveform.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for the STFT.
        n_mels (int): Number of mel bands.

    Returns:
        Tensor: Log-scaled mel spectrogram of shape [n_mels, T].
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log1p(mel_spec)

    return mel_spec


class _AbstractConverter(ABC):
    """
    Abstract base class for converters.

    Attributes:
        instance (Generic[T]): Instance of the child class.
        directory (str): Directory path for input files.
        file_name (str | List[str]): Name(s) of the file(s).
        is_error (bool): Indicates if an error occurred.
        error_reason (str): Reason for the error.
    """
    def __init__(self, instance: Generic[T], directory: str, file_name: str | List[str]):
        self.instance = instance
        self.directory = directory
        self.file_name = file_name
        self.is_error = False
        self.error_reason: str = "不明なエラー"

    def __call__(self, *args, **kwargs):
        self.convert(args, kwargs)

    @abstractmethod
    def save(self, save_directory: str) -> [bool, str]:
        """
        Abstract method to save the converted data.

        Args:
            save_directory (str): Directory to save the output.

        Returns:
            Tuple[bool, str]: Success status and message.
        """
        pass

    @abstractmethod
    def convert(self, *args, **kwargs):
        """
        Abstract method to perform the conversion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass


class _AbstractMidiConverter(_AbstractConverter):
    def __init__(self, instance: Generic[T], tokenizer: Tokenizer, directory: str, file_name: str, program_list,
                 key=None, inst_list_with_name=None, midi_data=None):
        '''
        MIDIをトークンのシーケンスに変換するクラスの抽象クラス
        :param instance: 子クラスのインスタンス
        :param tokenizer: 変換するトークナイザー
        :param directory: MIDIのディレクトリパス
        :param file_name: ディレクトリにあるMIDIのファイル名
        :param program_list: MIDIの楽器のプログラムリスト
        :param midi_data: PrettyMIDIのインスタンス(Optinal)
        '''
        super().__init__(instance, directory, file_name)
        self.token_converter: List[Token] = tokenizer.music_token_list
        self.tokenizer = tokenizer
        if midi_data is not None:
            self.midi_data: PrettyMIDI = midi_data
        else:
            try:
                self.midi_data: PrettyMIDI = PrettyMIDI(f"{directory}/{file_name}")
                time_s = self.midi_data.time_signature_changes
                for t in time_s:
                    t_s: TimeSignature = t
                    if not (t_s.numerator == 4 and t_s.denominator == 4):
                        self.is_error = True
                        self.error_reason = "旋律に変拍子が混じっていました。"
                        break
            except Exception:
                self.is_error = True
                self.error_reason = "MIDIのロードができませんでした。"

        if not self.is_error:
            self.tempo_change_time, self.tempo = self.midi_data.get_tempo_changes()
            if inst_list_with_name is not None:
                self.inst_list_with_name = inst_list_with_name
                self.inst_list = [inst for inst, name in self.inst_list_with_name]
            else:
                self.inst_list_with_name, found_program_names = self.is_in_correct_inst(program_list)
                self.inst_list = [inst for inst, name in self.inst_list_with_name]
                self.program_list = found_program_names

            if len(self.inst_list) == 0 and not self.is_error:
                self.is_error = True
                self.error_reason = "欲しい楽器がMIDIにありません。"
            elif not self.is_error and key is None:
                if len(self.inst_list) > 0:
                    self.key = get_key_dict(os.path.join(directory, file_name), window_measures=12)
                    self.key_dict = self.key['segments']
                print(f"{self.key_dict} <- キー情報")
            elif key is not None:
                if "major" in key:
                    self.key = f"{key.split(' major')[0]}M"
                elif "minor" in key:
                    self.key = f"{key.split(' minor')[0]}m"
                else:
                    self.key = key

    def is_in_correct_inst(self, program_list: List[str]) -> Tuple[List[Tuple[Instrument, str]], List[str]]:
        int_program_groups: List[Tuple[int]] = convert_str_int_program(program_list)

        inst_by_program = {}
        for inst in self.midi_data.instruments:
            if not inst.is_drum:
                inst_by_program.setdefault(inst.program, []).append(inst)

        found_instruments_with_name: List[Tuple[Instrument, str]] = []
        found_program_names: List[str] = []
        added_instrument_ids: set = set()

        # カテゴリごとにループ
        for name, group in zip(program_list, int_program_groups):
            candidate_instrument = None
            
            # カテゴリ内のプログラム番号をループして、最初に見つかった楽器を候補とする
            for program_num in group:
                if program_num in inst_by_program:
                    for inst in inst_by_program[program_num]:
                        # まだ全体で追加されていない楽器かチェック
                        if id(inst) not in added_instrument_ids:
                            candidate_instrument = inst
                            break  # このプログラム番号での検索を終了
                if candidate_instrument:
                    break  # このカテゴリでの検索を終了
            
            # このカテゴリで楽器が見つかった場合、結果に追加
            if candidate_instrument:
                candidate_instrument.notes.sort(key=lambda note: note.start)
                found_instruments_with_name.append((candidate_instrument, name))
                added_instrument_ids.add(id(candidate_instrument))
                if name not in found_program_names:
                    found_program_names.append(name)

        return found_instruments_with_name, found_program_names

    def get_midi_change_scale(self, scale_up_key):
        '''
        MIDIの音程を変更する。
        :param scale_up_key: いくつ音程を上げるか
        :return:
        '''
        midi = PrettyMIDI()

        for ins in self.midi_data.instruments:
            ins: Instrument = ins
            if not ins.is_drum:
                new_inst = Instrument(program=ins.program)
                for note in ins.notes:
                    note: Note = note
                    pitch = note.pitch + scale_up_key
                    if pitch > 127:
                        pitch -= 12
                    if pitch < 0:
                        pitch += 12

                    start = note.start
                    end = note.end
                    velo = note.velocity
                    new_note = Note(pitch=pitch, velocity=velo, start=start, end=end)
                    new_inst.notes.append(new_note)
                midi.instruments.append(new_inst)
            else:
                midi.instruments.append(ins)

        return midi

    def expansion_midi(self) -> List[Any]:
        '''
        データ拡張する関数。
        MIDIデータを全スケール分に拡張する。
        :return: MIDIのリスト
        '''
        converts = []
        key = 5
        if not self.is_error:
            for i in range(key):
                midi = self.get_midi_change_scale(i + 1)
                converts.append(self.instance(self.tokenizer, self.directory, f"{self.file_name}_scale_{i + 1}",
                                              self.program_list, midi_data=midi))
                midi = self.get_midi_change_scale(-(i + 1))
                converts.append(self.instance(self.tokenizer, self.directory, f"{self.file_name}_scale_{-(i + 1)}",
                                              self.program_list, midi_data=midi))

        return converts

    def get_tempo(self, start: float):
        '''
        MIDIのテンポを取得する関数
        :param start:
        :return:
        '''
        tempo = 0
        for i in range(len(self.tempo_change_time)):
            if start >= self.tempo_change_time[i]:
                tempo = self.tempo[i]

        return tempo


class _AbstractAudioConverter(_AbstractConverter):
    def __init__(self, instance: Generic[T], directory: str, file_name: str | List[str]):
        super().__init__(instance, directory, file_name)
        if not isinstance(file_name, list):
                self.waveform, self.sample_rate = self.load_wav(f"{directory}{file_name}")


    def load_wav(self, path:str):
        try:
            waveform, sample_rate = torchaudio.load(path, format="wav")
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return waveform, sample_rate
        except FileNotFoundError | RuntimeError as e:
            self.is_error = True
            self.error_reason = "このwavは読み込むことができない。"


class MIDI2Seq(_AbstractMidiConverter):
    '''
    MIDIをトークンのシーケンスに変換するクラス
    '''

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str,  program_list: List[str],
                 key=None, inst_list_with_name=None,
                 midi_data=None):
        super().__init__(MIDI2Seq, tokenizer, directory, file_name, program_list, key=key, inst_list_with_name=inst_list_with_name, midi_data=midi_data)
        self.aya_node = {}



    def convert(self):
        """
        以下のステップでMidiが変換される
        1. Instrumentsから楽器を取り出す。
        2. 楽器の音を1音ずつ取り出し、Tokenizerで変換する。
        3. clip = [<START>, S, P, D, S, P, D, ...<END>]
        :return:なし
        """
        if not self.is_error:
            print(f"Programs: {[i.program for i in self.midi_data.instruments]}")
            container = [ShiftTimeContainer(0, self.tempo[0]) for _ in self.inst_list]
            note_counts = [0 for _ in self.inst_list]

            for i, (inst, program) in enumerate(self.inst_list_with_name):

                inst_clip, note_count, is_finish = self.convert_inst(program, inst, container[i], note_counts[i])
                note_counts[i] = note_count
                s_e = self.tokenizer.get_length_tuple("s")

                if np.any(np.isin(inst_clip, [s for s in range(s_e[0], s_e[1])])):
                    self.aya_node[program] = inst_clip


    def convert_inst(self, program, inst: Instrument, container, note_count) -> np.ndarray | int:
        clip = []
        clip_count = 0
        back_note: Optional[Note] = None
        is_first = True

        while clip_count < 999:
            if note_count >= len(inst.notes):
                return np.array(clip), note_count, True
            note: Note = inst.notes[note_count]
            tempo = self.get_tempo(note.start)
            container.tempo = tempo
            is_continue_note = False
            #if is_first:
            #    self.measure_time_sort(note.start, container)
            #    is_first = False

            for conv in self.token_converter:
                conv: Token = conv

                if not isinstance(conv, ChordToken):
                    token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo, container=container)

                    if token is not None:
                        if conv.token_type == "<SME>":
                            clip_count += 1

                        if clip_count >= 999:
                            return np.array(clip), note_count, False

                        token_id = self.tokenizer.get(token)
                        clip.append(token_id)
                        progressed = True

                        if conv.token_type == "<BLANK>":
                            # 小節またぎ待ち（次ループで同一ノート再試行）
                            is_continue_note = True
                            break

                    if container.is_error:
                        self.is_error = True
                        self.error_reason = "MIDIの変換中にエラーが発生しました。"
                        break
            if not is_continue_note:
                note_count += 1
                back_note = note
        return np.array(clip), note_count, False


    def measure_time_sort(self, note_time, container: ShiftTimeContainer):
        """
        Calculates the start time of the most recent measure before a given note_time.
        This method accounts for tempo changes within the MIDI file.

        Args:
            note_time (float): The time of the note in seconds.
            container (ShiftTimeContainer): A container to store the calculated measure time.
        """
        tempo_change_times, tempos = self.tempo_change_time, self.tempo

        # 【修正】テンポ情報が無いときのみ 120 BPM を仮定
        if tempos is None or len(tempos) == 0:
            tempo = 120.0
            # For 4/4 time, a measure has 4 beats.
            measure_duration = (60.0 / tempo) * 4.0
            if measure_duration > 0:
                num_measures = int(note_time / measure_duration)
                container.measure_start_time = num_measures * measure_duration
            else:
                container.measure_start_time = 0.0
            return

        current_measure_start = 0.0
        last_measure_start = 0.0
        tempo_idx = 0

        # Iterate through measures until we pass the note_time
        while current_measure_start <= note_time:
            last_measure_start = current_measure_start

            # Find the correct tempo for the beginning of the current measure.
            # Advance tempo_idx if the current measure starts after the next tempo change.
            while (tempo_idx + 1 < len(tempo_change_times) and
                   last_measure_start >= tempo_change_times[tempo_idx + 1]):
                tempo_idx += 1

            current_tempo = tempos[tempo_idx]
            # Assuming 4/4 time signature (4 beats per measure)
            measure_beats = 4.0

            if current_tempo <= 0:  # Avoid division by zero or negative tempo
                break

            sec_per_beat = 60.0 / current_tempo

            # Calculate the duration of the measure if there were no tempo changes.
            measure_duration_sans_tempo_change = sec_per_beat * measure_beats
            next_measure_start = last_measure_start + measure_duration_sans_tempo_change

            # Check if the measure crosses a tempo change boundary.
            next_tempo_change_idx = tempo_idx + 1
            if (next_tempo_change_idx < len(tempo_change_times) and
                    next_measure_start > tempo_change_times[next_tempo_change_idx]):

                next_tempo_change_time = tempo_change_times[next_tempo_change_idx]

                # Ensure the tempo change happens *within* this measure, not at the very start.
                if next_tempo_change_time > last_measure_start:
                    # Part 1: Time spent in the current tempo
                    time_in_old_tempo = next_tempo_change_time - last_measure_start
                    beats_in_old_tempo = time_in_old_tempo / sec_per_beat

                    remaining_beats = measure_beats - beats_in_old_tempo

                    if remaining_beats > 1e-9:  # Use epsilon for float comparison
                        # Part 2: Time spent in the new tempo
                        new_tempo = tempos[next_tempo_change_idx]
                        if new_tempo > 0:
                            new_sec_per_beat = 60.0 / new_tempo
                            time_in_new_tempo = remaining_beats * new_sec_per_beat

                            # The actual duration of this measure is the sum of the two parts.
                            actual_measure_duration = time_in_old_tempo + time_in_new_tempo
                            next_measure_start = last_measure_start + actual_measure_duration

            current_measure_start = next_measure_start

        container.measure_start_time = last_measure_start

    def marge_clip(self, clip, aya_node_inst):
        aya_node_inst.append(clip)
        return aya_node_inst

    def save(self, save_directory: str) -> Tuple[bool, str]:
        if not self.is_error:
            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                np.savez(save_directory + "/" + self.file_name, **array_dict)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason


class MIDIConverter(_AbstractMidiConverter):
    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list: List[str],
                 use_midi2seq=True,
                 all_chords: Optional[List[str]]=None, all_chord_timestamps: Optional[List[float]]=None,
                 key=None):
        super().__init__(MIDIConverter, tokenizer, directory, file_name, program_list, key=key)
        if not self.is_error:
            self.midi2seq = MIDI2Seq(tokenizer, directory, file_name, program_list, key=self.key, inst_list_with_name=self.inst_list_with_name, midi_data=self.midi_data) if use_midi2seq else None

    def make_system_prompt(self, time, found_list):
        prompt = [self.tokenizer.get("<EOS>"),
                  self.tokenizer.get("<SYSTEM>")]

        for p in found_list:
            prompt.append(self.tokenizer.get(f"<INST_{p}>"))
        prompt.append(self.tokenizer.get(f"k_{self.get_key(time) if isinstance(self.key, dict) else self.key}"))
        prompt.append(self.tokenizer.get("<TAG_END>"))
        return np.array(prompt)

    def convert(self, *args, **kwargs):
        if self.is_error:
            return

        if self.midi2seq is not None:
            self.midi2seq.convert()

    def save(self, save_directory: str) -> Tuple[bool, str]:
        raise NotImplementedError("これ単体では保存できません。　DataMakerに受け渡してください")

    def get_key(self, time):
        for k in self.key_dict:
            if k['start_time'] <= time < k['end_time']:
                tonic = k['tonic']
                if k['key_name'] == "Unknown":
                    return k['key_name']
                if '-' in tonic:
                    tonic = tonic.replace('-', 'b')
                mode = 'M' if k['mode'] == 'major' else 'm'
                return f"{tonic}{mode}"

        if self.key_dict[-1]['end_time'] >= time:
            k = self.key_dict[-1]
            tonic = k['tonic']
            if k['key_name'] == "Unknown":
                return k['key_name']
            if '-' in tonic:
                tonic = tonic.replace('-', 'b')
            mode = 'M' if k['mode'] == 'major' else 'm'
            return f"{tonic}{mode}"

        return "Unknown"


class PreTrainDataMaker(_AbstractConverter):
    def __init__(self, converter: MIDIConverter, split_measure=12):
        super().__init__(PreTrainDataMaker, None, None)
        self.converter = converter
        self.tokenizer = converter.tokenizer
        self.aya_node = [0]
        self.seq_dict = converter.midi2seq.aya_node
        self.split_measure = split_measure

    def save(self, save_directory: str) -> Tuple[bool, str]:
        if not self.is_error:
            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                np.savez(save_directory + "/" + self.converter.file_name, **array_dict)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

    def convert(self, *args, **kwargs):
        for program in self.converter.program_list:
            seq = self.seq_dict[program]
            measure_token_id = self.tokenizer.get("<SME>")
            split_seqs = split_sequence_measure(seq, self.split_measure, measure_token_id)
            self.seq_dict[program] = split_seqs
        inst_finish_dict = {program: False for program in self.converter.program_list}
        count = 0
        while not all(inst_finish_dict.values()):
            clip = np.array([], dtype=int)
            active_program_list = []
            for program in self.converter.program_list:
                if inst_finish_dict[program]:
                    continue
                inst = self.seq_dict[program][count]
                s_e = self.tokenizer.get_length_tuple("s")

                if np.any(np.isin(inst, [s for s in range(s_e[0], s_e[1])])):
                    inst: np.ndarray
                    inst = np.concatenate([np.array([self.tokenizer.get(f"<INST_{program}>")]), inst, np.array([self.tokenizer.get("<ESEQ>")])])
                    clip = np.concatenate([clip, inst])
                    active_program_list.append(program)
                if count + 1 >= len(self.seq_dict[program]):
                    inst_finish_dict[program] = True
            if len(active_program_list) == 0:
                count += 1
                continue

            prompt = self.converter.make_system_prompt(0, active_program_list)
            clip = np.concatenate([prompt, np.array([self.tokenizer.get("<MGEN>")]), clip, np.array([self.tokenizer.get("<TE>")])])
            self.aya_node = self.aya_node + [clip]
            count += 1

class OmegaMIDI2SeqWithChord(MIDI2Seq):
    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list: List[str],
                 all_chords: List[str], all_chord_timestamps: List[float], midi_data=None,
                 key=None, split_measure=12, is_include_special_token=True):
        super().__init__(tokenizer, directory, file_name, program_list, key, midi_data, split_measure, is_include_special_token)
        self.chords = ChordMidi(all_chords, all_chord_timestamps)

    def convert_inst(self, program, inst: Instrument, container, note_count) -> np.ndarray | int:
        clip = np.array([self.tokenizer.get(f"<INST_{program}>")], dtype=int)
        clip_count = 0
        back_note: Optional[Note] = None
        is_first = True

        while clip_count < 999:
            if note_count >= len(inst.notes):
                return clip, note_count, True
            note: Note = inst.notes[note_count]
            tempo = self.get_tempo(note.start)
            container.tempo = tempo
            is_continue_note = False
            #if is_first:
            #    self.measure_time_sort(note.start, container)
            #    is_first = False

            for conv in self.token_converter:
                conv: Token = conv

                if not isinstance(conv, ChordToken):
                    token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo, container=container)

                    if token is not None:
                        if conv.token_type == "<SME>":
                            clip_count += 1

                        if clip_count >= 999:
                            if self.is_include_special_token:
                                clip = np.append(clip, self.tokenizer.get("<ESEQ>"))
                            return clip, note_count, False

                        token_id = self.tokenizer.get(token)
                        clip = np.append(clip, token_id)
                        progressed = True

                        if conv.token_type == "<BLANK>":
                            # 小節またぎ待ち（次ループで同一ノート再試行）
                            is_continue_note = True
                            break

                    if container.is_error:
                        self.is_error = True
                        self.error_reason = "MIDIの変換中にエラーが発生しました。"
                        break
                else:
                    token = conv(note=note,
                                 chords=self.chords, container=container)
                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

            if not is_continue_note:
                note_count += 1
                back_note = note
        return clip, note_count, False


class MetaData2Chord(_AbstractConverter):
    def save(self, save_directory: str) -> [bool, str]:
        if not self.is_error:

            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                np.savez(save_directory + "/" + self.file_name, **array_dict)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

    def make_system_prompt(self, clip: np.ndarray, key: str):
        prompt = [self.tokenizer.get("<EOS>"), self.tokenizer.get("<SYSTEM>")]

        prompt.append(self.tokenizer.get(f"k_{key}"))
        prompt.append(self.tokenizer.get("<TAG_END>"))
        prompt.append(self.tokenizer.get("<CGEN>"))
        cap = np.concatenate((np.array(prompt), clip))
        return cap

    def convert(self, *args, **kwargs):
        token_converter: List[Token] = self.tokenizer.music_token_list
        shift_time_container = ShiftTimeContainer(0, self.tempo)
        shift_time_container.is_code_mode = True
        back_chord: Optional[Chord] = None
        aya_node_split = []
        clip = np.array([], dtype=int)
        if self.is_include_special_token:
            clip = self.make_system_prompt(clip, self.key)
        clip_count = 0

        self.chords.sort(self.chords[0].time_stamp)
        chord_count = 0

        while chord_count < len(self.chords):
            c: Chord = self.chords[chord_count]
            is_continue_note = False
            for conv in token_converter:
                token = None
                if isinstance(conv, ChordToken):
                    token = conv(note=Note(pitch=0, start=c.time_stamp, end=c.time_stamp, velocity=100),
                                 chords=self.chords, container=shift_time_container)
                if isinstance(conv, MeasureToken):
                    token = conv(note=Note(pitch=0, start=c.time_stamp, end=c.time_stamp, velocity=100),
                                 back_notes=Note(pitch=0, start=back_chord.time_stamp, end=back_chord.time_stamp, velocity=100) if back_chord else None,
                                 container=shift_time_container, tempo=shift_time_container.tempo)
                    clip_count += 1
                if isinstance(conv, Blank):
                    token = conv(note=Note(pitch=0, start=c.time_stamp, end=c.time_stamp, velocity=100),
                                 back_notes=Note(pitch=0, start=back_chord.time_stamp, end=back_chord.time_stamp, velocity=100) if back_chord else None,
                                 container=shift_time_container, tempo=shift_time_container.tempo)

                if token is not None:

                    if clip_count >= self.split_measure:
                        clip = np.append(clip, self.tokenizer.get("<ESEQ>"))
                        clip = np.append(clip, self.tokenizer.get("<TE>"))
                        aya_node_split.append(clip)
                        clip = np.array([], dtype=int)
                        if self.is_include_special_token:
                            clip = self.make_system_prompt(clip, self.key)
                        back_chord = None
                        clip_count = 0
                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

                    if conv.token_type == "<BLANK>":
                        is_continue_note = True
                        break

            if not is_continue_note:
                chord_count += 1
                back_chord = c
        if len(clip) > 10:
            aya_node_split.append(clip)
        self.aya_node = self.aya_node + aya_node_split




    def __init__(self, tokenizer: Tokenizer, key: str, all_chords: List[str], all_chord_timestamps: List[float], tempo,
                directory: str, file_name: str | List[str], split_measure=12, is_include_special_token=True):
        super().__init__(MetaData2Chord, directory, file_name)
        self.aya_node = [-1]
        self.is_include_special_token = is_include_special_token
        self.tempo = tempo
        self.tokenizer = tokenizer
        self.split_measure = split_measure
        self.chords = ChordMidi(all_chords, all_chord_timestamps)

        if "major" in key:
            self.key = f"{key.split(' major')[0]}M"
        elif "minor" in key:
            self.key = f"{key.split(' minor')[0]}m"
        else:
            self.key = None

        print(self.key)


class SeqWithChord2Chord(_AbstractConverter):
    def save(self, save_directory: str) -> [bool, str]:
        pass

    def convert(self, *args, **kwargs):
        # トークンの範囲を取得
        shift_time = self.tokenizer.get_length_tuple("s")
        chord_root = self.tokenizer.get_length_tuple("CR")
        chord_quarter = self.tokenizer.get_length_tuple("CQ")
        chord_base = self.tokenizer.get_length_tuple("CB")

        # 【追加修正】取得した範囲が None でないかチェック (デバッグ用)
        if not self._check_tuple("s", shift_time) or \
                not self._check_tuple("CR", chord_root) or \
                not self._check_tuple("CQ", chord_quarter) or \
                not self._check_tuple("CB", chord_base):
            self.is_error = True
            return

        blank = self.tokenizer.get("<BLANK>")
        sme = self.tokenizer.get("<SME>")

        for s in self.base:
            if not self.is_error:
                # 配列自体が None でないかチェック
                if s is None:
                    continue

                try:
                    # マスク処理 (ここで NoneType エラーが起きていた)
                    in_shift = (s >= shift_time[0]) & (s <= shift_time[1])
                    in_chord = (s >= chord_root[0]) & (s <= chord_root[1])
                    in_quarter = (s >= chord_quarter[0]) & (s <= chord_quarter[1])
                    in_base = (s >= chord_base[0]) & (s <= chord_base[1])

                    bs = (s == blank) | (s == sme)
                    mask = (in_shift | in_chord | in_quarter | in_base | bs)
                    filter_seq = s[mask]

                    tokens = []
                    self.tokenizer.mode(to=TO_MUSIC)
                    for id in filter_seq:
                        tokens.append(self.tokenizer.rev_get(id))

                    # 変換して格納
                    self.aya_node = self.aya_node + [self._ct_seq(tokens)]

                except TypeError as e:
                    print(f"\033[31m[Error in SeqWithChord2Chord] {e}\033[0m")
                    print(f"Debug Info -> shift_time: {shift_time}, s type: {type(s)}")
                    self.is_error = True

    def _check_tuple(self, name, val):
        """範囲タプルが有効か確認するヘルパー関数"""
        if val is None or len(val) < 2 or val[0] is None or val[1] is None:
            print(f"\033[31m[Critical Error] Tokenizer definition missing for '{name}'. Got: {val}\033[0m")
            return False
        return True

    def _ct_seq(self, s):
        self.tokenizer.mode(to=TO_TOKEN)
        aya_node_list = []
        cash = 0
        blank_trigger = False
        for token in s:
            token: str
            if "s_" in token:
                cash += int(token.split("_")[1])
            if "CR_" in token:
                if cash >= 96:
                    print(f"Warning: {cash} is too long. It will be cut off.")
                    cash = 95
                new_s = self.tokenizer.get(f"s_{cash}")
                aya_node_list.append(new_s)
                aya_node_list.append(self.tokenizer.get(token))
                cash = 0
            if "CB_" in token or "CQ_" in token or "<BLANK>" == token:
                aya_node_list.append(self.tokenizer.get(token))
                blank_trigger = False
            if token == "<SME>":
                cash = 0
                if blank_trigger:
                    aya_node_list.append(self.tokenizer.get("<BLANK>"))
                    aya_node_list.append(self.tokenizer.get("<SME>"))
                else:
                    aya_node_list.append(self.tokenizer.get("<SME>"))
                blank_trigger = True
        if blank_trigger:
            aya_node_list.append(self.tokenizer.get("<BLANK>"))
        aya_node_list.append(self.tokenizer.get("<ESEQ>"))
        return np.array(aya_node_list, dtype=int)

    def __init__(self, tokenizer: Tokenizer, aya_node, ):
        super().__init__(SeqWithChord2Chord, None, None)
        self.base = aya_node[1:] if len(aya_node) > 1 else []
        self.is_error = len(self.base) == 0
        self.tokenizer = tokenizer
        self.tokenizer.mode()
        self.aya_node = [0]


class PackSeq:
    def __init__(self, directory, file_list):
        self.directory = directory
        self.file_list: list = file_list
        self.seq = [0]


    def convert(self):
        count = 0
        for file in self.file_list:
            seq = np.load(f"{self.directory}/{file}")
            for i in range(len(seq) - 1):
                s = seq[f'array{i + 1}']
                self.seq.append(s)
            count += 1
            print(f"\r 一つのパックに纏めています。。。。{count}/{len(self.file_list)}", end="")

    def save(self, dire, filename):
        array_dict = {f'array{i}': arr for i, arr in enumerate(self.seq)}
        if len(array_dict) > 1:
            np.savez(dire + "/ "+ filename, **array_dict)
            return True, "処理が正常に終了しました。"
        else:
            return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        pass

class Midi2Audio(_AbstractMidiConverter):

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, fluid_base: FluidSynth, split_time=None):
        super().__init__(Midi2Audio, tokenizer, directory, file_name, program_list)
        self.split_time = split_time
        self.is_split = split_time is not None
        self.fluid_base: FluidSynth = fluid_base

    def save(self, save_directory: str) -> [bool, str]:
        try:
            if not self.is_error:
                if not self.is_split:
                    self.fluid_base.midi_to_audio(f"{self.directory}/{self.file_name}", f"{save_directory}/{self.file_name}.wav")
                    return True, "変換が完了しました。"
            else:
                return False, "謎のエラーが発生しました。"
        except Exception as e:
            return False, "謎のエラーが発生しました。"

    def convert(self):
        if self.is_split:
            pass


class Audio2MelSpectrogramALL(_AbstractAudioConverter):

    def __init__(
            self,
            directory: str,
            file_name: str,
            n_fft: int = 1024,
            hop_length: int = 256,
            n_mels: int = 80,
            split_time: Optional[float] = None,
    ):
        super().__init__(Audio2MelSpectrogramALL, directory, file_name)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split_time = split_time
        self.comp: List[torch.Tensor] = []

    def save(self, save_directory: str) -> Tuple[bool, str]:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, f"{self.file_name}.pt")
        try:
            torch.save(self.comp, path)
            return True, f"保存に成功しました: {path}"
        except Exception as e:
            return False, f"保存に失敗しました: {e}"

    def convert(self):
        # 1) soundfile で読み込み（always_2d=True で [time, ch] 出力）
        full_path = os.path.join(self.directory, self.file_name)
        wav_np, sr = sf.read(full_path, always_2d=True)

        # 2) NumPy→Tensor, かつ [ch, time] に transpose
        wav_np = wav_np.T.astype("float32")       # shape: (ch, time)
        waveform = torch.from_numpy(wav_np)

        # 3) モノラル化
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # → [1, time]

        # 4) 分割長サンプル数の決定（必ず split_time 秒ごと）
        if self.split_time:
            seg_len = int(self.split_time * sr)
            total = waveform.shape[1]
            num_segments = (total + seg_len - 1) // seg_len  # ceil
        else:
            seg_len = waveform.shape[1]
            num_segments = 1

        # 5) メル変換器を一度だけ生成
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # 6) 各セグメントを切り出し、最後は無音でパディング
        self.comp = []
        for i in range(num_segments):
            start = i * seg_len
            end = start + seg_len
            if end <= waveform.shape[1]:
                seg = waveform[:, start:end]
            else:
                # 残り部分 + 無音パディング
                rest = waveform[:, start:]
                pad_len = end - waveform.shape[1]
                pad = torch.zeros((waveform.shape[0], pad_len), dtype=waveform.dtype)
                seg = torch.cat([rest, pad], dim=1)

            # mel: [1, n_mels, T]
            mel = mel_tf(seg)
            # squeeze → [n_mels, T]
            mel = mel.squeeze(0)
            # log1p
            logmel = torch.log1p(mel)
            self.comp.append(logmel)

        # デバッグ: 最初のセグメント形状を表示
#        print(f"Segment count: {len(self.comp)}, each shape: {self.comp[0].shape}")

class PareAudio2PareMelSpectrogram(_AbstractAudioConverter):

    def __init__(self, directory: str, src: str, tgt: str,  n_fft = 1024, hop_length = 256, n_mels = 80):
        super().__init__(PareAudio2PareMelSpectrogram, directory, src)
        self.tgt_file_name = tgt
        self.tgt_wave, self.tgt_sample = self.load_wav(f"{directory}/{tgt}")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.comp = dict()

    def convert(self):
        src_spec = conv_spectro(self.waveform, self.sample_rate, self.n_fft, self.hop_length, self.n_mels)
        tgt_spec = conv_spectro(self.tgt_wave, self.tgt_sample, self.n_fft, self.hop_length, self.n_mels)

        self.comp["src"] = src_spec
        self.comp["tgt"] = tgt_spec


    def save(self, save_directory: str) -> [bool, str]:
        import os
        os.makedirs(save_directory, exist_ok=True)

        # ファイル名は、srcファイル名に基づいて保存（拡張子を除く）
        base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        save_path = os.path.join(save_directory, f"{base_name}_pair.pt")

        try:
            torch.save(self.comp, save_path)
            return True, f"保存に成功しました: {save_path}"
        except Exception as e:
            return False, f"保存に失敗しました: {str(e)}"
