from typing import List, Any

import mido
import numpy as np
from numpy import ndarray
from AGSM.convert import ConvTempo
from pretty_midi.pretty_midi import PrettyMIDI, Instrument, Note, TimeSignature
from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from .token import Token
from .tokenizer import Tokenizer

T = TypeVar("T")
class _AbstractMidiToAyaNode(ABC):

    def __init__(self, instance: Generic[T],  tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None,):
        self.program_list = program_list
        self.directory = directory
        self.file_name = file_name
        self.is_error = False
        self.instance = instance
        self.token_converter: List[Token] = tokenizer.music_token_list
        self.error_reason: str = "不明なエラー"
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

    def __call__(self, *args, **kwargs):
        self.convert()

    def get_midi_change_scale(self, scale_up_key):
        midi = PrettyMIDI()

        for ins in self.midi_data.instruments:
            ins: Instrument = ins
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

        return midi

    def expansion_midi(self) -> List[Any]:
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
        tempo = 0
        for i in range(len(self.tempo_change_time)):
            if start >= self.tempo_change_time[i]:
                tempo = self.tempo[i]

        return tempo

    @abstractmethod
    def save(self, save_directory: str) -> [bool, str]:
        pass

    @abstractmethod
    def convert(self):
        pass

class MIDIToSequence(_AbstractMidiToAyaNode):

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None):
        super().__init__(MIDIToSequence, tokenizer, directory, file_name, program_list, midi_data)
        self.aya_node = [0]

    def convert(self):
        """
        以下のステップでMidiが変換される
        1. テンポ固定
            120テンポに変換される。
        2. 指定した楽器があるかを検索する
            インスタンス化した際に、指定したprogram_listに乗っ取り、該当する楽器があるかを検索する
        3. ノートを変換する。
            self.ct_aya_nodeにインストを渡すことで、MORTMに適した形に変換する。

        また、変換したインストゥルメントは以下のような配列構造になる。

        1_music = [
                    [1 inst's 60s clip],
                        ...
                    [1 inst's 60s clip],
                    [2 inst's 60s clip],
                        ...
                    [n inst's 60s clip]
                    ]
        :return:なし。
        """
        if not self.is_error:
            program_count = 0

            for inst in self.midi_data.instruments:
                inst: Instrument = inst
                if not inst.is_drum and inst.program in self.program_list:
                    aya_node_inst = self.ct_aya_node(inst)
                    self.aya_node = self.aya_node + aya_node_inst
                    program_count += 1

            if program_count == 0:
                self.is_error = True
                self.error_reason = f"{self.directory}/{self.file_name}に、欲しい楽器がありませんでした。"

    def ct_aya_node(self, inst: Instrument) -> list:

        """
        Instrumentsから1音ずつ取り出し、Tokenizerで変換する。
        clip = [<START>, S, P, V, D, H, S, P, V, D, H ...<END>]
        さらに60秒ごとにスプリットし、以下のような配列構造を作る。
        aya_node_inst = [clip_1, clip_2 ... clip_n]

        よって、二次元配列のndarrayを返す。
        :param inst: インストゥルメント
        :return: 60秒にクリッピングされた旋律の配列(2次元)
        """

        clip = np.array([], dtype=int)
        aya_node_inst = []
        back_note = None

        clip_count = 0

        sorted_notes = sorted(inst.notes, key=lambda notes: notes.start)

        for note in sorted_notes:
            note: Note = note

            tempo = self.get_tempo(note.start)

            for conv in self.token_converter:
                conv: Token = conv

                token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo)

                if token is not None:
                    if conv.token_type == "<SME>":
                        clip_count += 1
                    if clip_count >= 8:
                        aya_node_inst = self.marge_clip(clip, aya_node_inst)
                        clip = np.array([], dtype=int)
                        back_note = None
                        clip_count = 0

                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

            back_note = note

        if len(clip) > 4:
            aya_node_inst = self.marge_clip(clip, aya_node_inst)
        return aya_node_inst

    def marge_clip(self, clip, aya_node_inst):
        aya_node_inst.append(clip)

        return aya_node_inst

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


class MidiToSequece(_AbstractMidiToAyaNode):
    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None):
        super().__init__(MidiToSequece, tokenizer, directory, file_name, program_list, midi_data=midi_data)
        self.sequence_dict: dict = dict()
        self.sequence_count = 0
        self.context = 8

    def add_sequence(self, seq_pare: dict):
        self.sequence_dict[f'array_{self.sequence_count}'] = seq_pare
        self.sequence_count += 1

    def save(self, save_directory: str) -> [bool, str]:
        if not self.is_error:
            if len(self.sequence_dict) > 2:
                np.savez(save_directory + "/" + self.file_name, **self.sequence_dict)
                return True, "正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

    def convert(self):
        if not self.is_error:
            program_count = 0

            for inst in self.midi_data.instruments:
                inst: Instrument = inst
                if not inst.is_drum and inst.program in self.program_list:
                    self.ct_aya_node(inst)
                    program_count += 1

            if program_count == 0:
                self.is_error = True
                self.error_reason = f"{self.directory}/{self.file_name}に、欲しい楽器がありませんでした。"

    def ct_aya_node(self, inst: Instrument):
        back_note: Note | None = None
        key = np.array([], dtype=int)
        value = np.array([], dtype=int)
        sorted_notes = sorted(inst.notes, key=lambda notes: notes.start)

        for note in sorted_notes:
            note: Note = note

            for conv in self.token_converter:
                token: str = conv(inst=inst, back_notes=back_note, note=note, tempo=self.get_tempo(note.start))
                if token is not None:
                    if token == "<BLANK>":
                        token_id = self.tokenizer.get(token)
                        value = np.append(value, token_id)
                        self.step(key, value)
                        key = np.array([], dtype=int)
                        value = np.array([self.tokenizer.get("<SME>")], dtype=int)
                        back_note = None
                    elif token == "<SME>":
                        new_key, new_value = self.step(key, value)
                        key = new_key
                        value = new_value
                        token_id = self.tokenizer.get(token)
                        value = np.append(value, token_id)
                    else:
                        token_id = self.tokenizer.get(token)
                        value = np.append(value, token_id)

            back_note = note
        if len(value) >= 10:
            self.step(key, value)


    def step(self, key:ndarray, value:ndarray):
        if np.isin(1, value):
            key = value
            value = np.array([], dtype=int)
        elif np.sum(key==3) > self.context - 1:
            new_value = np.append(value, self.tokenizer.get("<ESEQ>"))
            new_value = np.insert(new_value, 0, self.tokenizer.get("<CONTINUE>"))
            clip = {
                'key': key.copy(),
                'value': new_value.copy()
            }
            self.add_sequence(clip)
            indices = np.where(key==3)[0][1]
            key = key[indices:]
            key = np.concatenate((key, value))
            value = np.array([], dtype=int)
        else:
            new_value = np.append(value, self.tokenizer.get("<ESEQ>"))
            new_value = np.insert(new_value, 0, self.tokenizer.get("<CONTINUE>"))
            clip = {
                'key': key.copy(),
                'value': new_value.copy()
            }
            self.add_sequence(clip)
            key = np.concatenate((key, value))
            value = np.array([], dtype=int)

        return key, value