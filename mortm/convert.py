from typing import List, Any

import mido
import numpy as np
from AGSM.convert import ConvTempo
from pretty_midi.pretty_midi import PrettyMIDI, Instrument, Note, TimeSignature
from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from . import constants
from .aya_node import Token
from .tokenizer import Tokenizer

T = TypeVar("T")
class _AbstractMidiToAyaNode(ABC):

    def __init__(self, instance: Generic[T],  tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None,):
        self.program_list = program_list
        self.directory = directory
        self.file_name = file_name
        self.is_error = False
        self.instance = instance
        self.token_converter: List[Token] = tokenizer.token_list
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

class MidiToAyaNode(_AbstractMidiToAyaNode):

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None):
        super().__init__(MidiToAyaNode, tokenizer, directory, file_name, program_list, midi_data)
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
                    #print(f"Instrument Number:{inst.program}")
                    aya_node_inst = self.ct_aya_node(inst)

                    self.aya_node = self.aya_node + aya_node_inst
                    #print(self.aya_node)
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

        clip_time: float = 0
        split_count: int = 1

        sorted_notes = sorted(inst.notes, key=lambda notes: notes.start)

        for note in sorted_notes:
            note: Note = note

            tempo = self.get_tempo(note.start)
            measure_time = (60 / tempo) * 4

            if clip_time >= measure_time * 8 * split_count:
                aya_node_inst = self.marge_clip(clip, aya_node_inst)

                clip = np.array([], dtype=int)
                back_note = None
                split_count += 1

            if back_note is None:
                clip = np.append(clip, self.tokenizer.get(constants.START_SEQ_TOKEN))

            for conv in self.token_converter:
                conv: Token = conv

                token = conv(back_notes=back_note, note=note, tempo=tempo)
                if token is not None:
                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

            back_note = note
            clip_time = note.start

        if len(clip) > 50:
            clip = np.append(clip, self.tokenizer.get(constants.END_SEQ_TOKEN))
            aya_node_inst = self.marge_clip(clip, aya_node_inst)
        return aya_node_inst

    def marge_clip(self, clip, aya_node_inst):
        #clip = np.append(clip, self.tokenizer.get(constants.END_SEQ_TOKEN))
        aya_node_inst.append(clip)

        return aya_node_inst

    def save(self, save_directory: str) -> [bool, str]:
        if not self.is_error:
            #print(f"Result shape is:{self.aya_node.shape}")

            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                np.savez(save_directory + "/" + self.file_name, **array_dict)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

    def ct_tempo(self):
        try:
            ct = ConvTempo(directory=self.directory + "/" + self.file_name, midi_data=self.midi_data, change_tempo=120)
            ct.convert()
            self.midi_data = ct.midi_data
        except OSError or IndexError or ValueError or mido.midifiles.meta.KeySignatureError or EOFError or KeyError or ZeroDivisionError:
            self.error_reason = f"{self.directory}/{self.file_name}でテンポ変換中にエラーが発生。処理を中断します。"
            self.is_error = True


class MidiToAyaNode_TGT(_AbstractMidiToAyaNode):

    def save(self, save_directory: str) -> [bool, str]:
        if not self.is_error:
            #print(f"Result shape is:{self.aya_node.shape}")

            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            if len(array_dict) > 1:
                np.savez(save_directory + "/" + self.file_name, **array_dict)
                return True, "処理が正常に終了しました。"
            else:
                return False, "オブジェクトが何らかの理由で見つかりませんでした。"
        else:
            return False, self.error_reason

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None):
        super().__init__(MidiToAyaNode_TGT, tokenizer, directory, file_name, program_list, midi_data)
        self.aya_node: List[dict] = []

    def convert(self):
        if not self.is_error:
            program_count = 0

            for inst in self.midi_data.instruments:
                inst: Instrument = inst
                if not inst.is_drum and inst.program in self.program_list:
                    #print(f"Instrument Number:{inst.program}")
                    aya_node_inst = self.ct_aya_node(inst)

                    self.aya_node = self.aya_node + aya_node_inst
                    #print(self.aya_node)
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
        clip_dict: dict = dict()
        aya_node_inst = np.array([], dtype=object)
        is_seq = True
        back_note = None

        clip_time: float = 0
        split_count: int = 1

        sorted_notes = sorted(inst.notes, key=lambda notes: notes.start)

        for note in sorted_notes:
            note: Note = note

            tempo = self.get_tempo(note.start)
            measure_time = (60 / tempo) * 4

            if clip_time >= measure_time * 4 * split_count:
                if is_seq:
                    clip_dict['src'] = clip.tolist()
                else:
                    clip_dict['tgt'] = clip.tolist()
                    aya_node_inst = np.append(aya_node_inst, clip_dict)

                clip_dict = dict()
                clip = np.array([], dtype=int)
                is_seq = not is_seq
                split_count += 1

            if back_note is None:
                clip = np.append(clip, self.tokenizer.get(constants.START_SEQ_TOKEN))

            for conv in self.token_converter:
                conv: Token = conv

                token = conv(back_notes=back_note, note=note, tempo=tempo)
                if token is not None:
                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

            back_note = note
            clip_time = note.start


        return aya_node_inst.tolist()

    def marge_clip(self, clip, aya_node_inst):
        #clip = np.append(clip, self.tokenizer.get(constants.END_SEQ_TOKEN))
        aya_node_inst.append(clip)

        return aya_node_inst
