import os
import random
from typing import List, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio
from pretty_midi.pretty_midi import PrettyMIDI, Instrument, Note, TimeSignature
from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from midi2audio import FluidSynth
import soundfile as sf

from mortm.train.custom_token import Token, ShiftTimeContainer, ChordToken, MeasureToken, Blank
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, TO_TOKEN
from mortm.train.utils.chord_midi import ChordMidi, Chord

T = TypeVar("T")


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
                 midi_data=None):
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
        self.program_list = program_list
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

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None, split_measure=12, is_include_special_token = True):
        super().__init__(MIDI2Seq, tokenizer, directory, file_name, program_list, midi_data)
        self.aya_node = [0]
        self.split_measure = split_measure
        self.is_include_special_token = is_include_special_token

    def convert(self):
        """
        以下のステップでMidiが変換される
        1. Instrumentsから楽器を取り出す。
        2. 楽器の音を1音ずつ取り出し、Tokenizerで変換する。
        3. clip = [<START>, S, P, D, S, P, D, ...<END>]
        :return:なし
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

        clip = np.array([], dtype=int)
        if self.is_include_special_token:
            clip = np.append(clip, self.tokenizer.get("<MGEN>"))
        aya_node_inst = []
        back_note = None

        clip_count = 0

        sorted_notes = sorted(inst.notes, key=lambda notes: notes.start)
        shift_time_container = ShiftTimeContainer(0, 0)
        note_count = 0
        while note_count < len(sorted_notes):
            note: Note = sorted_notes[note_count]

            tempo = self.get_tempo(note.start)
            shift_time_container.tempo = tempo

            for conv in self.token_converter:
                conv: Token = conv

                if not isinstance(conv, ChordToken):
                    token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo, container=shift_time_container)

                    if token is not None:
                        if conv.token_type == "<SME>":
                            clip_count += 1
                        if clip_count >= self.split_measure:
                            clip = np.append(clip, self.tokenizer.get("<ESEQ>"))
                            aya_node_inst = self.marge_clip(clip, aya_node_inst)
                            clip = np.array([], dtype=int)
                            if self.is_include_special_token:
                                clip = np.append(clip, self.tokenizer.get("<MGEN>"))
                            back_note = None
                            clip_count = 0

                        token_id = self.tokenizer.get(token)
                        clip = np.append(clip, token_id)
                        if conv.token_type == "<BLANK>":
                            break
                    if shift_time_container.is_error:
                        self.is_error = True
                        self.error_reason = "MIDIの変換中にエラーが発生しました。"
                        break
            back_note = note
            if not shift_time_container.shift_measure:
                note_count += 1

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


class Midi2SeqWithChord(_AbstractMidiConverter):

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name, key: str, all_chords: List[str], all_chord_timestamps: List[float],
                 program_list, split_measure=12, is_include_special_token=True):
        super().__init__(Midi2SeqWithChord, tokenizer, directory, file_name, program_list)
        self.aya_node = [0]
        if "major" in key:
            self.key = f"{key.split(' major')[0]}M"
        elif "minor" in key:
            self.key = f"{key.split(' minor')[0]}m"
        else:
            self.key = None
        self.split_measure = split_measure
        self.chords = ChordMidi(all_chords, all_chord_timestamps)
        self.is_include_special_token = is_include_special_token


    def convert(self, *args, **kwargs):
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

        clip = np.array([], dtype=int)
        if self.is_include_special_token:
            clip = np.append(clip, self.tokenizer.get("<MGEN>"))
            clip = np.append(clip, self.tokenizer.get(f"k_{self.key}"))
        aya_node_inst = []
        back_note = None

        clip_count = 0

        sorted_notes = sorted(inst.notes, key=lambda notes: notes.start)
        shift_time_container = ShiftTimeContainer(0, 0)
        note_count = 0
        while note_count < len(sorted_notes):
            note: Note = sorted_notes[note_count]

            tempo = self.get_tempo(note.start)
            shift_time_container.tempo = tempo

            for conv in self.token_converter:
                conv: Token = conv

                if isinstance(conv, ChordToken):
                    conv: ChordToken
                    token = conv(note=note, chords=self.chords, container=shift_time_container)
                else:
                    token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo, container=shift_time_container)

                if token is not None:
                    if conv.token_type == "<SME>":
                        clip_count += 1
                    if clip_count >= self.split_measure:
                        clip = np.append(clip, self.tokenizer.get("<ESEQ>"))
                        aya_node_inst = self.marge_clip(clip, aya_node_inst)
                        clip = np.array([], dtype=int)
                        if self.is_include_special_token:
                            clip = np.append(clip, self.tokenizer.get("<MGEN>"))
                            clip = np.append(clip, self.tokenizer.get(f"k_{self.key}"))
                        back_note = None
                        clip_count = 0

                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)
                    if conv.token_type == "<BLANK>":
                        break
                if shift_time_container.is_error:
                    self.is_error = True
                    self.error_reason = "MIDIの変換中にエラーが発生しました。"
                    break
            back_note = note
            if not shift_time_container.shift_measure:
                note_count += 1

        if len(clip) > 4:
            aya_node_inst = self.marge_clip(clip, aya_node_inst)

        self.chords.reset()
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

    def convert(self, *args, **kwargs):
        token_converter: List[Token] = self.tokenizer.music_token_list
        shift_time_container = ShiftTimeContainer(0, self.tempo)
        shift_time_container.is_code_mode = True
        back_chord: Optional[Chord] = None
        aya_node_split = []
        clip = np.array([], dtype=int)
        if self.is_include_special_token:
            clip = np.append(clip, self.tokenizer.get("<CGEN>"))
            clip = np.append(clip, self.tokenizer.get(f"k_{self.key}"))
        clip_count = 0

        self.chords.sort(self.chords[0].time_stamp)
        chord_count = 0
        while chord_count < len(self.chords):
            c: Chord = self.chords[chord_count]
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
                        aya_node_split.append(clip)
                        clip = np.array([], dtype=int)
                        if self.is_include_special_token:
                            clip = np.append(clip, self.tokenizer.get("<CGEN>"))
                            clip = np.append(clip, self.tokenizer.get(f"k_{self.key}"))
                        back_chord = None
                        clip_count = 0
                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

                    if conv.token_type == "<BLANK>":
                        break

            back_chord = c
            if not shift_time_container.shift_measure:
                chord_count += 1
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


class MIDI2TaskSeq(_AbstractMidiConverter):
    def __init__(self, tokenizer: Tokenizer, system: dict, directory: str, file_name: str, program_list, split_measure=8, out_measure=4):
        super().__init__(MIDI2TaskSeq, tokenizer, directory, file_name, program_list)
        self.aya_node = [0]
        self.system = system
        self.out_measure = out_measure
        self.prompt_max_measure = split_measure

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

    def convert(self, *args, **kwargs):
        if not self.is_error:
            program_count = 0

            for inst in self.midi_data.instruments:
                inst: Instrument = inst
                if not inst.is_drum and inst.program in self.program_list:
                    aya_node_inst = self.ct_inst2seq(inst)
                    if aya_node_inst is None:
                        break
                    self.aya_node = self.aya_node + aya_node_inst
                    program_count += 1
                    break

            if program_count == 0:
                self.is_error = True
                self.error_reason = f"{self.directory}/{self.file_name}に、欲しい楽器がありませんでした。"

    def ct_inst2seq(self, inst: Instrument) -> list:
        aya_node_inst = []
        melody_clip = MIDI2Seq(tokenizer=self.tokenizer, directory=self.directory, file_name=self.file_name,is_include_special_token=False,
                               program_list=self.program_list, midi_data=self.midi_data, split_measure=999)
        melody_clip.convert()
        melody_with_chord_clip = Midi2SeqWithChord(self.tokenizer, self.directory, self.file_name, is_include_special_token=False,
                                                   key=self.system["key"], all_chords=self.system["all_chords"],
                                                   all_chord_timestamps= self.system["all_chords_timestamps"],program_list=self.program_list, split_measure=999)
        melody_with_chord_clip.convert()
        chord_clip = SeqWithChord2Chord(self.tokenizer,melody_with_chord_clip.aya_node)
        chord_clip.convert()

        if melody_clip.is_error and melody_with_chord_clip.is_error and chord_clip.is_error:
            self.is_error = True
            self.error_reason = "MIDIの変換中にエラーが発生しました。"
            return None
        melody_all: np.ndarray = melody_clip.aya_node[1]
        melody_all_ind = np.where(melody_all == self.tokenizer.get("<SME>"))[0]

        if len(chord_clip.aya_node) == 1:
            self.is_error = True
            self.error_reason = "コード進行の情報がありませんでした。"
            print(chord_clip.aya_node)
            return None
        chord_all = chord_clip.aya_node[1]
        chord_all_ind = np.where(chord_all == self.tokenizer.get("<SME>"))[0]

        melody_with_chord_all = melody_with_chord_clip.aya_node[1]
        melody_with_chord_all_ind = np.where(melody_with_chord_all == self.tokenizer.get("<SME>"))[0]
        back_ind = 0

        print(len(melody_all_ind), len(chord_all_ind), len(melody_with_chord_all_ind))
        is_running = back_ind + self.prompt_max_measure + self.out_measure < len(melody_all_ind)
        self.tokenizer.mode(to=TO_TOKEN)

        while is_running:
            prompt_measure = random.randint(back_ind + 1, back_ind + self.prompt_max_measure)
            #print(f"Test Seq MIDI2Seq: {melody_all[melody_all_ind[back_ind]: melody_all_ind[prompt_measure]]}")
            #print(f"Test Seq With Chord: {melody_with_chord_all[melody_with_chord_all_ind[back_ind]: melody_with_chord_all_ind[prompt_measure]]}")
            #print(f"Test Chord: {chord_all[chord_all_ind[back_ind]: chord_all_ind[prompt_measure]]}")
            melody_task = self.get_melody_task(melody_with_chord_clip.key, melody_all, prompt_measure, melody_all_ind, back_ind)
            melody_task_with_chord = self.get_melody_task_with_chord(melody_with_chord_clip.key, melody_with_chord_all, prompt_measure, melody_with_chord_all_ind, back_ind)
            chord_task = self.get_chord_task(melody_with_chord_clip.key, melody_all, chord_all, prompt_measure, melody_all_ind, chord_all_ind, back_ind)
            melody_task_add_chord = self.get_melody_task_add_chord(melody_with_chord_clip.key, melody_with_chord_all, chord_all, prompt_measure, melody_with_chord_all_ind, chord_all_ind, back_ind)

            self.marge(aya_node_inst, melody_task, melody_task_with_chord, chord_task, melody_task_add_chord)

            back_ind = prompt_measure + self.out_measure
            is_running = back_ind + self.prompt_max_measure + self.out_measure < len(melody_all_ind)
        return aya_node_inst

    def marge(self, aya_node_inst, melody_task, melody_task_with_chord, chord_task, melody_task_add_chord):
        if len(np.where(melody_task == self.tokenizer.get("<BLANK>"))[0]) < 4:
            aya_node_inst.append(melody_task)
            aya_node_inst.append(melody_task_with_chord)
        if len(np.where(chord_task == self.tokenizer.get("<BLANK>"))[0]) < 4:
            aya_node_inst.append(chord_task)
            aya_node_inst.append(melody_task_add_chord)

    def get_melody_task_add_chord(self, key, melody_all_with_chord, chord_all, prompt_measure, melody_with_chord_ind, chord_ind, back_ind) -> np.ndarray:
        """
        コード進行制約付き旋律生成タスクを取得する関数
        :param key:
        :param melody_all_with_chord:
        :param chord_all:
        :param prompt_measure:
        :param melody_with_chord_ind:
        :param back_ind:
        :return:
        """
        melody_prompt: np.ndarray = melody_all_with_chord[melody_with_chord_ind[back_ind]:melody_with_chord_ind[prompt_measure]]
        chord_prompt: np.ndarray  = chord_all[chord_ind[prompt_measure]:chord_ind[prompt_measure + self.out_measure]]
        melody_tgt: np.ndarray    = melody_all_with_chord[melody_with_chord_ind[prompt_measure]:melody_with_chord_ind[prompt_measure + self.out_measure]]
        melody_task = np.array([self.tokenizer.get(f"k_{key}")], dtype=int)
        melody_task = np.append(melody_task,self.tokenizer.get("<QUERY_M>"))

        melody_task = np.concatenate((melody_task, melody_prompt))
        melody_task = np.append(melody_task, self.tokenizer.get("</QUERY_M>"))
        melody_task = np.append(melody_task, self.tokenizer.get("<QUERY_C>"))
        melody_task = np.concatenate((melody_task, chord_prompt))
        melody_task = np.append(melody_task, self.tokenizer.get("</QUERY_C>"))
        melody_task = np.append(melody_task, self.tokenizer.get("<MGEN>"))
        melody_task = np.append(melody_task, self.tokenizer.get(f"k_{key}"))

        melody_task = np.concatenate((melody_task, melody_tgt))
        melody_task = np.append(melody_task, self.tokenizer.get("<ESEQ>"))
        return melody_task



    def get_chord_task(self, key, melody_all, chord_all, prompt_measure, melody_ind, chord_ind, back_ind) -> np.ndarray:
        """
        メロディからコード進行を予測するタスクを取得する関数
        :param melody_all: メロディの全体の配列
        :param chord_all: コード進行の全体の配列
        :param prompt_measure: プロンプトの小節数
        :return: メロディのタスクの配列
        """
        melody_prompt: np.ndarray = melody_all[melody_ind[back_ind]:melody_ind[prompt_measure]]
        chord_tgt: np.ndarray    = chord_all[chord_ind[back_ind]:chord_ind[prompt_measure]]

        melody_task = np.array([self.tokenizer.get(f"k_{key}")], dtype=int)
        melody_task = np.append(melody_task,self.tokenizer.get("<QUERY_M>"))

        melody_task = np.concatenate((melody_task, melody_prompt))
        melody_task = np.append(melody_task, self.tokenizer.get("</QUERY_M>"))
        melody_task = np.append(melody_task, self.tokenizer.get("<CGEN>"))
        melody_task = np.append(melody_task, self.tokenizer.get(f"k_{key}"))
        melody_task = np.concatenate((melody_task, chord_tgt))
        melody_task = np.append(melody_task, self.tokenizer.get("<ESEQ>"))

        return melody_task

    def get_melody_task(self, key,  melody_all, prompt_measure, ind, back_ind) -> np.ndarray:
        """
        単純メロディ生成タスクを取得する関数
        :param melody_all: メロディの全体の配列
        :param prompt_measure: プロンプトの小節数
        :return: メロディのタスクの配列
        """
        melody_prompt: np.ndarray = melody_all[ind[back_ind]:ind[prompt_measure]]
        melody_tgt: np.ndarray    = melody_all[ind[prompt_measure]:ind[prompt_measure + self.out_measure]]

        melody_task = np.array([self.tokenizer.get(f"k_{key}")], dtype=int)
        melody_task = np.append(melody_task,self.tokenizer.get("<QUERY_M>"))

        melody_task = np.concatenate((melody_task, melody_prompt))
        melody_task = np.append(melody_task, self.tokenizer.get("</QUERY_M>"))
        melody_task = np.append(melody_task, self.tokenizer.get("<MGEN>"))
        melody_task = np.append(melody_task, self.tokenizer.get(f"k_{key}"))
        melody_task = np.concatenate((melody_task, melody_tgt))
        melody_task = np.append(melody_task, self.tokenizer.get("<ESEQ>"))

        return melody_task

    def get_melody_task_with_chord(self, key, melody_all_with_chord, prompt_measure, ind, back_ind) -> np.ndarray:
        """
        コード進行付きメロディ生成タスクを取得する関数
        :param key:
        :param melody_all_with_chord:
        :param prompt_measure:
        :param ind:
        :param back_ind:
        :return:
        """
        melody_prompt: np.ndarray = melody_all_with_chord[ind[back_ind]:ind[prompt_measure]]
        melody_tgt: np.ndarray    = melody_all_with_chord[ind[prompt_measure]:ind[prompt_measure + self.out_measure]]

        melody_task = np.array([self.tokenizer.get(f"k_{key}")], dtype=int)
        melody_task = np.append(melody_task,self.tokenizer.get("<QUERY_M>"))

        melody_task = np.concatenate((melody_task, melody_prompt))
        melody_task = np.append(melody_task, self.tokenizer.get("</QUERY_M>"))
        melody_task = np.append(melody_task, self.tokenizer.get("<MGEN>"))
        melody_task = np.append(melody_task, self.tokenizer.get(f"k_{key}"))
        melody_task = np.concatenate((melody_task, melody_tgt))
        melody_task = np.append(melody_task, self.tokenizer.get("<ESEQ>"))
        return melody_task



class SeqWithChord2Chord(_AbstractConverter):
    def save(self, save_directory: str) -> [bool, str]:
        pass

    def convert(self, *args, **kwargs):
        shift_time = self.tokenizer.get_length_tuple("s")
        chord_root = self.tokenizer.get_length_tuple("CR")
        chord_quarter = self.tokenizer.get_length_tuple("CQ")
        chord_base = self.tokenizer.get_length_tuple("CB")
        blank = self.tokenizer.get("<BLANK>")
        sme = self.tokenizer.get("<SME>")
        for s in self.base:
            if not self.is_error:
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
                self.aya_node = self.aya_node + [self._ct_seq(tokens)]

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



class MidiExpantion(_AbstractMidiConverter):

    def save(self, save_directory: str) -> [bool, str]:
        if not self.is_error:
            self.midi_data.write(f"{save_directory}/{self.file_name}.mid")
            return True, "正常に完了しました"
        else:
            return False, "MIDIを読み込む事ができませんでした。"

    def convert(self, *args, **kwargs):
        pass

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None):
        super().__init__(MidiExpantion, tokenizer, directory, file_name, program_list, midi_data=midi_data)



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

