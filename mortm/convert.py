import os
from typing import List, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio
from pretty_midi.pretty_midi import PrettyMIDI, Instrument, Note, TimeSignature
from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from midi2audio import FluidSynth
import soundfile as sf

from .custom_token import Token, ShiftTimeContainer
from mortm.train.tokenizer import Tokenizer

T = TypeVar("T")


def conv_spectro(waveform, sample_rate, n_fft, hop_length, n_mels):
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
        pass

    @abstractmethod
    def convert(self, *args, **kwargs):
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

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None, split_measure=12):
        super().__init__(MIDI2Seq, tokenizer, directory, file_name, program_list, midi_data)
        self.aya_node = [0]
        self.split_measure = split_measure

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

                token = conv(inst=inst, back_notes=back_note, note=note, tempo=tempo, container=shift_time_container)

                if token is not None:
                    if conv.token_type == "<SME>":
                        clip_count += 1
                    if clip_count >= self.split_measure:
                        clip = np.append(clip, self.tokenizer.get("<ESEQ>"))
                        aya_node_inst = self.marge_clip(clip, aya_node_inst)
                        clip = np.array([], dtype=int)
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