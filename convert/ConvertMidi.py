import string
from abc import abstractmethod
import os
from typing import Any

import mido.midifiles.meta
import music21.midi
import numpy
import pretty_midi as midi
from numpy import ndarray, dtype
from pretty_midi import PrettyMIDI, Instrument, Note
import music21 as m21
import numpy as np

import constants
from util.ArrayList import ArrayList
from AGSM.convert import ConvTempo, ConvKey
from transformer.tokenizer import Tokenizer
import transformer.tokenizer as tr

PITCH = 0
VELOCITY = 1
DURATION_INT = 2
DURATION_FEW = 3
BEGIN_TIME_INT = 4
BEGIN_TIME_FEW = 5
START_SEQ = np.array([[0, 0, 0, 0]], dtype=int)
END_SEQ = np.array([[5, 5, 5, 5]], dtype=int)
''''
規則：
    AyaNode:
        Aya_node = [音高, 強さ, 始まりの拍(16), 終わりの拍までの長さ(16)]
    TOKEN:
        <S_SEQ>: [0, 0, 0, 0]
        <E_SEQ>: [5, 5, 5, 5]
'''

def convert_tolist(target: list):
    new_target = None
    for t in target:
        t: ndarray
        cul = []
        for c in t:
            cul.append(c)
        if new_target is not None:
            new_target = new_target + [cul]
        else:
            new_target = [cul]

    return new_target


class ConvertMidi:

    def __init__(self, tokenizer: Tokenizer, directory: str, program_list: list, tempo: int, midi_data: PrettyMIDI = None):
        self.directory = directory
        self.midi_data = midi_data
        self.program_list = program_list
        self.tempo = tempo
        self.aya_node = None
        self.is_error = False
        self.tokenizer = tokenizer

    def convert(self):
        self._ct_tempo()
        #self._ct_key()
        if not self.is_error:
            convert_notes: np = None
            program_count = 0
            for inst in self.midi_data.instruments:
                inst: Instrument = inst
                if not inst.is_drum and inst.program in self.program_list:
                    cn = self.ct_aya_node(inst.notes)

                    if len(cn) >= 1000:
                        new_array = np.array_split(cn, np.arange(1000, len(cn), 1000))
                        new_array = convert_tolist(new_array)
                        cn: ndarray = np.array(self._padding(new_array))
                        print(cn.shape)
                    else:
                        cn: ndarray = np.array(self._padding([cn.tolist()], 1000))

                    if convert_notes is None:
                        convert_notes = cn
                    else:
                        #pad_convert_notes, pad_cn = self.padding(convert_notes, cn)
                        convert_notes = np.vstack((convert_notes, cn), dtype=int)
                    program_count += 1

            if program_count == 0:
                print(f"{self.directory}には欲しい楽曲がありませんでした")
                self.is_error = True
            elif np.min(convert_notes) < 0:
                print(f"{self.directory}の値の中に0以下の値が格納されていたため、中断します。")
                self.is_error = True

            self.aya_node = convert_notes
            return convert_notes


    def split_seq_data(self, target: ndarray):

        if len(target.shape) == 1:
            musics_seq = [target.tolist()]
        else:
            musics_seq = target.tolist()

        new_music_seq = [[]]
        for i in range(len(musics_seq)):
            result = []
            current_sublist = []
            for value in musics_seq[i]:
                current_sublist.append(value)
                if value == 2:
                    result.append(current_sublist)
                    current_sublist = []
            new_music_seq = new_music_seq + result

        return np.array(self._padding(new_music_seq))

        pass

    def _padding(self, target: list, max_length: int = None):
        if max_length is None:
            max_lengths = []
            for t in target:
                max_lengths.append(len(t))
            max_length = max(max_lengths)

        print(f"Max length is {max_length}")
        for t in target:
            if len(t) < max_length:
                for _ in range(max_length - len(t)):
                    t.append(0)
        return target
        pass

    def save(self) -> bool:
        if not self.is_error:
            print(f"Result shape is:{self.aya_node.shape}")

            current_dir = os.path.dirname(os.path.abspath(__file__))

            # ルートディレクトリまでの相対パスを計算
            project_root = os.path.abspath(os.path.join(current_dir, '..'))

            # ルートディレクトリからoutディレクトリへのパスを生成
            out_directory = os.path.join(project_root, 'out')

            split_direc = self.directory.split("\\")

            filename = split_direc[-1].split(".")[0]

            np.savez(out_directory + "/np/datasets/" + filename, self.aya_node)
            print("処理が正常に終了しました。")
            return True
        else:
            print("Transformerが望むデータ形式ではないため、保存ができませんでした。")
            return False

    def ct_aya_node(self, notes: list) -> ndarray[Any, dtype[Any]]:
        node = np.array([self.tokenizer.get(-1, constants.START_SEQ_TOKEN)])

        back_start = None
        for note in notes:

            note: Note = note
            start = self.ct_time_to_beat(note.start)
            end = self.ct_time_to_beat(note.end)

            if back_start is not None:
                shift = abs((back_start // 32) - (start // 32))
                if 4 > shift:
                    if shift > 0:
                        node = np.append(node, self.tokenizer.get(shift, tr.SHIFT_TYPE)) #何小節のブランクができたかを計算
                else:
                    node = np.append(node, self.tokenizer.get(-1, constants.END_SEQ_TOKEN))
                    node = np.append(node, self.tokenizer.get(-1, constants.START_SEQ_TOKEN))

            pitch_token = self.tokenizer.get(note.pitch, tr.PITCH_TYPE)
            velocity_token = self.tokenizer.get(note.velocity, tr.VELOCITY_TYPE)
            duration_token = self.tokenizer.get(min(abs(end - start), 99), tr.DURATION_TYPE)
            start_token = self.tokenizer.get(start % 32, tr.START_TYPE)
            note_token = np.array([start_token, pitch_token, velocity_token, duration_token])
            node = np.concatenate((node, note_token))
            back_start = start

        node = np.append(node, self.tokenizer.get(-1, constants.END_SEQ_TOKEN))
        return node

    def _ct_key(self):
        try:
            if self.midi_data is None:
                ct = ConvKey(self.directory, "C")
                ct.convert()
                self.midi_data = ct.midi_data
                if ct.is_Error:
                    self.is_error = ct.is_Error
            else:
                ct = ConvKey(directory=self.directory, midi_data=self.midi_data, key="C")
                ct.convert()
                self.midi_data = ct.midi_data
        except OSError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True
        except IndexError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True
        except mido.midifiles.meta.KeySignatureError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True
        except ValueError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True



    def _ct_tempo(self):
        try:
            if self.midi_data is None:
                ct = ConvTempo(self.directory, self.tempo)
                ct.convert()
                self.midi_data = ct.midi_data
                if ct.is_Error:
                    self.is_error = ct.is_Error
            else:
                ct = ConvTempo(directory=self.directory, midi_data=self.midi_data, change_tempo=120)
                ct.convert()
                self.midi_data = ct.midi_data
        except OSError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True
        except IndexError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True
        except ValueError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True

        except mido.midifiles.meta.KeySignatureError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True

        except EOFError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True

        except KeyError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True
        except ZeroDivisionError:
            print(f"{self.directory}でエラーが発生。処理を中断します。")
            self.is_error = True



    def ct_time_to_beat(self, time: float) -> int:
        b4 = 60 / self.tempo
        b8 = b4 / 2
        b16 = b8 / 2
        b32 = b16 / 2

        beat, sub = self.calc_time_to_beat(time, b32)

        return beat

    def calc_time_to_beat(self, time, beat_time) -> (int, int):
        main_beat: int = time // beat_time
        sub_time: int = time % beat_time
        return main_beat, sub_time

    def padding(self, array1: ndarray, array2: ndarray):
        print(f"a:{array1.shape[-1]}  b:{array2.shape[-1]}")

        max_len = max(array1.shape[-1], array2.shape[-1])

        # 0でパディングして同じ長さに揃える
        if len(array1.shape) > 1:
            padded_array1 = np.array([np.pad(row, (0, max_len - len(row)), mode='constant') for row in array1])
        else:
            padded_array1 = np.pad(array1, (0, max_len - len(array1)), mode='constant')

        if len(array2.shape) > 1:
            padded_array2 = np.array([np.pad(row, (0, max_len - len(row)), mode='constant') for row in array2])
        else:
            padded_array2 = np.pad(array2, (0, max_len - len(array2)), mode='constant')

        return padded_array1, padded_array2

