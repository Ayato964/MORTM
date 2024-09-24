from AGSM.convert import ConvTempo
from pretty_midi.pretty_midi import PrettyMIDI, Instrument, Note
import mido
import numpy as np
from numpy import ndarray
from typing import List, Callable

from . import constants
from .aya_node import Token, Pitch, Velocity, Duration, Shift, Start
from . import aya_node
from .tokenizer import Tokenizer

class MidiToAyaNode:

    def __init__(self, tokenizer: Tokenizer, directory: str, file_name: str, program_list, midi_data=None,):
        self.program_list = program_list
        self.directory = directory
        self.file_name = file_name
        self.tempo = 120
        self.is_error = False
        self.aya_node = [0]
        self.token_converter: List[Token] = tokenizer.token_list

        self.tokenizer = tokenizer

        if midi_data is not None:
            self.midi_data: PrettyMIDI = midi_data
        else:
            try:
                self.midi_data: PrettyMIDI = PrettyMIDI(f"{directory}/{file_name}")
            except Exception:
                self.is_error = True
                print("ロードできませんでした。")
        pass

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
            self.ct_tempo()
            program_count = 0
            for inst in self.midi_data.instruments:
                inst: Instrument = inst
                if not inst.is_drum and inst.program in self.program_list:
                    print(f"Instrument Number:{inst.program}")
                    aya_node_inst = self.ct_aya_node(inst)

                    self.aya_node = self.aya_node + aya_node_inst
                    #print(self.aya_node)
                    program_count += 1

            if program_count == 0:
                self.is_error = True
                print(f"{self.directory}/{self.file_name}に、欲しい楽器がありませんでした。")
        else:
            print(f"{self.directory}/{self.file_name}を変換できません。")


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

        for note in inst.notes:
            note: Note = note

            if back_note is None:
                clip = np.append(clip, self.tokenizer.get(constants.START_SEQ_TOKEN))

            for conv in self.token_converter:
                conv: Token = conv

                token = conv(back_note, note)
                if token is not None:
                    token_id = self.tokenizer.get(token)
                    clip = np.append(clip, token_id)

            back_note = note

            clip_time = note.end

            if clip_time >= 60 * split_count:
                if len(clip) >= 10:
                    clip = np.append(clip, self.tokenizer.get(constants.END_SEQ_TOKEN))

                    aya_node_inst.append(clip)

                clip = np.array([], dtype=int)
                back_note = None

                clip_time = 0.0
                split_count += 1

        if len(clip) >= 10:
            clip = np.append(clip, self.tokenizer.get(constants.END_SEQ_TOKEN))
            aya_node_inst.append(clip)

        return aya_node_inst


    def save(self, save_directory: str) -> bool:
        if not self.is_error:
            #print(f"Result shape is:{self.aya_node.shape}")

            array_dict = {f'array{i}': arr for i, arr in enumerate(self.aya_node)}
            np.savez(save_directory + "/" + self.file_name, **array_dict)
            print("処理が正常に終了しました。")
            return True
        else:
            print(f"{constants.MODEL_NAME}が望むデータ形式ではないため、保存ができませんでした。")
            return False

    def ct_tempo(self):
        try:
            ct = ConvTempo(directory=self.directory + "/" + self.file_name, midi_data=self.midi_data, change_tempo=120)
            ct.convert()
            self.midi_data = ct.midi_data
        except OSError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True
        except IndexError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True
        except ValueError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True

        except mido.midifiles.meta.KeySignatureError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True

        except EOFError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True

        except KeyError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True
        except ZeroDivisionError:
            print(f"{self.directory}/{ self.file_name}でエラーが発生。処理を中断します。")
            self.is_error = True


class AyaNodeToMidi:
    def __init__(self, directory:str):
        self.directory = directory
        npz = np.load(directory)
        self.npz_dict = npz
        self.midi_data = None
        print(self.npz_dict)

    def convert(self):
        pass