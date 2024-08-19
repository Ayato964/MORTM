import numpy as np
import pretty_midi as pm
from pretty_midi.pretty_midi import Instrument, Note

import constants
from transformer.tokenizer import Tokenizer

def sum_begin_time(a_int: int, a_few: int):
    sum_str = f"{a_int}.{a_few}"
    sum_float = float(sum_str)
    return sum_float


def get_int_note(note_str: str):
    if note_str == constants.START_SEQ_TOKEN or note_str == constants.END_SEQ_TOKEN:
        return None
    else:
        cleaned_string = note_str.strip("[]")

        # スペースで文字列を分割
        split_list = cleaned_string.split()

        # 各要素を整数に変換
        result_list = [int(i) for i in split_list]
        return result_list


def convert(seq: list, tokenizer: Tokenizer) -> pm:
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=3, is_drum=False)
    midi.instruments.append(inst)
    back_beat = 0
    time = 0

    for note in seq:
        note_str :str = tokenizer.rev_get(note)
        split_note = get_int_note(note_str)
        if split_note is not None:
            if back_beat > split_note[2]:
                new_beat = back_beat + back_beat % 32
                start = 0.125 * (split_note[2] + new_beat)
            else:
                start = time + 0.125 * (split_note[2])
            end = start + (0.125 * split_note[3])
            inst.notes.append(Note(pitch=split_note[0], velocity=100, start=start, end=end))
            back_beat += split_note[2] + split_note[3]
            time = end
        pass

    return midi

