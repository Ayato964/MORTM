import numpy as np
from numpy import ndarray
from pretty_midi.pretty_midi import Note
from typing import Callable

from .tokenizer import Tokenizer
from .tokenizer import PITCH_TYPE,SHIFT_TYPE,START_TYPE,VELOCITY_TYPE,DURATION_TYPE


def ct_time_to_beat(time: float, tempo: int) -> int:
    b4 = 60 / tempo
    b8 = b4 / 2
    b16 = b8 / 2
    b32 = b16 / 2

    beat, sub = calc_time_to_beat(time, b32)

    return beat


def calc_time_to_beat(time, beat_time) -> (int, int):
    main_beat: int = time // beat_time
    sub_time: int = time % beat_time
    return main_beat, sub_time


def get_token(tempo: int, tokenizer: Tokenizer, token_type: str) -> Callable[[Note, Note], int]:
    def get_pitch(back_notes: Note, note: Note) -> int:
        p: int = note.pitch
        return tokenizer.get(p, PITCH_TYPE)



    def get_velocity(back_notes: Note, note: Note) -> int:
        v: int = note.velocity
        return tokenizer.get(v, VELOCITY_TYPE)


    def get_duration(back_note: Note, note: Note) -> int:
        start = ct_time_to_beat(note.start, tempo)
        end = ct_time_to_beat(note.end, tempo)
        d = int(max(abs(end - start), 1))

        if 100 < d:
            d = 100

        return tokenizer.get(d, DURATION_TYPE)


    def get_start(back:Note, note: Note) -> int:
        s = ct_time_to_beat(note.start, tempo)

        return tokenizer.get(int(s % 32), START_TYPE)


    def get_shift(back: Note, note: Note) -> int:
        if back is None:
            return -1
        else:
            back_start = ct_time_to_beat(back.start, tempo)
            note_start = ct_time_to_beat(note.start, tempo)

            shift = int(abs((back_start // 32) - (note_start // 32)))

            if shift > 3:
                shift = 3

            return tokenizer.get(shift, SHIFT_TYPE)


    if token_type is PITCH_TYPE:
        return get_pitch
    elif token_type is VELOCITY_TYPE:
        return get_velocity
    elif token_type is DURATION_TYPE:
        return get_duration
    elif token_type is START_TYPE:
        return get_start
    elif token_type is SHIFT_TYPE:
        return get_shift
