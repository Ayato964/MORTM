
from pretty_midi.pretty_midi import Note

from .tokenizer import Tokenizer
from .tokenizer import PITCH_TYPE, SHIFT_TYPE, START_TYPE, VELOCITY_TYPE, DURATION_TYPE
from abc import abstractmethod


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


class Token:
    def __init__(self, tempo: int, tokenizer: Tokenizer, token_type: str):
        self.tempo = tempo
        self.tokenizer = tokenizer
        self.token_type = token_type
        self.token_position = 0

    @abstractmethod
    def get_token(self, back_notes: Note, note: Note) -> int:
        pass

    @abstractmethod
    def get_range(self) -> int:
        pass



class Pitch(Token):

    def __init__(self, tempo: int, tokenizer: Tokenizer, token_type: str):
        super().__init__(tempo, tokenizer, token_type)

    def get_range(self) -> int:
        return 128

    def get_token(self, back_notes: Note, note: Note) -> int:
        p: int = note.pitch
        return self.tokenizer.get(p, PITCH_TYPE)


class Velocity(Token):

    def __init__(self, tempo: int, tokenizer: Tokenizer, token_type: str):
        super().__init__(tempo, tokenizer, token_type)

    def get_token(self, back_notes: Note, note: Note) -> int:
        v: int = note.velocity
        return self.tokenizer.get(v, VELOCITY_TYPE)

    def get_range(self) -> int:
        return 128


class Duration(Token):

    def __init__(self, tempo: int, tokenizer: Tokenizer, token_type: str):
        super().__init__(tempo, tokenizer, token_type)

    def get_token(self, back_notes: Note, note: Note) -> int:
        start = ct_time_to_beat(note.start, self.tempo)
        end = ct_time_to_beat(note.end, self.tempo)
        d = int(max(abs(end - start), 1))

        if 100 < d:
            d = 100

        return self.tokenizer.get(d, DURATION_TYPE)

    def get_range(self) -> int:
        return 100


class Start(Token):
    def get_token(self, back_notes: Note, note: Note) -> int:
        s = ct_time_to_beat(note.start, self.tempo)
        return self.tokenizer.get(int(s % 32), START_TYPE)

    def get_range(self) -> int:
        return 32


class Shift(Token):
    def __init__(self, tempo: int, tokenizer: Tokenizer, token_type: str):
        super().__init__(tempo, tokenizer, token_type)

    def get_token(self, back_notes: Note, note: Note) -> int:
        if back_notes is None:
            return -1
        else:
            back_start = ct_time_to_beat(back_notes.start, self.tempo)
            note_start = ct_time_to_beat(note.start, self.tempo)

            shift = int(abs((back_start // 32) - (note_start // 32)))

            if shift > 3:
                shift = 3

        return self.tokenizer.get(shift, SHIFT_TYPE)

    def get_range(self) -> int:
        return 4