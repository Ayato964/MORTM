from pretty_midi.pretty_midi import Note

from abc import abstractmethod


def ct_time_to_beat(time: float, tempo: int) -> int:
    b4 = 60 / tempo
    b8 = b4 / 2
    b16 = b8 / 2
    b32 = b16 / 2

    beat, sub = calc_time_to_beat(time, b32)

    return beat
def ct_beat_to_time(beat: float, tempo: int) -> float:
    b4 = 60 / tempo
    b8 = b4 / 2
    b16 = b8 / 2
    b32 = b16 / 2

    return beat * b32


def calc_time_to_beat(time, beat_time) -> (int, int):
    main_beat: int = time // beat_time
    sub_time: int = time % beat_time
    return main_beat, sub_time


class Token:
    def __init__(self, token_type: str, convert_type: int):
        self.token_type = token_type
        self.token_position = 0
        self.convert_type = convert_type

    @abstractmethod
    def get_token(self, back_notes: Note, note: Note, tempo:int) -> int:
        pass

    @abstractmethod
    def get_range(self) -> int:
        pass

    @abstractmethod
    def de_convert(self, number: int, tempo: int):
        pass

    def __call__(self, back_notes: Note = None, note: Note= None, token:str=None, tempo: int= 120, *args, **kwargs):
        if self.convert_type == 0:
            symbol: int = self.get_token(back_notes, note, tempo)
            if symbol == -999:
                my_token = None
                pass
            else:
                my_token = f"{self.token_type}_{symbol}"
        else:
            if token is None:
                return None, None
            split = token.split("_")
            if split[0] is self.token_type:
                my_token = self.de_convert(int(float(split[-1])), tempo)
                return split[0], my_token
            else:
                return None, None

        return my_token


class StartRE(Token):

    def de_convert(self, number: int, tempo):
        return ct_beat_to_time(number, tempo)

    def get_range(self) -> int:
        return 96

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        now_start = ct_time_to_beat(note.start, tempo)
        if back_notes is not None:
            back_start = ct_time_to_beat(back_notes.start, tempo)
        else:
            back_start = 0
        shift = int(now_start - back_start)

        if shift < 0:
            print("WHATS!?!?!?!?!?")

        if shift > 96 :
            shift = 64 + shift % 32
        if back_notes is None:
            shift = shift % 32
        return shift


class Pitch(Token):

    def de_convert(self, number: int, tempo):
        return number

    def get_range(self) -> int:
        return 128

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        p: int = note.pitch
        return p


class Velocity(Token):


    def de_convert(self, number: int, tempo):
        return number

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        v: int = note.velocity
        return v

    def get_range(self) -> int:
        return 128


class Duration(Token):

    def de_convert(self, number: int, tempo):
        return ct_beat_to_time(number, tempo)

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        start = ct_time_to_beat(note.start, tempo)
        end = ct_time_to_beat(note.end, tempo)
        d = int(max(abs(end - start), 1))

        if 100 < d:
            d = 100

        return d

    def get_range(self) -> int:
        return 100


class Start(Token):

    def de_convert(self, number: int, tempo):
        return ct_beat_to_time(number, tempo)

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        s = ct_time_to_beat(note.start, tempo)
        return s % 32

    def get_range(self) -> int:
        return 32


class Shift(Token):

    def de_convert(self, number: int, tempo):
        b4 = 60 / tempo
        b8 = b4 / 2
        b16 = b8 / 2
        b32 = b16 / 2

        return b32 * 32 * number

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        if back_notes is None:
            return 0
        else:
            back_start = ct_time_to_beat(back_notes.start, tempo)
            note_start = ct_time_to_beat(note.start, tempo)

            shift = int(abs((back_start // 32) - (note_start // 32)))

            if shift > 3:
                shift = 3

        return shift

    def get_range(self) -> int:
        return 4
