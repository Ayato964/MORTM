from pretty_midi.pretty_midi import Note

from abc import abstractmethod


def ct_time_to_beat(time: float, tempo: int) -> int:
    b4 = 60 / tempo
    b8 = b4 / 2
    b16 = b8 / 2
    b32 = b16 / 2
    b64 = b32 / 2

    beat, sub = calc_time_to_beat(time, b64)

    return beat


def ct_beat_to_time(beat: float, tempo: int) -> float:
    b4 = 60 / tempo
    b8 = b4 / 2
    b16 = b8 / 2
    b32 = b16 / 2
    b64 = b32 / 2
    return beat * b64


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
    def get_token(self, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        pass

    @abstractmethod
    def get_range(self) -> int:
        pass

    @abstractmethod
    def de_convert(self, number: int, tempo: int):
        pass

    @abstractmethod
    def set_tokens(self, tokens: dict):
        pass

    def __call__(self, back_notes: Note = None, note: Note = None, token: str = None, tempo: int = 120, *args,
                 **kwargs):
        if self.convert_type == 0:
            symbol: int | str | None = self.get_token(back_notes, note, tempo)
            if symbol == -999 or symbol is None:
                my_token = None
                pass
            else:
                my_token = f"{self.token_type}_{symbol}"
        else:
            if token is None:
                return None, None
            split = token.split("_")
            if split[0] == self.token_type:
                my_token = self.de_convert(int(float(split[-1])), tempo)
                return split[0], my_token
            else:
                return None, None

        return my_token


class MeasureToken(Token):


    def get_range(self) -> int:
        return 1

    def de_convert(self, number: int, tempo: int):
        return None

    def set_tokens(self, tokens: dict):
        tokens[f'm_start'] = len(tokens)

    def get_token(self, back_notes: Note, note: Note, tempo: int) -> int or None or str:
        measure1 = 60 / tempo * 4
        if back_notes is not None:
            note_measure = note.start // measure1
            back_note_measure = note.start // measure1
            if note_measure > back_note_measure:
                return "start"
            else:
                return None
        else:
            return "start"



class StartRE(Token):

    def set_tokens(self, tokens: dict):
        max_length = 192
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f's_{i}'] = tokens_length + i

    def de_convert(self, number: int, tempo):
        #print(ct_beat_to_time(number, tempo))
        return ct_beat_to_time(number, tempo)

    def get_range(self) -> int:
        return 193

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        now_start = ct_time_to_beat(note.start, tempo)
        if back_notes is not None:
            back_start = ct_time_to_beat(back_notes.start, tempo)
        else:
            back_start = 0
        shift = int(now_start - back_start)

        if shift < 0:
            print("WHATS!?!?!?!?!?")

        if shift > 192:
            shift = 128 + shift % 64
        if back_notes is None:
            shift = shift % 64
        return shift


class Pitch(Token):

    def set_tokens(self, tokens: dict):
        max_length = 127
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'p_{i}'] = tokens_length + i

    def de_convert(self, number: int, tempo):
        return number

    def get_range(self) -> int:
        return 129

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        p: int = note.pitch
        return p


class Velocity(Token):

    def set_tokens(self, tokens: dict):
        max_length = 127
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'v_{i}'] = tokens_length + i

    def de_convert(self, number: int, tempo):
        return number

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        v: int = note.velocity
        return v

    def get_range(self) -> int:
        return 128


class Duration(Token):

    def set_tokens(self, tokens: dict):
        max_length = 192
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'd_{i}'] = tokens_length + i

    def de_convert(self, number: int, tempo):
        return ct_beat_to_time(number, tempo)

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        start = ct_time_to_beat(note.start, tempo)
        end = ct_time_to_beat(note.end, tempo)
        d = int(max(abs(end - start), 1))

        if 192 < d:
            d = 191

        return d

    def get_range(self) -> int:
        return 192


class Start(Token):

    def de_convert(self, number: int, tempo):
        return ct_beat_to_time(number, tempo)

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        s = ct_time_to_beat(note.start, tempo)
        return s % 64

    def get_range(self) -> int:
        return 64


class Shift(Token):

    def de_convert(self, number: int, tempo):
        b4 = 60 / tempo
        b8 = b4 / 2
        b16 = b8 / 2
        b32 = b16 / 2
        b64 = b32 / 2
        return b64 * 64 * number

    def get_token(self, back_notes: Note, note: Note, tempo) -> int:
        if back_notes is None:
            return 0
        else:
            back_start = ct_time_to_beat(back_notes.start, tempo)
            note_start = ct_time_to_beat(note.start, tempo)

            shift = int(abs((back_start // 64) - (note_start // 64)))

            if shift > 3:
                shift = 3

        return shift

    def get_range(self) -> int:
        return 4
