from pretty_midi import Note, Instrument
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


def _get_symbol(token: str):
    split = token.split("_")
    return int(float(split[-1]))



class Token:
    def __init__(self, token_type: str, convert_type: int):
        self.token_type = token_type
        self.token_position = 0
        self.convert_type = convert_type

        self.start = 0
        self.end = 0

    @abstractmethod
    def get_token(self, inst:Instrument, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        pass

    @abstractmethod
    def de_convert(self, number: int | str, back_note:Note, note: Note, tempo: int):
        pass

    @abstractmethod
    def _set_tokens(self, tokens: dict):
        pass

    def set_tokens(self, tokens:dict):
        self.start = len(tokens)
        self._set_tokens(tokens)
        self.end = len(tokens) - 1

    @abstractmethod
    def is_my_token(self, seq):
        pass

    @abstractmethod
    def __call__(self, inst:Instrument=None,  back_notes: Note = None, note: Note = None, token:str =None, tempo=120, *args, **kwargs):
        pass

class SpecialToken(Token):

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int):
        pass

    def _set_tokens(self, tokens: dict):
        tokens[self.token_type] = len(tokens)

    def is_my_token(self, seq):
        pass

    @abstractmethod
    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        pass

    def __call__(self, inst:Instrument=None, back_notes: Note = None, note: Note = None, token:str =None, tempo=120, *args, **kwargs):
        if self.convert_type == 0:
            return self.get_token(inst=inst, back_notes=back_notes, note=note, tempo=tempo)
        else:
            return token == self.token_type


class MusicToken(Token):

    def __call__(self,inst:Instrument=None, back_notes: Note = None, note: Note = None, token:str =None, tempo=120, *args, **kwargs,):
        if self.convert_type == 0:
            return f"{self.token_type}_{self.get_token(inst=inst, back_notes=back_notes, note=note, tempo=tempo)}"
        else:
            if token is None:
                return None
            split = token.split("_")
            if split[0] == self.token_type:
                self.de_convert(split[1], back_notes, note, tempo)
                return split[0]
            else:
                return None

    @abstractmethod
    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        pass

    @abstractmethod
    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int):
        pass

    @abstractmethod
    def _set_tokens(self, tokens: dict):
        pass

    @abstractmethod
    def is_my_token(self, seq):
        pass


class MeasureToken(SpecialToken):
    def __init__(self, convert_type: int):

        super().__init__("<SME>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int) -> int or None or str:
        measure1 = 60 / tempo * 4
        if back_notes is not None:
            note_measure = note.start // measure1
            back_note_measure = back_notes.start // measure1
            if note_measure > back_note_measure:
                return self.token_type
            else:
                return None
        else:
            return self.token_type


class TrackStart(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<TS>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        if inst.notes[0] == note:
            return self.token_type
        else:
            return None


class TrackEnd(SpecialToken):

    def __init__(self, convert_type: int):
        super().__init__("<TE>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        if inst.notes[-1] == note:
            return self.token_type
        else:
            return None
class Blank(SpecialToken):

    def __init__(self, convert_type: int):
        super().__init__("<BLANK>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int) -> int | str | None:
        measure1 = 60 / tempo * 4
        if back_notes is not None:
            note_measure = note.start // measure1
            back_note_measure = back_notes.start // measure1
            if note_measure > back_note_measure + 1:
                return self.token_type
            else:
                return None
        else:
            return None


class StartRE(MusicToken):

    def _set_tokens(self, tokens: dict):
        max_length = 64
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f's_{i}'] = tokens_length + i

    def de_convert(self, number: int, back_note, note: Note, tempo):
        shift = ct_beat_to_time(number, tempo)
        note.start = shift if back_note is None else shift + back_note.start

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo) -> int:
        measure1 = 60 / tempo * 4
        now_start = ct_time_to_beat(note.start, tempo)
        if back_notes is not None:
            note_measure = note.start // measure1
            back_note_measure = back_notes.start // measure1
            if note_measure > back_note_measure:
                shift = int(now_start)
                if shift < 0:
                    print("WHATS!?!?!?!?!?")
                print(shift % 64)
                return shift % 64
            else:
                back_start = ct_time_to_beat(back_notes.start, tempo)
                shift = int(now_start - back_start)
                if shift < 0:
                    print("WHATS!?!?!?!?!?")
                return shift
        else:
            shift = int(now_start)
            if shift < 0:
                print("WHATS!?!?!?!?!?")
            return shift % 64



class Pitch(MusicToken):

    def is_my_token(self, seq):
        pass

    def _set_tokens(self, tokens: dict):
        max_length = 127
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'p_{i}'] = tokens_length + i

    def de_convert(self, number: int, back_note, note: Note, tempo):
        note.pitch = number

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo) -> int:
        p: int = note.pitch
        return p


class Duration(MusicToken):

    def _set_tokens(self, tokens: dict):
        max_length = 192
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'd_{i}'] = tokens_length + i

    def de_convert(self, number: int, back_note: Note, note: Note, tempo):
        duration = ct_beat_to_time(number, tempo)
        note.end = note.start + duration

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo) -> int:
        start = ct_time_to_beat(note.start, tempo)
        end = ct_time_to_beat(note.end, tempo)
        d = int(max(abs(end - start), 1))

        if 192 < d:
            d = 191

        return d
