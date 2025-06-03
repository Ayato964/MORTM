from pretty_midi import Note, Instrument
from abc import abstractmethod
from .train.utils.chord_midi import ChordMidi

roots = ['C', 'C#', 'D', 'D#', 'E', 'E#', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'Cb','Db','Eb','Fb','Gb','Ab','Bb']
bases = ['/C', '/C#', '/D', '/D#', '/E', '/E#', '/F', '/F#', '/G', '/G#', '/A', '/A#', '/B', "/B#", "/Db","/Eb", "/Fb", "/Gb","/Ab","/Bb","/Cb", "None"]
qualities = [
    'None', 'm', '7', 'maj7', 'm7', 'dim', 'aug', "m7b5", "b5",
    'sus2', 'sus4', '6', 'm6', '9', 'maj9', 'm9', '11', '13', 'add9'
]

key = ['CM','DM','EM','FM','GM','AM','BM','C#M','D#M','F#M','G#M','AM', 'DbM','EbM','GbM','AbM','BbM',
       'Cm','Dm','Em','Fm','Gm','Am','Bm','C#m','D#m','F#m','G#m','Am', 'Dbm','Ebm','Gbm','Abm','Bbm']

def parse_chord(chord: str):
    """
    コード文字列を root, quality, base に分解する。
    例: "Em/G" -> ("E", "m", "/G")
    """
    # 1) ベース音の抽出（"/" 以下を base とする）
    if '/' in chord:
        core, bass_note = chord.split('/', 1)
        base = '/' + bass_note
    else:
        core = chord
        base = 'None'  # ベースなしの場合は空文字

    # 2) ルートの抽出（シャープ付きを優先）
    #    roots を長さ順にソートして先にマッチさせる
    sorted_roots = sorted(roots, key=lambda x: -len(x))
    for r in sorted_roots:
        if core.startswith(r):
            root = r
            quality = core[len(r):] or 'None'
            break
    else:
        raise ValueError(f"Unknown root in chord '{chord}'")

    # 3) クオリティがリスト外なら 'None' にフォールバック
    if quality not in qualities:
        quality = 'None'

    return root, quality, base


def ct_time_to_beat(time: float, tempo: int) -> int:
    '''
    Convert time to beat.
    :param time:
    :param tempo:
    :return:
    '''
    b4 = 60 / tempo
    measure = b4 * 4
    # 96 subdivisions per measure
    b96 = measure / 96

    beat, sub = calc_time_to_beat(time, b96)

    return beat


def ct_beat_to_time(beat: float, tempo: int) -> float:
    '''
    Convert beat to time.
    :param beat:
    :param tempo:
    :return:
    '''
    b4 = 60 / tempo
    measure = b4 * 4
    # 96 subdivisions per measure
    b96 = measure / 96

    return float(beat) * b96


def calc_time_to_beat(time, beat_time) -> (int, int):
    '''
    Convert time to beat.
    :param time:
    :param beat_time:
    :return:
    '''
    main_beat: int = time // beat_time
    sub_time: int = time % beat_time
    return main_beat, sub_time


def _get_symbol(token: str):
    split = token.split("_")
    return int(float(split[-1]))


class ShiftTimeContainer:
    def __init__(self, time, tempo):
        self.measure_start_time = time
        self.shift_measure = False
        self.is_error = False
        self.tempo = tempo
        self.is_code_mode = False

    def shift(self):
        self.measure_start_time += (60 / self.tempo) * 4
        self.shift_measure = True


class Token:
    def __init__(self, token_type: str, convert_type: int):
        self.token_type = token_type
        self.token_position = 0
        self.convert_type = convert_type

        self.start = 0
        self.end = 0

    @abstractmethod
    def get_token(self, *args, **kwargs) -> int | str | None:
        pass

    @abstractmethod
    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    @abstractmethod
    def _set_tokens(self, tokens: dict):
        pass

    def set_tokens(self, tokens: dict):
        self.start = len(tokens)
        self._set_tokens(tokens)
        self.end = len(tokens) - 1

    @abstractmethod
    def is_my_token(self, seq):
        pass

    @abstractmethod
    def __call__(self,  *args, **kwargs):
        pass


class SpecialToken(Token):

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def _set_tokens(self, tokens: dict):
        tokens[self.token_type] = len(tokens)

    def is_my_token(self, seq):
        pass

    @abstractmethod
    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    def __call__(self, inst: Instrument = None, back_notes: Note = None, note: Note = None, token: str = None,
                 tempo=120, container: ShiftTimeContainer = None, *args, **kwargs):
        if self.convert_type == 0:
            return self.get_token(inst=inst, back_notes=back_notes, note=note, tempo=tempo, container=container)
        else:
            return token == self.token_type


class MusicToken(Token):

    def __call__(self, inst: Instrument = None, back_notes: Note = None, note: Note = None, token: str = None,
                 tempo=120, container: ShiftTimeContainer = None,*args, **kwargs, ):
        if self.convert_type == 0:
            k = self.get_token(inst=inst, back_notes=back_notes, note=note, tempo=tempo, container=container)
            return f"{self.token_type}_{k}" if k else None
        else:
            if token is None:
                return None
            split = token.split("_")
            if split[0] == self.token_type:
                self.de_convert(split[1], back_notes, note, tempo, container)
                return split[0]
            else:
                return None

    @abstractmethod
    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

    @abstractmethod
    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    @abstractmethod
    def _set_tokens(self, tokens: dict):
        pass

    @abstractmethod
    def is_my_token(self, seq):
        pass


class ChordToken(Token):

    def __call__(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer, token=None, *args, **kwargs):
        if self.convert_type == 0:
            token_c = self.get_token(note=note, chords=chords, container=container)
            if token_c is not None:
                return f"{self.token_type}_{token_c}"
            else:
                return None
        else:
            t_s = token.split("_")
            return t_s[1]

    def __init__(self, token_type: str, convert_type: int):
        super().__init__(token_type, convert_type)

    @abstractmethod
    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> int | str | None:
        pass

class MeasureToken(SpecialToken):
    def __init__(self, convert_type: int):

        super().__init__("<SME>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int or None or str:

        measure1 = 60 / tempo * 4
        if back_notes is not None and not container.shift_measure:
            note_measure = note.start // measure1
            back_note_measure = back_notes.start // measure1
            if note_measure > back_note_measure:
                container.shift()
                return self.token_type
            else:

                return None
        else:
            return self.token_type


class TrackStart(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<TS>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        if inst.notes[0] == note:
            return self.token_type
        else:
            return None


class TrackEnd(SpecialToken):

    def __init__(self, convert_type: int):
        super().__init__("<TE>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        if inst.notes[-1] == note:
            return self.token_type
        else:
            return None


class Blank(SpecialToken):

    def __init__(self, convert_type: int):
        super().__init__("<BLANK>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        measure1 = 60 / tempo * 4
        if back_notes is not None:
            note_measure = note.start // measure1
            back_note_measure = back_notes.start // measure1 if not container.shift_measure else container.measure_start_time // measure1
            if note_measure > back_note_measure + 1:
                container.shift()
                return self.token_type
            else:
                return None
        else:
            return None


class SequenceEnd(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<ESEQ>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        return None


class MGen(SpecialToken):

    def __init__(self, convert_type: int):
        super().__init__("<MGEN>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        return None


class CGen(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<CGEN>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class CLS(SpecialToken):
    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int,
                  container: ShiftTimeContainer) -> int | str | None:
        pass
    def __init__(self, convert_type: int):
        super().__init__("<CLS>", convert_type)


class QueryMelody(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<QUERY_M>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

class QueryMelodyEnd(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("</QUERY_M>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class QueryChord(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("<QUERY_C>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass

class QueryChordEnd(SpecialToken):
    def __init__(self, convert_type: int):
        super().__init__("</QUERY_C>", convert_type)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int, container: ShiftTimeContainer) -> int | str | None:
        pass


class Key(MusicToken):
    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo: int,
                  container: ShiftTimeContainer) -> int | str | None:
        pass

    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, k in enumerate(key):
            tokens[f'k_{k}'] = base + i

    def __init__(self, convert_type: int):
        super().__init__("k", convert_type)



class StartRE(MusicToken):

    def _set_tokens(self, tokens: dict):
        max_length = 96
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f's_{i}'] = tokens_length + i

    def de_convert(self, number: int, back_note, note: Note, tempo, container: ShiftTimeContainer):
        shift = ct_beat_to_time(number, tempo)
        if not container.shift_measure:
            note.start = shift if back_note is None else shift + back_note.start
        else:
            note.start = container.measure_start_time + shift
            container.shift_measure = False

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        measure1 = 60 / tempo * 4
        now_start = ct_time_to_beat(note.start, tempo)
        if back_notes is not None and not container.shift_measure:
            note_measure = note.start // measure1
            back_note_measure = back_notes.start // measure1
            if note_measure > back_note_measure:
                shift = int(now_start)
                if shift < 0:
                    container.is_error = True
                return shift % 96
            else:
                back_start = ct_time_to_beat(back_notes.start, tempo)
                shift = int(now_start - back_start)
                if shift < 0:
                    container.is_error = True
                return shift
        else:
            shift = int(now_start - ct_time_to_beat(container.measure_start_time, tempo))
            container.shift_measure = False
            if shift < 0:
                container.is_error = True
            return shift % 96


class Duration(MusicToken):

    def _set_tokens(self, tokens: dict):
        # allow up to 3 measures (96 * 3)
        max_length = 96 * 3
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'd_{i}'] = tokens_length + i

    def de_convert(self, number: int, back_note: Note, note: Note, tempo, container: ShiftTimeContainer):
        duration = ct_beat_to_time(number, tempo)
        note.end = note.start + duration

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        start = ct_time_to_beat(note.start, tempo)
        end = ct_time_to_beat(note.end, tempo)
        d = int(max(abs(end - start), 1))

        if 96 * 3 < d:
            d = 96 * 3 - 1

        return d



class Pitch(MusicToken):

    def is_my_token(self, seq):
        pass

    def _set_tokens(self, tokens: dict):
        max_length = 127
        tokens_length = len(tokens)
        for i in range(max_length + 1):
            tokens[f'p_{i}'] = tokens_length + i

    def de_convert(self, number: int, back_note, note: Note, tempo, container: ShiftTimeContainer):
        note.pitch = int(number)

    def get_token(self, inst: Instrument, back_notes: Note, note: Note, tempo, container: ShiftTimeContainer) -> int:
        p: int = note.pitch
        return p


#--- ChordRoot トークン ------------------------------------------------
class ChordRoot(ChordToken):
    def __init__(self, convert_type: int):
        super().__init__('CR', convert_type)

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, r in enumerate(roots):
            tokens[f'CR_{r}'] = base + i

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> str:
        # NOTE: note.chord_root 属性を事前にセットしておく前提
        c, _ = chords.get_chord(note.start)
        if not c.is_called:
            root, _, _ = parse_chord(c.chord)
            return root
        else:
            return None


#--- ChordQuality トークン ----------------------------------------------
class ChordQuality(ChordToken):
    def __init__(self, convert_type: int):
        super().__init__('CQ', convert_type)

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, q in enumerate(qualities):
            tokens[f'CQ_{q}'] = base + i

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> str:
        # NOTE: note.chord_root 属性を事前にセットしておく前提
        c, _ = chords.get_chord(note.start)
        if not c.is_called:
            _, qualities, _ = parse_chord(c.chord)
            return qualities
        else:
            return None


#--- ChordBass トークン ------------------------------------------------
class ChordBass(ChordToken):
    def __init__(self, convert_type: int):
        super().__init__('CB', convert_type)

    def _set_tokens(self, tokens: dict):
        base = len(tokens)
        for i, r in enumerate(bases):
            tokens[f'CB_{r}'] = base + i

    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> str:
        # NOTE: note.chord_root 属性を事前にセットしておく前提
        c,_ = chords.get_chord(note.start)
        if not c.is_called:
            _, _, base = parse_chord(c.chord)
            c.is_called = True
            return base
        else:
            return None


class ChordShiftRE(ChordToken):
    def get_token(self, note: Note, chords: ChordMidi, container: ShiftTimeContainer) -> int | str | None:
        if container.is_code_mode:
            now_c, i = chords.get_chord(note.start)
            now_tick = ct_time_to_beat(now_c.time_stamp, container.tempo)
            if container.shift_measure:
                container.shift_measure = False
                shift = int(now_tick - ct_time_to_beat(container.measure_start_time, container.tempo))
                if shift < 0:
                    container.is_error = True
                return shift % 96

            if i == 0:
                #print(now_tick)
                return 0
            back_c = chords[i - 1]
            back_tick = ct_time_to_beat(back_c.time_stamp, container.tempo)
            shift = int(now_tick - back_tick)
            if shift < 0:
                container.is_error = True
            return shift % 96
        return None


    def de_convert(self, number: int | str, back_note: Note, note: Note, tempo: int, container: ShiftTimeContainer):
        pass

    def _set_tokens(self, tokens: dict):
        pass

    def __init__(self, token_type: str, convert_type: int):
        super().__init__(token_type, convert_type)
