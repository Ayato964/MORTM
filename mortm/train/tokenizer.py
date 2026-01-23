import json
from typing import List, Tuple

from mortm import constants
from mortm.train.custom_token import *

'''旋律トークン'''
PITCH_TYPE = 'p'
VELOCITY_TYPE = 'v'
DURATION_TYPE = 'd'

'''指示トークン'''
START_TYPE = 's'
SHIFT_TYPE = 'h'
MEASURE_TYPE = 'm'

'''変換の向き'''
TO_TOKEN = 0
TO_MUSIC = 1



def get_token_converter_pro(convert: int, old_version=True) -> List[Token]:
    register: List[Token] = list()

    register.append(EOS(convert))
    register.append(TagEnd(convert))

    register.append(PastMelody(convert))

    register.append(ConstChord(convert))
    register.append(ConstMelody(convert))

    register.append(FutureMelody(convert))

    register.append(Motif(convert))

    register.append(System(convert))
    register.append(Instrument(convert))

    register.append(GenMotif(convert))
    register.append(MGen(convert))
    register.append(CGen(convert))
    register.append(Key(convert))

    register.append(MeasureToken(convert))
    register.append(Blank(convert))

    register.append(StartRE(START_TYPE, convert))
    register.append(ChordShiftRE(START_TYPE, convert))

    register.append(ChordRoot(convert))
    register.append(ChordQuality(convert))
    register.append(ChordBass(convert))

    register.append(Pitch(PITCH_TYPE, convert))
    register.append(Duration(DURATION_TYPE, convert))

    register.append(SequenceEnd(convert))
    register.append(TrackEnd(convert))
    register.append(Meta(convert))

    register.append(GenMeasureCount(convert))
    register.append(NoteDense(convert))
    return register


def get_token_converter_melody_only(convert: int) -> List[Token]:
    register: List[Token] = list()

    register.append(EOS(convert))

    register.append(MGen(convert))
    register.append(CGen(convert))
    register.append(Key(convert))

    register.append(MeasureToken(convert))
    register.append(Blank(convert))
    register.append(StartRE(START_TYPE, convert))

    register.append(Pitch(PITCH_TYPE, convert))
    register.append(Duration(DURATION_TYPE, convert))

    register.append(SequenceEnd(convert))
    register.append(TrackEnd(convert))

    return register


def bertm_converter(convert: int) -> List[Token]:
    register: List[Token] = get_token_converter_pro(convert)
    register.append(Eval(convert))
    register.append(Human(convert))
    register.append(AI(convert))
    return register

def omega_converter(convert: int) -> List[Token]:
    register: List[Token] = get_token_converter_pro(convert)
    register.append(GenMeasureCount(convert))
    return register


class Tokenizer:
    def __init__(self,music_token: List[Token], load_data: str = None):
        if load_data is None:

            self.music_token_list = music_token
            self.tokens: dict = dict()
            self.tokens[constants.PADDING_TOKEN] = 0
            for t in music_token:
                t.set_tokens(self.tokens)

            self.token_max: dict = self._init_mx_dict(len(self.tokens))
            self.rev_tokens = None
        else:
            self.music_token_list = music_token
            with open(load_data, 'r') as file:
                self.tokens: dict = json.load(file)
                self.rev_tokens: dict = {v: k for k, v in self.tokens.items()}

    def _init_mx_dict(self, mx) -> dict:
        my_dict = dict()
        for i in range(0, mx):
            my_dict[i] = 0
        return my_dict

    def rev_get(self, a):
        return self.rev_tokens[a]

    def get(self, token: str):
        if token is not None:
            if token not in self.tokens and "<" not in token:
                sp = token.split('_')
                length = self.get_length(sp[0])

                self.tokens[token] = length
            return self.tokens[token]

    def get_length(self, token_type: str):
        for t in self.music_token_list:
            if t.token_type == token_type:
                p = t.token_position
                t.token_position += 1
                return p
        pass

    def get_length_tuple(self, token_type: str) -> Tuple[int, int]:
        for t in self.music_token_list:
            if t.token_type == token_type:
                return t.get_token_length_tuple()
        pass

    def save(self, save_directory):
        json_string = json.dumps(self.tokens)
        with open(save_directory + "/vocab_list.json", 'w') as file:
            file.write(json_string)

        json_s = json.dumps(self.token_max)
        with open(save_directory + "/vocab_max.json", 'w') as file:
            file.write(json_s)

    def mode(self, to=TO_MUSIC):
        for li in self.music_token_list:
            li.convert_type = to

        if to == TO_MUSIC and self.rev_tokens is None:
            self.rev_tokens: dict = {v: k for k, v in self.tokens.items()}
        #print(self.rev_tokens)
        pass

    def begin_token(self, token_type):
        for i in range(len(self.rev_tokens)):
            if token_type in self.rev_tokens[i] :
                return i
        return "NOT FOUND"

    def end_token(self, token_type):
        for i in range(len(self.rev_tokens)):
            token = len(self.rev_tokens) - 1 - i
            if token_type in self.rev_tokens[token]:
                return token

    def get_token_converter(self, token_type) -> Token:
        for token in self.music_token_list:
            if token_type == token.token_type:
                return token
