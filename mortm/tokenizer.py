import json
from typing import List
from numpy import ndarray

import re
from . import constants
from .aya_node import Token, Start,Shift,Pitch,Velocity,Duration, StartRE, MeasureToken

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

def get_token_converter(convert: int) -> List[Token]:
    register: List[Token] = list()

    register.append(MeasureToken(MEASURE_TYPE, convert))
    register.append(StartRE( START_TYPE, convert))
    register.append(Pitch( PITCH_TYPE, convert))
    #register.append(Velocity(tempo, VELOCITY_TYPE))
    register.append(Duration(DURATION_TYPE, convert))

    return register


class Tokenizer:
    def __init__(self, token: List[Token], load_data: str = None):
        if load_data is None:
            # 特殊トークン
            self.special_token_position = 3
            self.token_list = token
            self.tokens: dict = dict()
            self.tokens[constants.PADDING_TOKEN] = 0
            self.tokens[constants.START_SEQ_TOKEN] = 1
            self.tokens[constants.END_SEQ_TOKEN] = 2
            for t in token:
                t.set_tokens(self.tokens)

            self.token_max: dict = self._init_mx_dict(len(self.tokens))
            self.is_converter = False
        else:
            self.is_converter = True
            self.token_list = token
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
            if not self.is_converter:
                self.token_max[self.tokens[token]] += 1
            return self.tokens[token]

    def get_length(self, token_type: str):
        for t in self.token_list:
            if t.token_type == token_type:
                p = t.token_position
                t.token_position += 1
                return p
        pass

    def save(self, save_directory):
        json_string = json.dumps(self.tokens)
        with open(save_directory + "/vocab_list.json", 'w') as file:
            file.write(json_string)

        json_s = json.dumps(self.token_max)
        with open(save_directory + "/vocab_max.json", 'w') as file:
            file.write(json_s)

    pass

    def rev_mode(self):
        self.is_converter = True
        for li in self.token_list:
            li.convert_type = TO_MUSIC
        self.rev_tokens: dict = {v: k for k, v in self.tokens.items()}
        print(self.rev_tokens)
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
