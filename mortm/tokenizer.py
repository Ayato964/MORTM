import json
from typing import List

from . import constants
from .aya_node import Token

'''旋律トークン'''
PITCH_TYPE = 'p'
VELOCITY_TYPE = 'v'
DURATION_TYPE = 'd'

'''指示トークン'''
START_TYPE = 's'
SHIFT_TYPE = 'h'


class Tokenizer:
    def __init__(self, save_directory: str,  token: List[Token], load_data: str = None):
        self.save_directory = save_directory
        if load_data is None:
            # 特殊トークン
            self.special_token_position = 3
            p = self.special_token_position
            for t in token:
                t.token_position = p
                p += t.get_range() + 1

            self.tokens: dict = dict()
            self.token_max: dict = self._init_mx_dict()
            self.tokens[constants.PADDING_TOKEN] = 0
            self.tokens[constants.START_SEQ_TOKEN] = 1
            self.tokens[constants.END_SEQ_TOKEN] = 2
        else:
            with open(load_data, 'r') as file:
                self.tokens: dict = json.load(file)
                self.rev_tokens: dict = {v: k for k, v in self.tokens.items()}


    def _init_mx_dict(self) -> dict:
        my_dict = dict()
        for i in range(0, constants.SHIFT_BEGIN_ID + 3):
            my_dict[i] = 0
        return my_dict

    def rev_get(self, a):
        return self.rev_tokens[a]

    def get(self, a: int, token_type: str, b = False):
        if a == -1:
            b = True
            my_token = token_type
        else:
            my_token = f"{token_type}_{a}"

        if my_token in self.tokens:
            if not b:
                self.token_max[self.tokens[my_token]] += 1
            return self.tokens[my_token]
        else:
            if token_type is PITCH_TYPE:
                self.tokens[my_token] = self.pitch_position
                self.pitch_position += 1

            elif token_type is VELOCITY_TYPE:
                self.tokens[my_token] = self.velocity_position
                self.velocity_position += 1

            elif token_type is DURATION_TYPE:
                self.tokens[my_token] = self.duration_position
                self.duration_position += 1

            elif token_type is START_TYPE:
                self.tokens[my_token] = self.instruction_start_position
                self.instruction_start_position += 1

            elif token_type is SHIFT_TYPE:
                self.tokens[my_token] = self.instruction_shift_position
                self.instruction_shift_position += 1

            self.token_max[self.tokens[my_token]] = 1
            return self.tokens[my_token]

    def save(self):
        json_string = json.dumps(self.tokens)
        with open(self.save_directory + "/vocab_list.json", 'w') as file:
            file.write(json_string)

        json_s = json.dumps(self.token_max)
        with open(self.save_directory + "/vocab_max.json", 'w') as file:
            file.write(json_s)

    pass