import math

import torch
from typing import List
from torch import Tensor
from .tokenizer import Tokenizer, PITCH_TYPE

from .aya_node import Token, _get_symbol


def remove_subsequence_tensor(a, b):
    len_b = b.shape[0]
    for i in range(a.shape[0] - len_b + 1):
        if torch.equal(a[i:i+len_b], b):
            return torch.cat((a[:i], a[i+len_b:]))
    return a


def get_convert_measure_list(sequence):
    '''
    シーケンスから小節の区切り目のトークンを境目にスプリットし、その配列を返します。
    :param sequence:  小節ごとに区切られたシーケンスの配列
    :return: sequence
    '''
    segments = []
    start_idx = None
    for i, val in enumerate(sequence):
        #print(i, val)
        if val == 3:
            if start_idx is not None and i > start_idx:
                segments.append(sequence[start_idx:i])
            start_idx = i + 1
    return segments


def compose_sequence_reward(sequence: Tensor, tokenizer: Tokenizer):
    '''
    シーケンスを受け取り、正しい生成の順番であるかを評価します。
    通常は S -> P -> Dで生成されるはずです.


    :param sequence: 一小節分のシーケンス
    :param tokenizer: MORTMのトークナイザー
    :return: 報酬
    '''

    reward_count = 0
    token_list: List[Token] = tokenizer.music_token_list[1:]
    token_count = 0
    for seq in enumerate(sequence):
        if not token_list[token_count].is_my_token(seq):
            reward_count += 1

        token_count += 1
        if len(token_list) <= token_count:
            token_count = 0

    reward = _calc_loss(reward_count if token_count == 0 else 10, 0, 1e-4)

    return reward




def compose_measure_reward(measure, tokenizer: Tokenizer, last_duration):
    pitch_token: Token = tokenizer.get_token_converter(PITCH_TYPE)

    measure = measure[(pitch_token.start <= measure) | (measure >= pitch_token.end)]
    i = 0
    duration = 0
    ld = 0
    while i < len(measure):
        if i == 0:
            if last_duration is None:
                duration += _get_symbol(tokenizer.rev_get(measure[i].item()))
            else:
                duration += _get_symbol(tokenizer.rev_get(measure[i].item())) - last_duration
        elif i == len(measure) - 1:
            ld = _get_symbol(tokenizer.rev_get(measure[i].item()))
            if duration < 64 < duration + ld:
                duration = 64
            else:
                duration += ld
        elif i % 2 == 0:
            duration += _get_symbol(tokenizer.rev_get(measure[i].item())) - _get_symbol(tokenizer.rev_get(measure[i - 1].item()))
        else:
            duration += _get_symbol(tokenizer.rev_get(measure[i].item()))

        i += 1
    reward = _calc_loss(duration, 64, 1e-5)

    return reward, ld


def _calc_loss(x, n:int, k:float):
    return math.e ** (k * abs(x - n)) - 1


def _reward_function(base_seq, sequence, tokenizer: Tokenizer):
    if base_seq is not None:
        seq = remove_subsequence_tensor(sequence, base_seq)
    else:
        seq = sequence
    measure_list = get_convert_measure_list(seq)
    reward = 100 if len(measure_list) is 0 else 0
    last_duration = None
    for measure in measure_list:
        cmr, ld = compose_measure_reward(measure, tokenizer, last_duration)
        cmr /= len(measure_list)
        last_duration = ld
        csr = compose_sequence_reward(measure, tokenizer) / len(measure_list)
        reward += (csr + cmr)

    return reward


def calc_mortm_loss(probs: Tensor, reward):
    loss = -(reward * probs.sum()) / 200
    return loss

