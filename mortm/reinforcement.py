import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List
from torch.distributions import Categorical
from torch import Tensor
from .mortm import MORTM
from .tokenizer import Tokenizer, PITCH_TYPE
from .train import _set_train_data
from .progress import _DefaultLearningProgress, LearningProgress
from .datasets import MORTM_DataSets
from .aya_node import Token
import os


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
    通常は S -> P -> Dで生成されるはずですが、もし逸脱していると3ポイント減点します。

    :param sequence: 一小節分のシーケンス
    :param tokenizer: MORTMのトークナイザー
    :return: 報酬
    '''

    reward = 0
    token_list: List[Token] = tokenizer.token_list[1:]
    token_count = 0
    for seq in enumerate(sequence):
        if not token_list[token_count].is_my_token(seq):
            reward -= 3

        token_count += 1
        if len(token_list) <= token_count:
            token_count = 0
    return reward


def compose_measure_reward(measure, tokenizer: Tokenizer):
    reward = 0
    pitch_token: Token = tokenizer.get_token_converter(PITCH_TYPE)
    measure = measure[(pitch_token.start <= measure) & (measure <= pitch_token.end)]

    for token in enumerate(measure):
        pass

    return reward


def _reward_function(base_seq, sequence, tokenizer: Tokenizer):
    seq = remove_subsequence_tensor(sequence, base_seq)
    measure_list = get_convert_measure_list(seq)
    print(measure_list)
    reward = 0
    for measure in measure_list:
        reward += (compose_sequence_reward(measure, tokenizer)
                   + compose_measure_reward(measure, tokenizer))
    return 0


def calculate_scst_loss(log_probs, sample_reward, baseline_reward):
    loss = - (sample_reward - baseline_reward) * log_probs.sum()
    print(sample_reward - baseline_reward)
    return loss


def re_train(epoch, model: MORTM, dataset_directory:str, tokenizer: Tokenizer,
             lr=8e-6, progress:LearningProgress =_DefaultLearningProgress(), goal_loss= 1.0):
    direct = os.listdir(dataset_directory)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset:MORTM_DataSets = _set_train_data(dataset_directory, direct, progress, is_fine_turing=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for e in range(1, epoch + 1):
        model.train()
        optimizer.zero_grad()

        for seq in loader:
            is_reinforcement_goal = False
            while ~is_reinforcement_goal:
                seq: Tensor = seq.clone().to(progress.get_device())
                seq = seq.squeeze(0)
                seq = seq[:-1]
                model.eval()
                # サンプル
                sample, log_probs = model.top_k_sampling_length_encoder(seq, temperature=1.1, top_k=8, max_length=500)

                # ベースライン(argmax)
                baseline = model.argmax_sampling_encoder(seq, max_length=500)

                sample_reward = _reward_function(seq, sample, tokenizer)
                baseline_reward = _reward_function(seq, baseline, tokenizer)


                model.train()
                loss = calculate_scst_loss(log_probs, sample_reward, baseline_reward)
                print(f"現在の損失は[{loss: 4f}]になっています。")
                loss.backward()
                optimizer.step()

                if loss < goal_loss:
                    is_reinforcement_goal = True

            pass
