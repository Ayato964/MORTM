"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
import torch
import datetime
from mortem.tokenizer import Tokenizer
from messager import Messenger
from mortem.train import train_mortem
import json
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message = Messenger()
model = train_mortem("out/np/datasets/", "out/model", "0.9.6",
                     654, 3, "out/vocab/vocab_max.json", message)
