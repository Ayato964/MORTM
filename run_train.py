"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
import torch
import datetime
from mortm.tokenizer import Tokenizer
from messager import Messenger
from mortm.train import train_mortm
import json
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message = Messenger()
model = train_mortm("out/np/datasets/", "out/model", "0.10.0",
                    654, 3, "out/vocab/vocab_max.json", message)
