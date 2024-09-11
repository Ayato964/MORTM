"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.gmail_messanger import GmailMessanger
from mortm.messager import Messenger
from mortm.train import train_mortm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp')

model = train_mortm("out/np/datasets/", "out/model", "test",
                    654, 3, "out/vocab/vocab_max.json")
