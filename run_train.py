"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train.train import train_mortm

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_mortm("configs/models/mortm/B.json", "configs/train/pre_training.json",
                    "out/np/research/", "out/model/mortm/",
                    "Research_without_MoE",
                    message=message)


