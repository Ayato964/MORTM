"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train import train_mortm
import mortm.constants as cs
from mortm.mask import get_masks, METRIC_RANDOM_MASK
from mortm.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=50000)

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_mortm(tokenizer, "out/np/datasets/", "out/model", "1.1-b3",
                    518, 50, "out/vocab/vocab_max.json",
                    is_save_training_progress=True,
                    warmup_steps=12000,
                    message=message,



                    load_model_directory="out/model/MORTMv3.1.1-b2-Horn.pth")

