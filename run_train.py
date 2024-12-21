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

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_mortm(tokenizer, "out/np/datasets/", "out/model", "1.1-b5",
                    393, 50, "out/vocab/vocab_max.json",
                    is_save_training_progress=True,
                    d_model=1024,
                    dim_feedforward=4096,
                    d_layer=12,
                    e_layer=9,
                    warmup_steps=4000,
                    num_heads=16,
                    batch_size=1,
                    accumulation_steps=32,
                    message=message)

                    #load_model_directory="out/model/MORTM.error_end.18.pth")

