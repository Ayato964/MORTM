"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train import train_mortm

from mortm.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_mortm("configs/512_3.6B.json", "out/np/datasets5_large/", "out/model", "3.1t6-LARGE",
                    #load_model_directory="out/model/MORTM.3.0t6-MEDIUM_0.33.pth"
                    num_epochs=30,
                    train_dataset_split=0.99,
                    message=message,
                    is_save_training_progress=True,
                    batch_size=16,
                    #lr_param=5e-6,
                    accumulation_steps=1,
                    warmup_steps=4000)

