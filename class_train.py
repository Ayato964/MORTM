"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os

from mortm.train.re_train import train_bertm

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_bertm(human_dir="out/np/Sax/dis/human", ai_dir="out/np/Sax/dis/ai",
                    args_dir="configs/class_file.json",
                    save_directory="out/model/class/",version="1.0",
                    train_split=0.99,
                    message=message,
                    epoch=100, batch_size=4,
                    warmup_steps=4000,
                    accumlation_steps=2,
                    #lr_param=None,
                    is_save_training_progress=True)

