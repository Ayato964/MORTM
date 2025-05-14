"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os

from mortm.train.train import train_bertm

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_bertm(human_dir="out/np/Sax/dis/human", ai_dir="out/np/Sax/dis/ai",
                    model_config="configs/models/bertm/class_file.json",
                    save_directory="out/model/class/", version="1.1Ex2",
                    train_config="configs/train/pre_training.json", )
