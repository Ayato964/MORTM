"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
#from mortm.train import train_mortm
from mortm.train_decoder import train_mortm
import mortm.constants as cs
from mortm.mask import get_masks, METRIC_RANDOM_MASK
from mortm.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_mortm(tokenizer, "out/np/datasets_small_decoder/", "out/model", "2.0-LARGE-LITE",
                    393, 10, "out/vocab/vocab_max.json",
                    train_dataset_split=0.99,
                    is_save_training_progress=True,
                    position_length=320,
                    d_model=768,
                    dim_feedforward=3072,
                    #d_layer=15,
                    e_layer=15,
                    warmup_steps=4000,
                    num_heads=12,
                    batch_size=16,
                    accumulation_steps=1,
                    message=message,)
                    #load_model_directory="out/model/MORTM.2.0-SMALL-LITE_0.25.pth")
