"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train import train_mortm
import mortm.constants as cs
from mortm.mask import get_masks, METRIC_RANDOM_MASK
from mortm.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp')

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = train_mortm("out/np/turing/", "out/model", "test",
                    517, 5, "out/vocab/vocab_max.json",
                    begin_tuning_epoch=2,
                    fine_turing_mode=True,
                    is_save_training_progress=True,
                    src_mask_method=get_masks(tokenizer, METRIC_RANDOM_MASK), #ファインチューニング
                    load_model_directory="out/model/MORTM.1.1-b1-Horn.pth")

