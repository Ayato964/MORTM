"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train.train import train_mortm

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.gmail_messanger import GmailMessanger, Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))

model = train_mortm(tokenizer, "configs/models/mortm/4_5/research/preview3/pro.json", "configs/train/pre_training.json",
                    "C:/Users/Nagoshi Takaaki.KTHRLab/MORTM/pre_train/ver2",
                    "out/models/mortm/4_5/",
                    "Scaling_",
                    log_scale=False,
                    project_name="MORTM4.5_Scale",
#                    eval_list_json="out/models/mortm/4_5/eval_paths_4.5-Pro-Preview-4.json",
                    message=message)