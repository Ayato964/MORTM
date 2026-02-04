"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train.train import train_mortm

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.gmail_messanger import GmailMessanger, Messenger
import wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))

param = ["10M", "20M", "40M", "80M", "160M"]
for p in param:
    model = train_mortm(tokenizer, f"configs/models/mortm/4_5/research/preview4/{p}.json", f"configs/train/mortm/4_5/preview4/{p}.json",
                        "out/models/mortm/4_5/scaling_test/800M/train.json",
                        f"out/models/mortm/4_5/scaling_test/800M/",
                        f"D800M_N{p}",
                        log_scale=True,
                        project_name="MORTM4.5_Scale2",
                        eval_list_json="out/models/mortm/4_5/scaling_test/eval.json",
                        message=message)
    wandb.finish()