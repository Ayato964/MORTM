"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train import train_mortm
import mortm.constants as cs

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp')

model = train_mortm("out/np/turing/", "out/model", "test",
                    517, 5, "out/vocab/vocab_max.json",
                    begin_tuning_epoch=1,
                    fine_turing_mode=True,
                    load_model_directory="out/model/MORTM.1.1-b1-Horn.pth")

