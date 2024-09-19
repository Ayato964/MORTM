"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train import train_mortm
import mortm.constants as cs

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp')

model = train_mortm("out/np/datasets/", "out/model", "test",
                    cs.SHIFT_BEGIN_ID + 4, 3, "out/vocab/vocab_max.json",
                    position_length=2500, dim_feedforward=2500, batch_size=8, accumulation_steps=1)
