"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import messager
import transformer.AyatoTransFormer as atf
import os
import torch
import datetime
from transformer.tokenizer import Tokenizer
from messager import Messenger
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_version = "0.9.0"
today_date = datetime.date.today().strftime('%Y%m%d')

print(f"ToDay is{datetime.date.today()}! start generating AyatoModel.{model_version}_{today_date}")

#directory = "out/np/test/"
directory = "out/np/datasets/"
datasets = os.listdir(directory)
tokenizer = Tokenizer("out/vocab/vocab_list.json")
train_data = atf.set_train_data(directory, datasets)  # 前処理されたデータをTransformerのデータセットクラスに変換する

message = Messenger()

model, loss = atf.train(train_data, message, 654, 3,
                        d_model=512,
                        dim_feedforward=4500,
                        trans_layer=7,
                        position_length=4500,
                        dropout=0.1
                        )  # 20エポック分機械学習を行う。
message.send_mail("機械学習終了のお知らせ",
                  f"AyatoModel.{model_version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {loss}")
torch.save(model.state_dict(), f"out/model/AyatoModel.{model_version}_{loss}.pth")  # できたモデルをセーブする




