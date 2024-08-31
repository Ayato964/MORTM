"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import messager
import mortem.mortem as atf
import os
import torch
import datetime
from mortem.tokenizer import Tokenizer
from messager import Messenger
import json
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_version = "0.9.2"
today_date = datetime.date.today().strftime('%Y%m%d')

print(f"ToDay is{datetime.date.today()}! start generating MORTEM_Model.{model_version}_{today_date}")

#directory = "out/np/test/"
directory = "out/np/datasets/"
datasets = os.listdir(directory)
tokenizer = Tokenizer("out/vocab/vocab_list.json")
train_data = atf.set_train_data(directory, datasets)  # 前処理されたデータをTransformerのデータセットクラスに変換する

message = Messenger()

try:
    with open("out/vocab/vocab_max.json", 'r') as file:
        freq_dict = json.load(file)
        # 逆数を取り、頻出度が0の場合は小さい値に設定
        epsilon = 1e-10  # 非ゼロの小さい値を設定しておく
        weights = []

        for i in range(len(freq_dict)):
            freq = freq_dict[str(i)]  # JSONのキーは文字列なのでstrに変換
            if freq == 0:
                weights.append(epsilon)
            else:
                weights.append(1.0 / freq)
        # テンソルに変換
        weight_tensor = torch.tensor(weights)
        weight_tensor = weight_tensor / weight_tensor.sum()
    model, loss = atf.train(train_data, message, 654, 1, weight_tensor,
                            d_model=1024,
                            dim_feedforward=2048,
                            trans_layer=12,
                            num_heads=16,
                            position_length=2048,
                            dropout=0.2
                            )  # 20エポック分機械学習を行う。

    message.send_mail("機械学習終了のお知らせ",
                  f"AyatoModel.{model_version}の機械学習が終了しました。 \n 結果の報告です。\n 損失関数: {loss}")
    torch.save(model.state_dict(), f"out/model/AyatoModel.{model_version}_{loss}.pth")  # できたモデルをセーブする
except torch.cuda.OutOfMemoryError:
    message.send_mail("エラーが発生し、処理を中断しました", "学習中にモデルがこのPCのメモリーの理論値を超えました。\nバッチサイズを調整してください")



