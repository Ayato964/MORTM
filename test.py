import torch
import random

def create_mask(tensor):
    # インデックスの値が200 ~ 400および401 ~ 500である位置を取得
    indices_1 = (tensor >= 200) & (tensor <= 400)  # 200 ~ 400
    indices_2 = (tensor >= 401) & (tensor <= 500)  # 401 ~ 500

    # 20%の確率でインデックスにマスクをかけるためのブール型テンソルを作成
    mask_1 = torch.tensor([random.random() < 0.4 if val else False for val in indices_1])
    mask_2 = torch.tensor([random.random() < 0.4 if val else False for val in indices_2])

    # マスクを結合
    combined_mask = mask_1 | mask_2

    return combined_mask

# サンプルテンソルの作成
tensor = torch.tensor([1, 100, 400, 200, 500, 300])  # 長さ6のテンソル
masked_tensor = create_mask(tensor)
print(masked_tensor)
