import torch
from torch import Tensor
from typing import Callable
import random
from .tokenizer import Tokenizer, START_TYPE, DURATION_TYPE
from .progress import _DefaultLearningProgress
METRIC_RANDOM_MASK = 'mrm'

def get_masks(tokenizer: Tokenizer, get_type:str, progress=_DefaultLearningProgress()) -> Callable[[Tensor], Tensor]:
    def metric_random_mask(src: Tensor) -> Tensor:
        # インデックスの値が200 ~ 400および401 ~ 500である位置を取得
        src = src[-1, :]
        indices_1 = (src >= tokenizer.begin_token(START_TYPE)) & (src <= tokenizer.end_token(START_TYPE))  # 200 ~ 400
        indices_2 = (src >= tokenizer.begin_token(DURATION_TYPE)) & (src <= tokenizer.end_token(DURATION_TYPE))  # 401 ~ 500

        # 20%の確率でインデックスにマスクをかけるためのブール型テンソルを作成
        mask_1 = torch.tensor([random.random() < 0.4 if val else False for val in indices_1])
        mask_2 = torch.tensor([random.random() < 0.4 if val else False for val in indices_2])

        # マスクを結合
        combined_mask = mask_1 | mask_2

        return combined_mask.to(device=progress.get_device())

    if get_type is METRIC_RANDOM_MASK:
        return metric_random_mask