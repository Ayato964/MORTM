from torch.nn.modules.loss import _Loss
import torch
from torch.nn import CrossEntropyLoss, Softmax
from torch import Tensor

from . import constants


class MORTCrossEntropyLoss(_Loss):

    def __init__(self, device, penalty=0.1, ignore_index=0, weight=None):
        super().__init__()
        self.device = device
        self.cross: CrossEntropyLoss = CrossEntropyLoss(ignore_index=ignore_index, weight=weight).to(device)
        self.softmax: Softmax = Softmax(dim=-1).to(device)
        self.penalty = penalty

    def forward(self, outputs: Tensor, targets: Tensor):
        '''
        Scoreは通常予測トークンの確率分布(Vocab_size分)がシーケンスの長さの分はいっている。
        MORTM専用のクロスエントロピーは、確率分布のうち、S -> P -> V -> D -> Hの順番を遵守していないバッチに対して
        強い減点を与えることを目的としている。

        :param outputs:
        :param targets:
        :return:
        '''
        score: Tensor = self.softmax(outputs).to(self.device)
        score: Tensor = torch.argmax(score, dim=1).to(self.device)
        score: Tensor = self.get_group_tensor(score).to(self.device)

        t_score: Tensor = self.get_group_tensor(targets.to(self.device)).to(self.device)

        mask = (score != 0) & (t_score != 0)  # 両方とも0でない位置だけを選択
        diff_count = torch.sum(score[mask] != t_score[mask])

        total_tokens = torch.sum(score).item()  # 評価対象となるトークンの数
        if total_tokens > 0:
            error_percentage = diff_count.item() / total_tokens  # 間違いの割合を計算
        else:
            error_percentage = 0  # トークンがない場合はエラーはゼロ


        return self.cross(outputs, targets) + error_percentage * self.penalty

    def get_group_tensor(self, input_tensor: Tensor) -> Tensor:
        output_tensor = torch.zeros_like(input_tensor).to(device=self.device)
        # 各範囲に対してグループ番号を割り当て
        output_tensor[torch.isin(input_tensor, torch.tensor(constants.PITCH_GROUP).to(self.device)).to(self.device)] = 1
        output_tensor[torch.isin(input_tensor, torch.tensor(constants.VELOCITY_GROUP).to(self.device)).to(self.device)] = 2
        output_tensor[torch.isin(input_tensor, torch.tensor(constants.DURATION_GROUP).to(self.device)).to(self.device)] = 3
        output_tensor[torch.isin(input_tensor, torch.tensor(constants.START_GROUP).to(self.device)).to(self.device)] = 4
        output_tensor[torch.isin(input_tensor, torch.tensor(constants.SHIFT_GROUP).to(self.device)).to(self.device)] = 5
        output_tensor[torch.isin(input_tensor, torch.tensor([1, 2]).to(self.device)).to(self.device)] = 5
        output_tensor[torch.isin(input_tensor, torch.tensor([0]).to(self.device)).to(self.device)] = 0
        return output_tensor