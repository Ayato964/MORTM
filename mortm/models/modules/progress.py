from abc import abstractmethod
import math
import torch


class LearningProgress:

    @abstractmethod
    def step_optimizer(self, optimizer, model, accumulation_steps, **kwargs):
        pass

    @abstractmethod
    def get_device(self):
        pass



class _DefaultLearningProgress(LearningProgress):

    # 勾配クリップの上限。
    max_grad_norm = 3.0
    # 警告を出す閾値。クリップ上限をこの倍数以上超えたときだけ鳴らす。
    # 旧実装の `not (1e-4 < norm < 3.0)` はクリップ上限そのものを閾値にしていたため、
    # この規模のモデルでは健全な状態でも鳴り続けていた。
    grad_norm_warn_factor = 10.0

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        pass

    def step_optimizer(self, optimizer, model, accumulation_steps, **kwargs):
        # clip_grad_norm_ はクリップ前の総ノルムを返す。
        # 旧実装はパラメータごとに .item() を呼んでいたため、optimizer step ごとに
        # パラメータ数と同じ回数のGPU同期が発生していた。ここでは1回で済ませる。
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm))

        if math.isnan(norm) or math.isinf(norm):
            print(f"\033[31m 警告\033[0m：勾配NORMがNaN/Infです({norm})。パラメータ更新をスキップして勾配をリセットします。")
            optimizer.zero_grad()
            return

        if norm > self.max_grad_norm * self.grad_norm_warn_factor:
            print(
                f"\033[31m 警告\033[0m：勾配NORMがクリップ上限({self.max_grad_norm})を大きく超えています"
                f"({norm:.4f})。学習率、またはバッチサイズを調整してください。")

        optimizer.step()
        optimizer.zero_grad()
        pass

    def get_gradient_norm(self, model):
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)  # L2ノルムを計算
                total_norm += param_norm.item() ** 2  # 勾配ノルムの2乗を足す
        total_norm = total_norm ** 0.5  # 最終的に平方根をとってL2ノルムを計算
        return total_norm
