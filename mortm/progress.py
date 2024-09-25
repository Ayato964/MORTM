from abc import abstractmethod
import torch



class LearningProgress:

    @abstractmethod
    def step_optimizer(self, optimizer, model, accumulation_steps, **kwargs):
        pass

    @abstractmethod
    def get_device(self):
        pass


class _DefaultLearningProgress(LearningProgress):

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        pass

    def step_optimizer(self, optimizer, model, accumulation_steps, **kwargs):

        # パラメータの勾配を累積ステップ数でスケーリング
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= accumulation_steps

        optimizer.step()  # オプティマイザを更新

        print(f"現在のNORMは{self.get_gradient_norm(model)}です。")

        optimizer.zero_grad()
        pass

    def get_gradient_norm(self, model):
        total_norm = 0
        # モデルのすべてのパラメータについて、勾配がNoneでないものを対象にする
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)  # L2ノルムを計算
                total_norm += param_norm.item() ** 2  # 勾配ノルムの2乗を足す
        total_norm = total_norm ** 0.5  # 最終的に平方根をとってL2ノルムを計算
        return total_norm
