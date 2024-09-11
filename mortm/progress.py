from abc import abstractmethod
import torch


class LearningProgress:

    @abstractmethod
    def step_optimizer(self, optimizer, **kwargs):
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

    def step_optimizer(self, optimizer, **kwargs):
        optimizer.step()  # オプティマイザを更新
        optimizer.zero_grad()
        pass
