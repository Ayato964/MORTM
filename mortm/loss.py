import math


from torch.nn import CrossEntropyLoss, Softmax
from torch import Tensor
from typing import Optional
from .reinforcement import  _reward_function
from .tokenizer import Tokenizer


class ReinforceCrossEntropy(CrossEntropyLoss):

    def __init__(self, tokenizer: Tokenizer, k=1, warmup=100, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce,
                         reduction=reduction, label_smoothing=label_smoothing)
        self.softmax = Softmax(dim=1)
        self.tokenizer = tokenizer
        self.cs: float = 1
        self.te: float = 0
        self.stepup = 0
        self.k = k
        self.warmup = warmup

        #self.step()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        score: Tensor = self.softmax(input)
        seq = score.argmax(dim=1)
        l = _reward_function(None, seq, self.tokenizer)

        return self.cs * super().forward(input, target) + self.te * l

    def step(self):

        self.stepup += 1
        self.te =0.5 * (1 + math.cos(math.pi * (self.stepup - (2 * self.warmup)) / (2 * self.warmup)))
        self.cs = 1 - self.te
        pass
