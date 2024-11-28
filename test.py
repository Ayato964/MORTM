import torch
from mortm.reinforcement import _calc_loss
from mortm.loss import ReinforceCrossEntropy
print(_calc_loss(104, 64, 1))

'''
re = ReinforceCrossEntropy(None, k=1, warmup=10)

for _ in range(20):
    re.step()
'''