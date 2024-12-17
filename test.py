import torch
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.reinforcement import _calc_loss
from mortm.loss import ReinforceCrossEntropy
from mortm.train import _set_train_data
from mortm.progress import _DefaultLearningProgress
from torch.utils.data import DataLoader

print(_calc_loss(104, 64, 1))


tokenizer = Tokenizer(token=get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")
re = ReinforceCrossEntropy(tokenizer, k=1, warmup=10)
progress = _DefaultLearningProgress()
dataset = _set_train_data("out/np/datasets/", ["9be9b0430506d0f3f310417b88eab8e0.mid.npz"], progress)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for input in dataloader:
    input = input[0]
    loss = re(input, input)
    print(loss)