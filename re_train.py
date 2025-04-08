
from mortm.reinforcement import re_train
from mortm.models.mortm import MORTM
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.models.modules.progress import LearningProgress, _DefaultLearningProgress
import torch

tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")
progress: LearningProgress = _DefaultLearningProgress()


model = MORTM(
    progress=progress,
    vocab_size=518,
    position_length=8500,
    trans_layer=9, num_heads=32, d_model=1024,
    dim_feedforward=4096,
).to(progress.get_device())

model.load_state_dict(torch.load("out/model/MORTM.re.1.137_loss.0.0000.pth"))

re_train(1, model, "out/np/turing/", tokenizer)
