import torch

import mortm.eval as ev
from mortm.mortm import MORTM
from mortm.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.progress import _DefaultLearningProgress
import numpy as np
tokenizer = Tokenizer(token=get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=518,
    position_length=8500,
    trans_layer=9, num_heads=32, d_model=1024,
    dim_feedforward=4096,)
model.load_state_dict(torch.load("out/model/MORTMv3.1.1b3-Horn.pth")) # モデルをロードする。

eval = ev.EvalPianoRoll(model, "out/generate_test.midi")
eval.view()