import torch

import mortm.eval as ev
from mortm.mortm import MORTM
from mortm.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.progress import _DefaultLearningProgress
import numpy as np

tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=393,
    position_length=400,
    e_layer=15, d_layer=15, num_heads=12, d_model=768,
    dim_feedforward=3072,)
model.load_state_dict(torch.load("out/model/MORTM.2.0-b4-SMALL-LITE_0.4687381123652983.pth")) # モデルをロードする。
model.to(_DefaultLearningProgress().get_device())

eval = ev.EvalSoftMaxScale(model, tokenizer)
np_notes = np.load("out/np/Sample.mid.npz")

start = np_notes[f'array1'][:-1]

eval.view(start)