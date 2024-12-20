import torch
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MidiToSequece
import numpy as  np
tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN))

node = np.load("./out/np/datasets/0004e1e213c9ccb9758dce0d227bd2ad.mid.npz", allow_pickle=True)
print(node['array_1'])