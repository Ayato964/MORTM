
import  numpy as np
import torch
from mortm.de_convert import ct_tokens_to_midi
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC

n = np.load("out/np/datasets2/002677f2e6e4666fdf9827d28bfa639f.mid.npz")

print(n['array1'])

tokenizer = Tokenizer(get_token_converter(120, TO_MUSIC), "out/vocab/vocab_list.json")
ct_tokens_to_midi(tokenizer, torch.tensor(n['array1']), "out/test.midi")


