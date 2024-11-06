
import  numpy as np
import torch
from mortm.de_convert import ct_tokens_to_midi_b5
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MidiToAyaNode
from mortm.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.de_convert import ct_tokens_to_midi_b5

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")

sample = np.load("out/np/turing/grooving_hard.mid.npz")['array3']
print(sample)
midi = ct_tokens_to_midi_b5(tokenizer, torch.tensor(sample, dtype=torch.long), "out/sample.mid")