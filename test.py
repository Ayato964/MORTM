
import  numpy as np
import torch
from mortm.de_convert import ct_tokens_to_midi_b5
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MidiToAyaNode, MidiToAyaNode_TGT
from mortm.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.de_convert import ct_tokens_to_midi_b5

tokenizer = Tokenizer(get_token_converter(TO_TOKEN), load_data="out/vocab/vocab_list.json")

con = MidiToAyaNode_TGT(tokenizer, "./data/other/", "DontDreamOfAnybodyButMe.mid", [65, 66])
con()

is_saved, reason = con.save("out/")
print(is_saved, reason)

seq = np.load("./out/DontDreamOfAnybodyButMe.mid.npz", allow_pickle=True)
print(seq['array1'])