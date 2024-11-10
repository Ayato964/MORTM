from mortm.tokenizer import Tokenizer, TO_TOKEN, get_token_converter
from mortm.convert import MidiToAyaNode
from mortm.de_convert import ct_tokens_to_midi_b5

import torch
tokenizer = Tokenizer(get_token_converter(TO_TOKEN))
datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"

con = MidiToAyaNode(tokenizer, "out/", file_name="Sample.mid", program_list=[0, 1,2,3,4,5,6,7,8,9,10])
con.convert()

node = con.aya_node[1]

tokenizer.rev_mode()

for i in node:
    print(i, tokenizer.rev_get(i))
tokenizer.save("out/vocab/")
#midi = ct_tokens_to_midi_b5(tokenizer, torch.tensor(node), "out/d.mid")