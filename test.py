
import  numpy as np
import torch
from mortm.de_convert import ct_tokens_to_midi_b5
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MidiToAyaNode

tokenizer = Tokenizer(get_token_converter(TO_TOKEN))
co = MidiToAyaNode(tokenizer, "data/other/", "A-Beautiful-Friendship.mid", [0, 1, 2,3,4,5,6,7,8,9,10])
co.convert()
v, a = co.save("./out/np/")
print(v, a)
npz = np.load("./out/np/A-Beautiful-Friendship.mid.npz")['array1']
print(npz)

tokenizer.rev_mode()

midi = ct_tokens_to_midi_b5(tokenizer, torch.tensor(npz.tolist(), dtype=torch.long), "./out/np/Sample.mid")
