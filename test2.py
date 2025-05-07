import numpy as np

from mortm.convert import MIDI2Seq
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.de_convert import ct_token_to_midi
tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

con = MIDI2Seq(tokenizer, "data/generate", "blank.mid", program_list=[0])

con.convert()

a, b = con.save("./out/")

node = np.load("out/np/Sax/dis/ai/11_1.npz")
print(node)
tokenizer.rev_mode()

ct_token_to_midi(tokenizer, node['array1'], "out/test.midi")

print(con.aya_node)