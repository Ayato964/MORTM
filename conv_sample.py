import numpy as np

from mortm.convert import MIDI2Seq, Midi2SeqWithChord
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.de_convert import ct_token_to_midi

TEST_MIDI = "bf3f2ad01a2eee92407fb6870b7360ff.mid"

tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

#con = MIDI2Seq(tokenizer, "data/generate", TEST_MIDI, program_list=[0])

#con.convert()

#a, b = con.save("./out/")

node = np.load(f"out/{TEST_MIDI}.npz")
#print(node)
tokenizer.rev_mode()
#ct_token_to_midi(tokenizer, node['array1'], "out/test.midi")

#print(con.aya_node)