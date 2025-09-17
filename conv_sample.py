import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDI2Seq
from mortm.utils.de_convert import ct_token_to_midi

#TEST_MIDI = "A_Beautiful_Friends.mid"
#TEST_MIDI = "blank.mid"
TEST_MIDI = "9d6db1fc171f8270c2233d8f848abd82.mid"

tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))

con = MIDI2Seq(tokenizer, "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI/9/d/6/", TEST_MIDI, program_list=[65])
#con = MIDI2Seq(tokenizer, "data/generate", TEST_MIDI, program_list=[0])

con.convert()

a, b = con.save("./out/")
node = np.load(f"out/{TEST_MIDI}.npz")
print(node)
tokenizer.save("out/vocab/")
tokenizer.mode()
ct_token_to_midi(tokenizer, node['array1'], "out/test.midi", tempo=128)

print(con.aya_node)