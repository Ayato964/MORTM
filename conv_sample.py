import numpy as np

from mortm.train.tokenizer import *
from mortm.utils.convert import MIDI2Seq
from mortm.utils.de_convert import ct_token_to_midi

TEST_MIDI = "ARemarkYouMade.mid"
#TEST_MIDI = "blank.mid"
#TEST_MIDI = "6a7bbda6b67fe8fe495f084e39001ac6.mid"

tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
print(tokenizer.tokens)
#con = MIDI2Seq(tokenizer, "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI/6/a/7/", TEST_MIDI, program_list=[ "SAX", "PIANO",])
con = MIDI2Seq(tokenizer, "./data/other/", TEST_MIDI, program_list=["PIANO"])

con.convert()

a, b = con.save("./out/")
print(a, b)
node = np.load(f"out/{TEST_MIDI}.npz")

tokenizer.save("out/vocab/")
tokenizer.mode()
ct_token_to_midi(tokenizer, node['array2'], "out/test.midi", tempo=160)

