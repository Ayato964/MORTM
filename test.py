import torch
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MIDIToSequence, MidiToSequece
from mortm.de_convert import ct_token_to_midi
import numpy as  np

tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN))

con = MidiToSequece(tokenizer, "data/generate", "Sample2.mid", [0])

con()

i, r = con.save("out/")
print(con.sequence_dict)
print(i, r)
tokenizer.rev_mode()
#ct_token_to_midi(tokenizer, con.aya_node[1], save_directory="out/generate.mid", tempo=180, program=65)