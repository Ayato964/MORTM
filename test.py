import torch
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN, TrackStart, TrackEnd, get_special_token_converter
from mortm.convert import MIDIToSequence

tokenizer = Tokenizer(special_token=get_special_token_converter(TO_TOKEN), music_token=get_token_converter(TO_TOKEN))
seq = MIDIToSequence(tokenizer, "./out", "Sample1_1.2.midi", [1])
seq()

print(seq.aya_node)
tokenizer.save("out/")