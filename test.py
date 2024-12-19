import torch
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN, TrackStart, TrackEnd
from mortm.convert import MIDIToSequence

tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN))
seq = MIDIToSequence(tokenizer, "./data/generate", "Sample2.mid", [0])
seq()

print(seq.aya_node)
tokenizer.save("out/")