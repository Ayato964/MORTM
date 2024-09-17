from mortm.convert import MidiToAyaNode, AyaNodeToMidi
from mortm.tokenizer import Tokenizer

tokenizer = Tokenizer("out/vocab/")
print(tokenizer.instruction_shift_position)
BRASS = [57, 58, 65, 66, 67, 68]

con = MidiToAyaNode(tokenizer, "data/other", "along.mid", program_list=BRASS)
con.convert()
con.save("out")
tokenizer.save()

dec = AyaNodeToMidi("out/along.mid.npz")

print(dec.npz_dict['array1'])
