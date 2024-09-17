from mortm.convert import MidiToAyaNode, AyaNodeToMidi
from mortm.tokenizer import Tokenizer
from mortm.train import _set_train_data
from mortm.progress import _DefaultLearningProgress
import os

'''
tokenizer = Tokenizer("out/vocab/")
print(tokenizer.instruction_shift_position)
BRASS = [57, 58, 65, 66, 67, 68]

con = MidiToAyaNode(tokenizer, "data/other", "along.mid", program_list=BRASS)
con.convert()
con.save("out")
tokenizer.save()

dec = AyaNodeToMidi("out/along.mid.npz")

print(dec.npz_dict['array1'])

'''

datasets = os.listdir("out/np/datasets")
_set_train_data("out/np/datasets/", datasets, _DefaultLearningProgress())
