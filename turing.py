from mortm.convert import MIDI2Seq, MidiToAyaNode_TGT
from mortm.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
import numpy
import os
from typing import List


tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

directory = "./data/other/"
file_name = os.listdir(directory)

for file in file_name:
    con = MidiToAyaNode_TGT(tokenizer, directory, file, program_list=[65, 66])
    con_list:List[MIDI2Seq] = con.expansion_midi()
    con.convert()
    is_saved, reason = con.save("./out/np/turing/")
    print(is_saved, reason)

'''
    for c in con_list:
        c.convert()
        is_saved, reason = c.save("./out/np/turing/")
        print(reason)

    is_saved, reason = con.save("./out/np/turing/")
'''

tokenizer.save("./out/vocab/")
