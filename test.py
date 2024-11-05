
from mortm.convert import MidiToAyaNode
from mortm.tokenizer import Tokenizer, get_token_converter, TO_TOKEN

tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

con = MidiToAyaNode(tokenizer, directory="data/other", file_name="2ndtime.mid", program_list=[65, 66])
con_lists = con.expansion_midi()
print(len(con_lists))