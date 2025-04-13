from mortm.convert import MIDI2Seq
from mortm.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.de_convert import ct_token_to_midi
tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

con = MIDI2Seq(tokenizer, "data/generate", "Sample4.mid", program_list=[0])

con.convert()

a, b = con.save("./out/")

tokenizer.rev_mode()

ct_token_to_midi(tokenizer, con.aya_node[1], "out/test.midi")

print(con.aya_node)