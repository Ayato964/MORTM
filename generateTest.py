import json

import constants
import torch
from transformer.AyatoTransFormer import AyatoModel
import pretty_midi as pm
from convert import ConvertAyaNodeToMidi as nm
import transformer.tokenizer as token
model_directory = "out/model/"

tokenizer = token.Tokenizer("out/vocab/vocab_list_before.json")

model = AyatoModel(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    dim_feedforward=3000,
    trans_layer=7,
    position_length=3000,
)
model.load_state_dict(torch.load("out/model/AyatoModel.ALL_0.9.0_0.026402872055768967.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("HEEEE")
# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")

start = tokenizer.get(constants.START_SEQ_TOKEN)

#gene = model.generate(tokenizer.get(constants.START_SEQ_TOKEN), max_length=10)
gene = model.top_p_sampling(start, tokenizer, max_length=20)
output = gene[0]
for t in output:
    print(f"{t}  {tokenizer.rev_get(t)}")

midi: pm.PrettyMIDI = nm.convert(gene[0], tokenizer)

print(midi.instruments[0].notes[0])

midi.write("out/generated/test.mid")
