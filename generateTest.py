import json

import constants
import torch
from transformer.mortem import MORTEM
import pretty_midi as pm
from convert import ConvertAyaNodeToMidi as nm
import transformer.tokenizer as token
model_directory = "out/model/"

tokenizer = token.Tokenizer("out/vocab/vocab_list.json")

model = MORTEM(
    vocab_size=654,
    d_model=1024,
    dim_feedforward=2048,
    trans_layer=12,
    num_heads=16,
    position_length=2048
)
model.load_state_dict(torch.load("out/model/AyatoModel.0.9.5_6.238151550292969.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("HEEEE")
# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")

start = tokenizer.get(-1, constants.START_SEQ_TOKEN, b=True)

#gene = model.top_p_sampling(start, tokenizer, max_length=20, temperature=0.2)
gene = model.generate_by_length(start, max_length=20)

output = gene[0]
print(output)
for t in output:
    print(f"{t}  {tokenizer.rev_get(t.tolist())}")

midi: pm.PrettyMIDI = nm.convert(gene[0], tokenizer)

print(midi.instruments[0].notes[0])

midi.write("out/generated/test.mid")
