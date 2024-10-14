from mortm import constants
import torch
from mortm.mortm import MORTM
import pretty_midi as pm
import mortm.tokenizer as token
import numpy as np

from mortm.progress import _DefaultLearningProgress
from mortm.tokenizer import get_token_converter

model_directory = "out/model/"

tokenizer = token.Tokenizer(token=get_token_converter(120), load_data="out/vocab/vocab_list.json")

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=327,
    position_length=8500,
    trans_layer=12, num_heads=16, d_model=2048,
    dim_feedforward=4096
)
model.load_state_dict(torch.load("out/model/MORTM.train.0.3.6299.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
#np_notes = np.load("out/np/0a0e71e3f20423688a1321a23501c443.mid.npz")

#start = np_notes[f'array1'][:50]

start = [tokenizer.get(constants.START_SEQ_TOKEN)]

print(f"First:{start}")

#gene = model.top_p_sampling(start, tokenizer, max_length=20, temperature=2.0)
gene = model.top_k_sampling_with_temperature_sequence(start, max_length=100, temperature=1.0, top_k=5)

#gene = model.generate_by_length(start, max_length=20)

output = gene
for t in output:
    t: torch.Tensor = t
    print(f"{t}  {tokenizer.rev_get(t.tolist())}")

#midi: pm.PrettyMIDI = nm.convert(gene[0], tokenizer)

#print(midi.instruments[0].notes[0])

#midi.write("out/generated/test.mid")
