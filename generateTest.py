from mortm import constants
import torch
from mortm.mortm import MORTM
import pretty_midi as pm
import mortm.tokenizer as token
import numpy as np

from mortm.tokenizer import TO_TOKEN, TO_MUSIC

from mortm.progress import _DefaultLearningProgress
from mortm.tokenizer import get_token_converter
from mortm.de_convert import ct_tokens_to_midi_b5, ct_token_to_midi_1_0
model_directory = "out/model/"

tokenizer = token.Tokenizer(token=get_token_converter(120, TO_MUSIC), load_data="out/vocab/vocab_list.json")

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=267,
    position_length=8000,
    trans_layer=9, num_heads=16, d_model=1024,
    dim_feedforward=2048
)
model.load_state_dict(torch.load("out/model/MORTM.error_end.1.2.0406.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
np_notes = np.load("out/np/Sample.mid.npz")

start = np_notes[f'array1'][:-1]

#start = [tokenizer.get(constants.START_SEQ_TOKEN)]

print(f"First:{start}")

#gene = model.top_p_sampling(start, tokenizer, max_length=20, temperature=2.0)
gene = model.top_k_sampling_with_temperature_sequence(start, max_length=500, temperature=0.8, top_k=5)

#gene = model.generate_by_length(start, max_length=20)

output = gene
for t in output:
    t: torch.Tensor = t
    print(f"{t}  {tokenizer.rev_get(t.tolist())}")


midi = ct_token_to_midi_1_0(tokenizer, output, "out/generate_test.midi")
