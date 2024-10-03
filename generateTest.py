from mortm import constants
import torch
from mortm.mortm import MORTM
import pretty_midi as pm
import mortm.tokenizer as token

from mortm.progress import _DefaultLearningProgress
from mortm.tokenizer import get_token_converter

model_directory = "out/model/"

tokenizer = token.Tokenizer(token=get_token_converter(120), load_data="out/vocab/vocab_list.json")

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=267,
    position_length=8500,
    trans_layer=12, num_heads=8, d_model=2048,
    dim_feedforward=4096
)
model.load_state_dict(torch.load("out/model/MORTM.train.1.2.6222.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")

start = [tokenizer.get(constants.START_SEQ_TOKEN), tokenizer.get("s_31.0"), tokenizer.get("p_60"), tokenizer.get("d_4"), tokenizer.get("h_1"), tokenizer.get("s_1.0")]

print(f"First:{start}")

#gene = model.top_p_sampling(start, tokenizer, max_length=20, temperature=2.0)
gene = model.p_sampling_with_temperature_sequence(start, max_length=10, temperature=1)
#gene = model.generate_by_length(start, max_length=20)

output = gene
print(output)
for t in output:
    t: torch.Tensor = t
    print(f"{t}  {tokenizer.rev_get(t.tolist())}")

#midi: pm.PrettyMIDI = nm.convert(gene[0], tokenizer)

#print(midi.instruments[0].notes[0])

#midi.write("out/generated/test.mid")
