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
    trans_layer=12, num_heads=8, d_model=1024,
    dim_feedforward=2048
)
model.load_state_dict(torch.load("out/model/MORTM.train.8.3.5537.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")

start = tokenizer.get(constants.START_SEQ_TOKEN)

gene = model.top_p_sampling(start, tokenizer, max_length=20, temperature=0.3)
#gene = model.generate_by_length(start, max_length=20).tolist()

output = gene[0]
print(output)
for t in output:
    print(f"{t}  {tokenizer.rev_get(t)}")

#midi: pm.PrettyMIDI = nm.convert(gene[0], tokenizer)

#print(midi.instruments[0].notes[0])

#midi.write("out/generated/test.mid")
