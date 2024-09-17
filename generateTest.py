from mortm import constants
import torch
from mortm.mortm import MORTM
import pretty_midi as pm
from mortm.convert import ConvertAyaNodeToMidi as nm
import mortm.tokenizer as token
from mortm.progress import _DefaultLearningProgress
model_directory = "out/model/"

tokenizer = token.Tokenizer(save_directory="out/vocab/vocab_list.json", load_data="out/vocab/vocab_list.json")

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=654,
    d_model=1024,
    dim_feedforward=2048,
    trans_layer=12,
    num_heads=16,
    position_length=2048
)
model.load_state_dict(torch.load("out/model/MORTM.test_5.849682997076823.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")

start = tokenizer.get(-1, constants.START_SEQ_TOKEN, b=True)

gene = model.top_p_sampling(start, tokenizer, max_length=20, temperature=0.1)
#gene = model.generate_by_length(start, max_length=20)

output = gene[0]
print(output)
for t in output:
    print(f"{t}  {tokenizer.rev_get(t)}")

midi: pm.PrettyMIDI = nm.convert(gene[0], tokenizer)

print(midi.instruments[0].notes[0])

midi.write("out/generated/test.mid")
