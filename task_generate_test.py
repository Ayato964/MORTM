import torch
from mortm.models.mortm import MORTM, MORTMArgs
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
import numpy as np

from mortm.train.tokenizer import TO_MUSIC

from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.de_convert import ct_token_to_midi


tokenizer:Tokenizer  = Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()
#args = MORTMArgs("configs/models/mortm/not_moe/A.json")
args = MORTMArgs("configs/models/mortm/A.json")
args.use_lora = True
model = MORTM(progress=_DefaultLearningProgress(), args=args)
model.load_state_dict(torch.load("out/model/mortm/MORTM.4.0-SAX-Phase2_0.28.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
model.to(device)

"""--------コード推定----------"""
#np_notes = np.load("out/Sample4.mid.npz")
#midi_prompt = np_notes[f'array1'][2:-190]

#start = np.array([tokenizer.get(f"k_Cm"), tokenizer.get("<QUERY_M>")] + midi_prompt.tolist() + [tokenizer.get("</QUERY_M>"), tokenizer.get("<CGEN>")])
"""----------------------------"""


"""--------旋律自己回帰生成----------"""
np_notes = np.load("out/Sample4.mid.npz")
midi_prompt = np_notes[f'array1'][2:129]

start = np.array([tokenizer.get(f"k_Cm"), tokenizer.get("<QUERY_M>")] + midi_prompt.tolist() + [tokenizer.get("</QUERY_M>"), tokenizer.get("<MGEN>")])
"""----------------------------"""


"""--------コード進行付き旋律自己回帰生成----------"""
"""
np_notes = np.load("out/Sample4.mid.npz")
midi_prompt = np_notes[f'array1'][2:98]
chord_prompt = np.array([tokenizer.get("<SME>"),
                         tokenizer.get("s_0"), tokenizer.get("CR_F"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                         tokenizer.get("s_24"), tokenizer.get("CR_G"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                         tokenizer.get("s_48"), tokenizer.get("CR_Ab"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                         tokenizer.get("s_70"), tokenizer.get("CR_Eb"), tokenizer.get("CQ_7"), tokenizer.get("CB_None"),
                         tokenizer.get("<SME>"),
                         tokenizer.get("s_0"), tokenizer.get("CR_C"), tokenizer.get("CQ_m"), tokenizer.get("CB_/A"),
                         tokenizer.get("s_48"), tokenizer.get("CR_Ab"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                         tokenizer.get("<SME>"),
                         tokenizer.get("s_0"), tokenizer.get("CR_G"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                         tokenizer.get("s_48"), tokenizer.get("CR_Gb"), tokenizer.get("CQ_aug"), tokenizer.get("CB_None"),
                         tokenizer.get("<SME>"),
                         tokenizer.get("s_0"), tokenizer.get("CR_F"), tokenizer.get("CQ_m"), tokenizer.get("CB_None"),
                         tokenizer.get("s_48"), tokenizer.get("CR_G"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),


                         ])
start = np.array([tokenizer.get(f"k_Cm"), tokenizer.get("<QUERY_M>")] + midi_prompt.tolist() + [tokenizer.get("</QUERY_M>"), tokenizer.get("<QUERY_C>")] +
                 chord_prompt.tolist() + [tokenizer.get("</QUERY_C>"), tokenizer.get("<MGEN>")])
"""
"""----------------------------"""
gene, all = model.top_p_sampling_measure(start, p=0.95, max_measure=20, temperature=1.2)

output = all
#output = torch.tensor(start)
for t in output:
    t: torch.Tensor = t
    print(f"{t}  {tokenizer.rev_get(t.tolist())}")


midi = ct_token_to_midi(tokenizer, output, "out/generate.midi", program=65, tempo=120) #生成したトークンをMIDIに変換する。

