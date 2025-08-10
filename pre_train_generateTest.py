import torch
from torch import compile
from mortm.models.mortm import MORTM, MORTMArgs
import mortm.train.tokenizer as token
import numpy as np

from mortm.train.tokenizer import TO_MUSIC

from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import get_token_converter
from mortm.de_convert import ct_token_to_midi
'''
MORTMのバージョンは常に新しくなる為、モデルのバージョンとvocab_list.jsonを確認してください。
うまくメロディが生成できない場合や、エラーが発生する場合、以下の項目を確認してください。

1.model.load_state_dictでエラーが発生する。
    -ハイパーパラメータが正しいか確認してください。モデルのバージョンによって、パラメータが異なる可能性があります。
    -CPUを使っているか、GPUを使っているかを確認してください。
    もし、CPUを使っている場合、 torch.load("model/ *** ", map_location="cpu")を設定してください。

'''

tokenizer =token.Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.mode()
#args = MORTMArgs("configs/models/mortm/not_moe/A.json")
args = MORTMArgs("configs/models/mortm/A.json")

model = MORTM(progress=_DefaultLearningProgress(), args=args)
model.load_state_dict(torch.load("out/model/mortm/MORTM.4.0.1-PIANO_0.98.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
model.to(device)
model = compile(model)

'''
既存の楽曲からその続きを生成する場合、以下を実行し、NPZから解凍してください。
システム的な事情でarray1からメロディが記録されています。

！MIDIから直接旋律を生成することはできません。！
!実行する際はconvert.pyモジュールを使用し、MIDIをトークンのシーケンスに変換してください。!
'''

""""------ 旋律の自己回帰生成を行う場合------"""
np_notes = np.load("out/Sample4.mid.npz")
start = np_notes[f'array1'][:128]
print(start)
"""-------------------------------------"""

"""------- コードの自己回帰生成を行う場合-----"""
#start = np.load("out/np/Sax/pre-train/Phase2/with_chord/a5360659764a37938776844025fadc61.mid.npz")["array4"] # 既存のメロディをロードする。
#print(f"Length: {len(start)}")

#start = np.array([tokenizer.get("<CGEN>"), tokenizer.get("k_Cm"), tokenizer.get("<SME>"),
#                  tokenizer.get("s_0"), tokenizer.get("CR_Ab"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),tokenizer.get("<SME>"),
#                  tokenizer.get("s_0"), tokenizer.get("CR_G"), tokenizer.get("CQ_7"), tokenizer.get("CB_None"),])

"""--------------------------------------"""

all, gene = model.top_sampling_measure_kv_cache(start, p=0.95, max_measure=20, temperature=1.0)

output = all
#output = torch.tensor(start)
count = 0
for out in output:
    for t in out:
        t: torch.Tensor = t
        print(f"{t}  {tokenizer.rev_get(t.tolist())}")
    midi = ct_token_to_midi(tokenizer, out, f"out/generate_{count}.midi", program=0, tempo=120) #生成したトークンをMIDIに変換する。
    count += 1