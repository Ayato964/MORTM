from mortm import constants
import torch
from mortm.mortm import MORTM
import pretty_midi as pm
import mortm.tokenizer as token
import numpy as np

from mortm.tokenizer import TO_TOKEN, TO_MUSIC

from mortm.progress import _DefaultLearningProgress
from mortm.tokenizer import get_token_converter
from mortm.de_convert import ct_token_to_midi


'''
MORTMのバージョンは常に新しくなる為、モデルのバージョンとvocab_list.jsonを確認してください。
うまくメロディが生成できない場合や、エラーが発生する場合、以下の項目を確認してください。

1.model.load_state_dictでエラーが発生する。
    -ハイパーパラメータが正しいか確認してください。モデルのバージョンによって、パラメータが異なる可能性があります。
    -CPUを使っているか、GPUを使っているかを確認してください。
    もし、CPUを使っている場合、 torch.load("model/ *** ", map_location="cpu")を設定してください。
    
2. 生成する時にエラーが発生する
    - 配列構造が不正である可能性があります。サンプリングに入力する配列は1次元配列になるはずです。
    
3. 意味不明なメロディが生成される。
    - 生成できたが、メロディとして成り立っていない場合、vocab_list.jsonが古い場合があります。
    モデルによって異なるので、再度確認してください。
'''

tokenizer = token.Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()

model = MORTM(
    progress=_DefaultLearningProgress(),
    vocab_size=393,
    d_layer=12,
    e_layer=12,
    num_heads=12,
    position_length=400
)
model.load_state_dict(torch.load("out/model/MORTM.3.0EX2_0.22.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
model.to(device)

#np_note = np.load("out/np/Sample.mid.npz", allow_pickle=True)['array_1'].tolist()
#start = np_note['key']
'''
既存の楽曲からその続きを生成する場合、以下を実行し、NPZから解凍してください。
システム的な事情でarray1からメロディが記録されています。

！MIDIから直接旋律を生成することはできません。！
!実行する際はconvert.pyモジュールを使用し、MIDIをトークンのシーケンスに変換してください。!
'''

np_notes = np.load("out/np/Sample.mid.npz")

start = np_notes[f'array1'][:-1]

'''
一から、もしくはメロディをプログラマーが設定したい場合、以下を実行します。
シーケンスは数値の配列です。tokenizerで文字列からトークンに変換してください。
'''
#start = [tokenizer.get(constants.START_SEQ_TOKEN)]

print(f"First:{start}") # ロードしたシーケンスを表示

'''
シーケンスの生成は以下の2つから選べます。
1. Top P sampling
    -これは、確率の閾値Pを設定し、それ以上に該当するトークンからサンプリングを行います。複数存在する場合、ランダムでトークンを選びます。
2. Top K sampling
    - これは、確率の高い順番からK個のトークンを取得し、サンプリングを行います。複数存在する場合、ランダムでトークンを選びます。
'''

#gene = generate_note(100, start, model, t=1.05, p=0.95)

gene = model.top_p_sampling_measure(start, p=0.95, max_measure=20, temperature=1.0, context_measure=20)
#gene = model.top_p_sampling_measure_decoder(start, p=0.95, max_measure=20, temperature=1.0)

output = gene
for t in output:
    t: torch.Tensor = t
    print(f"{t}  {tokenizer.rev_get(t.tolist())}")


midi = ct_token_to_midi(tokenizer, output, "out/generate_test.midi", program=65) #生成したトークンをMIDIに変換する。
