import torch
from mortm.models.mortm import MORTM, MORTMArgs
import mortm.train.tokenizer as token
import numpy as np

from mortm.train.tokenizer import TO_MUSIC

from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import get_token_converter
from mortm.de_convert import ct_token_to_midi
from mortm.models.bertm import BERTM, MORTMArgs
from torch.nn.modules import Softmax
from torch.nn import functional as F

tokenizer = token.Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()
args = MORTMArgs("configs/models/bertm/class_file.json")

bertm = BERTM(
    args=args,
    progress=_DefaultLearningProgress()
)

bertm.load_state_dict(torch.load("out/model/class/MORTM.train.9.0.1906.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
bertm.to(device)
bertm.eval()

src = np.load("out/generate_test.midi")
src = torch.Tensor(src['array1']).to(device, dtype=torch.long).unsqueeze(0)
sm = Softmax()
out = bertm(src)

print(F.sigmoid(out)) # 出力の形状を表示する。

midi = ct_token_to_midi(tokenizer, src[0], "out/generate_test.midi", program=65, tempo=135) #生成したトークンをMIDIに変換する。
