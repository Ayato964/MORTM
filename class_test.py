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


tokenizer = token.Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()
args = MORTMArgs("configs/class_file.json")

bertm = BERTM(
    args=args,
    progress=_DefaultLearningProgress()
)

bertm.load_state_dict(torch.load("out/model/class/BERTM.train.22.0.0148.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
bertm.to(device)
bertm.eval()

src = np.load("out/np/Sax/dis/val/7fe217c3817af19eccd2ba74905e6bbb.mid_scale_2.npz")
src = torch.Tensor(src['array3']).to(device, dtype=torch.long).unsqueeze(0)
sm = Softmax()
out = sm(bertm(src))

if out.argmax() == 0:
    print("人間")
else:
    print("AI")

midi = ct_token_to_midi(tokenizer, src[0], "out/generate_test.midi", program=65, tempo=135) #生成したトークンをMIDIに変換する。
