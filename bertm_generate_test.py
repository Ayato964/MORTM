import torch
from mortm.models.bertm import BERTM, MORTMArgs
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
import numpy as np
import torch.nn.functional as F
from mortm.train.tokenizer import TO_MUSIC

from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.de_convert import ct_token_to_midi


tokenizer:Tokenizer  = Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()
#args = MORTMArgs("configs/models/mortm/not_moe/A.json")
args = MORTMArgs("configs/models/bertm/class_file.json")
model = BERTM(progress=_DefaultLearningProgress(), args=args)
model.load_state_dict(torch.load("out/model/class/BERTM4.0-PIANO_.0.0377.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
model.to(device)

np_notes = np.load("out/np/Piano/rl/ai/\\3694_1.npz")['array1']
src = torch.tensor(np_notes).unsqueeze(0).to(device)
prompt_padding_mask = (src != 0)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        out = model(src, prompt_padding_mask)
        print(F.sigmoid(out))