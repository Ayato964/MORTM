import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mortm.models.mortm import MORTM, MORTMArgs
import mortm.train.tokenizer as token
import numpy as np

from mortm.train.tokenizer import TO_MUSIC
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import get_token_converter
from mortm.de_convert import ct_token_to_midi
from mortm.train.datasets import MORTM_SEQDataset
from mortm.train.train import _set_train_data, find_files

progress = _DefaultLearningProgress()
tokenizer = token.Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()

args = MORTMArgs("configs/models/mortm/A.json")
model = MORTM(progress=progress, args=args)
model.load_state_dict(torch.load("out/model/mortm/MORTM.4.0-PIANO_0.9285.pth")) # モデルをロードする。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスを設定
model.to(device)


directory, file_name = find_files("out/np/Piano/pre_train/small", ".npz")
datasets = MORTM_SEQDataset(progress, args.position_length, args.min_length)
datasets = _set_train_data(directory, file_name,datasets)
loader = DataLoader(datasets, batch_size=1, shuffle=True)
count = 0

with torch.no_grad():
    for src in loader:
        src: Tensor
        indices: Tensor = (src == 8).nonzero(as_tuple=True)[1]

        if indices is not None and len(indices) > 4:
            src = src[:, :indices[4]]
        all, gene = model.top_sampling_measure_kv_cache(src.repeat(3, 1), p=0.95, max_measure=20, temperature=1.3)
        for i in range(3):
            aya_node = [0]
            ind: Tensor = (all[i] == 585).nonzero(as_tuple=True)[0]
            if len(ind) == 0:
                continue
            ind: int = ind[0].item()
            if len(all[i]) == ind:
                aya_node.append(all[i].tolist())
            else:
                #print(all[i][:ind+1], ind)
                aya_node.append(all[i][:ind+1].tolist())
            array_dict = {f'array{c}': arr for c, arr in enumerate(aya_node)}
            if len(array_dict) > 1:
                np.savez(f"out/np/Piano/rl/eval/{count}_{i}", **array_dict)
            print(f"\r Processing...{count} phase:{i}", end="")
        count += 1
        if count >= 5000:
            break