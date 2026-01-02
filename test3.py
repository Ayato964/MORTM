
import torch

from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.models.mortm import MORTMArgs, MORTM

# 重み読み込みの例
def load_new_gate_weights(model, state_dict):
    new_state_dict = {}
    for key, val in state_dict.items():
        # Gateの重みを新しい名前にマッピング
        if "gate.weight" in key:
            new_key = key.replace("gate.weight", "gate.gate_proj.weight")
            new_state_dict[new_key] = val
        else:
            new_state_dict[key] = val

    # strict=False で読み込む（未学習のLoRAパラメータなどは無視・初期化状態になる）
    model.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully with adapted keys.")

# 実行
progress = _DefaultLearningProgress()
mortm = MORTM(MORTMArgs("./out/table/config.json"), progress)
load_new_gate_weights(mortm, torch.load("./out/table/model.pth"))

torch.save(mortm.state_dict(), f"./out/table/out.pth")