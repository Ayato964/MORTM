"""フル微調整トレーナ（研究設計書 v1.6 §9.1 / §4.4-1）— E2 用。

【重要】E2 は §4.4-1 により **LoRA 禁止・全パラメータ解凍・事前学習同等 LR(×倍率)** が必須。
本ドライバは LoRA 版 `train_sft.py`(r=4) を使わず、**既存の全パラ学習器 `train_mortm`** を用いる。
理由: `MORTMTrainSet` は既に (a) `load_directory` でベース(A2)重みをロード、(b) `model.parameters()`
全体を AdamW に渡す=全パラ学習、(c) `nn.CrossEntropyLoss`(全系列 AR CE, マスク無し)=事前学習と同一損失、
を満たす。よって新トレーナクラスは不要で、E2 = 「A2 チェックポイントから A1 形式(順列・削除あり)データで
全パラ継続学習」に帰着する。

E2 の要件と本実装の対応:
- LoRA 不使用          : train_mortm は LoRA を一切使わない(全パラ)。
- LR = 事前学習 LR ×倍率 : train_config の `lr_param` に倍率適用済みの値を設定。
- cosine 再スケジュール  : train_config の `scheduler`(type=cos, total_steps=FT予算) で指定。
- {1,5,10,25,50}% 途中保存 : train_config の `checkpoint_percents`=[1,5,10,25,50](§9.1)。
                           self_turing が total_steps の該当%時点で `<name>.<version>.ckpt_pN.pth` を保存。
- 起点 = A2 チェックポイント : `BASE_CKPT`(A2 の学習済み重み)を load_model_directory に渡す。
- FT データ = A1 形式(aug)   : ROOT に json_v5(A1 形式) の予算索引を渡す(§E2: 1.6B = 事前学習の50%)。

使い方(例, E2-noaug-lr{mult}):
    torchrun --nproc_per_node=2 train_fullft.py
"""
import os

from mortm.train.train import train_mortm
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.messager import _DefaultMessenger


def run_fullft(model_config, train_config, base_checkpoint, root_directory, save_directory,
               version, eval_list_json=None, project_name="MORTM_E2_FullFT", log_scale=True):
    """E2 フル微調整。train_mortm を base_checkpoint 起点で回す(全パラ・全AR CE)。
    train_config に checkpoint_percents=[1,5,10,25,50] と cosine(total_steps=FT予算) を設定しておく。"""
    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))
    train_mortm(
        tokenizer,
        model_config,
        train_config,
        root_directory,
        save_directory,
        version,
        message=_DefaultMessenger(),
        load_model_directory=base_checkpoint,   # ← A2 の学習済み重みから継続(全パラ)
        eval_list_json=eval_list_json,
        log_scale=log_scale,
        project_name=project_name,
    )


if __name__ == "__main__":
    # --- E2 の1本(例): A2(M規模, 80M) を起点に A1形式データで全パラFT ---
    # ※ BASE_CKPT/配下パスは E1 で A2 を学習後に確定する。以下は雛形。
    MODEL_CONFIG = "configs/models/mortm/foundation/80M.json"
    TRAIN_CONFIG = "configs/train/mortm/e2/fullft_lr1.0.json"   # lr=事前学習LR×1.0, checkpoint_percents=[1,5,10,25,50]
    BASE_CKPT = "out/models/paper/A2_80M/<A2 checkpoint>.pth"
    ROOT = ("/home/takaaki-nagoshi/data/scaling/json_v5/1.6B/train.json",)   # A1形式(aug) 1.6B
    EVAL = ("/home/takaaki-nagoshi/data/scaling/json_v5/eval.json",)
    SAVE_DIR = "out/models/paper/E2"
    VERSION = "E2-noaug-lr1.0"

    os.makedirs(SAVE_DIR, exist_ok=True)
    run_fullft(MODEL_CONFIG, TRAIN_CONFIG, BASE_CKPT, ROOT, SAVE_DIR, VERSION, eval_list_json=EVAL)
