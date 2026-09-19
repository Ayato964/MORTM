import os
from mortm.train.train import train_mortm
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.messager import _DefaultMessenger

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    message = _DefaultMessenger()

    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))

    # 再開するチェックポイントファイルのパスを明示的に指定
    checkpoint_path = "out/models/mortm/4_5/MORTM.4.5E-A80M-E128.checkpoint.pt"

    # 事前学習済み重み（.pth）からモデル重みのみを初期化ロードしたい場合はパスを指定（Noneで使用しない）
    load_model_path = None

    model = train_mortm(
        tokenizer,
        "configs/models/mortm/foundation/A80M_E128.json",
        "configs/train/mortm/foundation/A80M_E128.json",
        ("out/music_train.json", "out/cm_train.json"),
        "out/models/mortm/4_5/",
        "4.5E-A80M-E128",
        load_model_directory=load_model_path,
        log_scale=True,
        project_name="MORTM4.5_Foundation",
        message=message,
        eval_list_json=("out/music_eval.json", "out/cm_eval.json"),
        resume=True,
        resume_checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()

