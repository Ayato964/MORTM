import os
from mortm.train.train import train_mortm
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
from mortm.utils.messager import _DefaultMessenger

local_rank = int(os.environ.get("LOCAL_RANK", 0))

message = _DefaultMessenger()

tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))

model = train_mortm(
    tokenizer,
    "configs/models/mortm/4_5/research/preview3/pro.json",
    "configs/train/mortm/4_5/preview3/80M.json",
    "out/models/mortm/4_5/preview3/train_updated.json",
    "out/models/mortm/4_5/preview3/",
    "4.5E-LITE",
    log_scale=True,
    project_name="MORTM4.5_Scale2",
    eval_list_json="out/models/mortm/4_5/preview3/eval_updated.json",
    message=message
)