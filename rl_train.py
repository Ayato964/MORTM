import os

from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.rl.reinforcement import *
from mortm.train.train import train_custom, collate_fn

from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter
from mortm.gmail_messanger import GmailMessanger, Messenger

ROOT_DIRECTORY = "out/model/class/val_paths_4.0.1.json"
TRAIN_CONFIG = "configs/train/rl_training.json"
MODEL_CONFIG = "configs/models/mortm/A.json"
REWARD_CONFIG = "configs/models/bertm/class_file.json"
SAVE_DIRECTORY = "out/model/mortm/"
LOAD_MORTM = "out/model/mortm/MORTM.4.0.1-PIANO_0.98.pth"
LOAD_BERTM = "out/model/class/MORTM.4.0.1_0.07.pth"
VERSION = "4.1RL"



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)
progress = _DefaultLearningProgress()

tokenizer = Tokenizer(get_token_converter(TO_MUSIC), load_data="out/vocab/vocab_list.json")
t_args = RLTrainerArgs(json_directory=TRAIN_CONFIG)
m_args = MORTMArgs(json_directory=MODEL_CONFIG)
b_args = MORTMArgs(json_directory=REWARD_CONFIG)
trainer = RLDF(t_args, m_args, b_args, progress, LOAD_MORTM, LOAD_BERTM)

model = train_custom(
        trainer=trainer,
        t_args=t_args,
        root_directory=ROOT_DIRECTORY,
        save_directory=SAVE_DIRECTORY,
        extention=".npz",
        version=VERSION,
        coll_fn=collate_fn,
        message=message)

