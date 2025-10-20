"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import os
from mortm.train.train import train_custom, VisionTrainSet, TrainArgs

from mortm.models.modules.config import MORTM_LIVE_Args
from mortm.utils.gmail_messanger import GmailMessanger, Messenger
from mortm.models.modules.progress import LearningProgress

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

message: Messenger = GmailMessanger("token.json", "client_secret.json", 'nagoshi@kthrlab.jp', step_by_message_count=100000)

args = MORTM_LIVE_Args("configs/models/live/A.json")
t_args = TrainArgs("configs/train/vision_train.json")
model = train_custom(
    trainer=VisionTrainSet(args, LearningProgress()),
    t_args=t_args,
    root_directory="C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI",
    extention=".mid",
    save_directory="out/model/mortm_live/",
    version="Research_vision",
    message=message)