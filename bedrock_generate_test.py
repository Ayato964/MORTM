from mortm.models.modules.config import MORTMArgs
from mortm.models.mortm import MORTM
from mortm.utils.generate import *
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import Tokenizer, TO_TOKEN, get_token_converter

midi_path = "data/generate/Sample4.mid"
args_path = "configs/models/mortm/A.json"
model_save_path = "out/model/mortm/MORTM.4.0EX5-SAX-Phase1_1.4054.pth"
program = [0]

if __name__ == "__main__":
    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))
    args = MORTMArgs(args_path)
    p = _DefaultLearningProgress()
    model = MORTM(progress=p, args=args)
    model.load_state_dict(torch.load(model_save_path))  # Load the model
    model.to(p.get_device())

    pre_train_generate(model, tokenizer, "out", midi_path, split_measure=4, program=program, temperature=1.1)
