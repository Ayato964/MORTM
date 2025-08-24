from mortm.models.modules.config import MORTMArgs
from mortm.models.mortm import MORTM
from mortm.utils.generate import *
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import Tokenizer, TO_TOKEN, get_token_converter

midi_path = "data/generate/Sample4.mid"
args_path = "configs/models/mortm/A.json"
#model_save_path = "out/model/mortm/MORTM.4.0EX5-SAX-Phase1_1.4054.pth"
model_save_path = "out/model/mortm/MORTM.4.1-SAX-Phase2_0.29.pth"
sft_model = True
program = [0]

if __name__ == "__main__":
    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))
    args = MORTMArgs(args_path)
    if sft_model:
        args.use_lora = True
    p = _DefaultLearningProgress()
    model = MORTM(progress=p, args=args)
    model.load_state_dict(torch.load(model_save_path))  # Load the model
    model.to(p.get_device())

    if not sft_model:
        pre_train_generate(model, tokenizer, "out", midi_path, split_measure=4, program=program, temperature=1.0)
    else:
        """
        chord = np.array([tokenizer.get("<SME>"),
                          tokenizer.get("s_0"), tokenizer.get("CR_F"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                          tokenizer.get("s_24"), tokenizer.get("CR_G"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                          tokenizer.get("s_48"), tokenizer.get("CR_Ab"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                          tokenizer.get("s_70"), tokenizer.get("CR_Eb"), tokenizer.get("CQ_7"), tokenizer.get("CB_None"),
                          tokenizer.get("<SME>"),
                          tokenizer.get("s_0"), tokenizer.get("CR_C"), tokenizer.get("CQ_m"), tokenizer.get("CB_/A"),
                          tokenizer.get("s_48"), tokenizer.get("CR_Ab"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                          tokenizer.get("<SME>"),
                          tokenizer.get("s_0"), tokenizer.get("CR_G"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),
                          tokenizer.get("s_48"), tokenizer.get("CR_Gb"), tokenizer.get("CQ_aug"), tokenizer.get("CB_None"),
                          tokenizer.get("<SME>"),
                          tokenizer.get("s_0"), tokenizer.get("CR_F"), tokenizer.get("CQ_m"), tokenizer.get("CB_None"),
                          tokenizer.get("s_48"), tokenizer.get("CR_G"), tokenizer.get("CQ_m7"), tokenizer.get("CB_None"),


                          ])
        """
        chord = None
        task_trained_generate(model, tokenizer, "out",
                              input_prompt_midi=midi_path, chord_prompt=chord,
                              program=program, task=MELODY_GEM, split_measure=3, temperature=1.2)
