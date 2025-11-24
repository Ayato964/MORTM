from mortm.models.modules.config import MORTMArgs
from mortm.models.mortm import MORTM
from mortm.utils.generate import *
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.tokenizer import *

midi_path = "data/generate/Piano_Sample.mid"
args_path = "configs/models/mortm/A.json"
model_save_path = "out/model/mortm/45_research/MORTM.4.5-PRO_1.0899.pth"
#model_save_path = "out/model/mortm/MORTM.4.1-SAX-Phase2_0.29.pth"
sft_model = False
generate_count = 1
program = ["PIANO", "SAX"]
out_program = [65 for _ in range(generate_count)]
midi_path = [midi_path for _ in range(generate_count)]

if __name__ == "__main__":
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    args = MORTMArgs(args_path)
    if sft_model:
        args.use_lora = True
    p = _DefaultLearningProgress()
    model = MORTM(progress=p, args=args)
    model.load_state_dict(torch.load(model_save_path))  # Load the model
    model.to(p.get_device())

    if not sft_model:
        pre_train_generate(model, tokenizer, "out", midi_path, end_tokens=(tokenizer.get("<TE>")), split_measure=3, program=program, key="Fm",  temperature=1.2)
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
                              program=program, task=MELODY_GEM, split_measure=6, temperature=1.0)
