import torch
from torch import Tensor

from .mortm import MORTM


def generate_note(note_max: int, input_seq, model: MORTM, t=1.0):
    model.eval()
    if not isinstance(input_seq, torch.Tensor):
        input_seq = torch.tensor(input_seq, dtype=torch.long, device=model.progress.get_device())

    generated = input_seq.tolist()
    for _ in range(note_max):
        for i in range(3):
            logits = model(input_seq)
            if i == 0:
                token = model.top_p_sampling(logits, p=0.9, temperature=0.8)
            elif i == 1:
                token = model.top_p_sampling(logits, p=0.85, temperature=1.2)
            else:
                token = model.top_p_sampling(logits, p=0.9, temperature=0.8)
            generated.append(token)
    return torch.tensor(generated, dtype=torch.long, device=model.progress.get_device())

def generate_measure(measure: int, input_seq, model: MORTM):
    model.eval()
    if not isinstance(input_seq, torch.Tensor):
        input_seq = torch.tensor(input_seq, dtype=torch.long, device=model.progress.get_device())

    generated = input_seq.tolist()
    for _ in range(measure):
        pass