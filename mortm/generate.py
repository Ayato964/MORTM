import torch
from torch import Tensor

from .mortm import MORTM


def generate_note(note_max: int, input_seq, model: MORTM, t=1.0, p=0.90):
    model.eval()
    if not isinstance(input_seq, torch.Tensor):
        input_seq = torch.tensor(input_seq, dtype=torch.long, device=model.progress.get_device())

    generated = input_seq.tolist()
    for c in range(note_max):
        for i in range(3):
            logits = model(input_seq.unsqueeze(0))
            logits = logits[-1, -1, :]
            if c <= note_max - (note_max / 4):
                logits[2] = -6
            if 4 <= generated[-1] <= 196:
                token = model.top_p_sampling(logits, p=p, temperature=t)
            else:
                token = model.top_p_sampling(logits, p=0.95, temperature=0.9)
            generated.append(token)
            if token == 2:
                break

            input_seq = torch.tensor(generated, dtype=torch.long, device=model.progress.get_device())
        print(f"\r Note Max::{(c / note_max) * 100 :.4f}", end="")
    return torch.tensor(generated, dtype=torch.long, device=model.progress.get_device())

def generate_measure(measure: int, input_seq, model: MORTM):
    model.eval()
    if not isinstance(input_seq, torch.Tensor):
        input_seq = torch.tensor(input_seq, dtype=torch.long, device=model.progress.get_device())

    generated = input_seq.tolist()
    for _ in range(measure):
        pass