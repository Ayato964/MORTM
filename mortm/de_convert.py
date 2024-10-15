from torch import Tensor
from pretty_midi import Instrument, Note, PrettyMIDI
from typing import List

from .tokenizer import Tokenizer
from .aya_node import Token


def ct_tokens_to_midi(tokenizer: Tokenizer, seq: Tensor, save_directory:str):
    seq_hot = seq[1:]
    split_tokens = seq_hot.split(split_size=3)
    midi = PrettyMIDI()
    inst = Instrument(program=66)
    back: Note = None

    for tokens in split_tokens:
        note = get_note(tokens, back, tokenizer, tokenizer.token_list)
        back = note
        if note is not None:
            inst.notes.append(note)
        else:
            break

    midi.instruments.append(inst)
    midi.write(save_directory)
    return midi



def get_note(tokens: Tensor, back_note: Note, tokenizer: Tokenizer, token_converter: List[Token]) -> Note:
    if 3 == len(tokens):
        pitch = token_converter[1](token=tokenizer.rev_get(tokens[1].item()))
        duration = token_converter[2](token=tokenizer.rev_get(tokens[2].item()))
        shift = token_converter[0](token=tokenizer.rev_get(tokens[0].item()))
        if back_note is not None:
            start = back_note.start + shift
        else:
            start = shift
        end = start + duration

        note = Note(pitch=pitch, velocity=100, start=start, end=end)
        print(note)
        return note
    else:
        return None
