from torch import Tensor
from pretty_midi import Instrument, Note, PrettyMIDI
from typing import List

from .tokenizer import Tokenizer, PITCH_TYPE, START_TYPE, SHIFT_TYPE, VELOCITY_TYPE, DURATION_TYPE
from .aya_node import Token


def ct_token_to_midi(tokenizer: Tokenizer, seq: Tensor, save_directory:str, program=1):
    seq = seq[1:]
    midi = PrettyMIDI()
    inst: Instrument = Instrument(program=program)
    note = Note(pitch=0, velocity=100, start=0, end=0)
    back_note = None
    token_converter_list = tokenizer.music_token_list
    for token_id in seq:
        token = tokenizer.rev_get(token_id.item())
        if token_id == 2:
            break
        for con in token_converter_list:
            token_type = con(token=token, back_notes=back_note, note=note)
            if token_type == DURATION_TYPE:
                inst.notes.append(note)
                back_note = note
                note = Note(pitch=0, velocity=100, start=0, end=0)
    midi.instruments.append(inst)

    midi.write(save_directory)
    return midi
