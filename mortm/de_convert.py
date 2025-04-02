from torch import Tensor
from pretty_midi import Instrument, Note, PrettyMIDI
from typing import List

from .tokenizer import Tokenizer, PITCH_TYPE, START_TYPE, SHIFT_TYPE, VELOCITY_TYPE, DURATION_TYPE
from .custom_token import Token, ShiftTimeContainer




def ct_token_to_midi(tokenizer: Tokenizer, seq: Tensor, save_directory:str, program=1, tempo=120):
    seq = seq[1:]
    midi = PrettyMIDI()
    inst: Instrument = Instrument(program=program)
    note = Note(pitch=0, velocity=100, start=0, end=0)
    back_note = None
    token_converter_list = tokenizer.music_token_list
    container = ShiftTimeContainer(0, tempo)
    for token_id in seq:
        token = tokenizer.rev_get(token_id.item())
        if token_id == 392:
            break
        if token_id == 3:
            container.shift()

        for con in token_converter_list:
            token_type = con(token=token, back_notes=back_note, note=note, container=container, tempo=tempo)
            if token_type == DURATION_TYPE:
                inst.notes.append(note)
                back_note = note
                note = Note(pitch=0, velocity=100, start=0, end=0)
    midi.instruments.append(inst)
    print(inst.notes)
    midi.write(save_directory)
    return midi
