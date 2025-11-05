from pretty_midi import PrettyMIDI
import pretty_midi.instrument as inst
from torch import Tensor

from mortm.train.custom_token import ShiftTimeContainer, ChordToken
from mortm.train.tokenizer import *


def ct_token_to_midi(tokenizer: Tokenizer, seq: Tensor, save_directory:str, tempo=120):
    seq = seq[1:]
    print("これから処理するトークン列:", seq)
    midi = PrettyMIDI()
    back_note = None
    token_converter_list = tokenizer.music_token_list
    container = ShiftTimeContainer(0, tempo)
    init_inst = None
    note = Note(pitch=0, velocity=100, start=0, end=0)

    for token_id in seq:
        token = tokenizer.rev_get(token_id.item())
        if token_id == tokenizer.get("<TE>"):
            break

        for con in token_converter_list:
            if not isinstance(con, ChordToken):
                token_type = con(token=token, note=note, back_notes=back_note, container=container, tempo=tempo)

                if container.get_inst() is not None:
                    if init_inst is not None:
                        midi.instruments.append(init_inst)
                    init_inst = inst.Instrument(program=container.get_inst(), is_drum=False)
                    container = ShiftTimeContainer(0, tempo)

                if token_type == DURATION_TYPE:
                    back_note = note
                    init_inst.notes.append(note)
                    note = Note(pitch=0, velocity=100, start=0, end=0)
    midi.instruments.append(init_inst)
    midi.write(save_directory)

