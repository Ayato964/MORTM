from torch import Tensor
from pretty_midi import Instrument, Note, PrettyMIDI
from typing import List

from .tokenizer import Tokenizer, PITCH_TYPE, START_TYPE, SHIFT_TYPE, VELOCITY_TYPE, DURATION_TYPE
from .aya_node import Token


def ct_tokens_to_midi_b5(tokenizer: Tokenizer, seq: Tensor, save_directory:str):
    seq_hot = seq[1:]
    split_tokens = seq_hot.split(split_size=3)
    midi = PrettyMIDI()
    inst = Instrument(program=1)
    back: Note = None

    for tokens in split_tokens:
        note = get_note_b5(tokens, back, tokenizer, tokenizer.token_list)
        back = note
        if note is not None:
            inst.notes.append(note)
        else:
            break

    midi.instruments.append(inst)
    midi.write(save_directory)
    return midi



def get_note_b5(tokens: Tensor, back_note: Note, tokenizer: Tokenizer, token_converter: List[Token]) -> Note:
    if 3 == len(tokens) and not (2 in tokens):
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


def get_type_number(token_list, t):
    for i in range(len(token_list)):
        t_type, number = token_list[i](token=t)
        if t_type is not None:
            return t_type, number
    return None, None



def ct_token_to_midi_1_0(tokenizer: Tokenizer, seq: Tensor, save_directory:str, program=1):
    midi = PrettyMIDI()
    inst: Instrument = Instrument(program=program)
    token_list = tokenizer.token_list
    note_list = {'START':0, 'PITCH':0, 'DURATION':0, 'VELOCITY':100}
    shift_time = 0
    for seq_t in seq:
        t = tokenizer.rev_get(seq_t.item())
        t_type, number = get_type_number(token_list, t)
        if t_type is not None:
            if t_type is SHIFT_TYPE:
                shift_time += number
            elif t_type is START_TYPE:
                note_list['START'] = number
            elif t_type is PITCH_TYPE:
                note_list['PITCH'] = number
            elif t_type is DURATION_TYPE:
                note_list['DURATION'] = number
                inst.notes.append(get_note_1_0(note_list, shift_time))
    midi.instruments.append(inst)
    midi.write(save_directory)
    return midi

def get_note_1_0(note_list, shift_time)-> Note:
    return Note(pitch=note_list['PITCH'], start=note_list['START'] + shift_time,
                end=note_list['START'] + note_list['DURATION'] + shift_time, velocity=note_list['VELOCITY'])
