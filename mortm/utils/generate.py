import os.path
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from pretty_midi import PrettyMIDI

from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, TO_MUSIC
from mortm.utils.convert import MIDI2Seq
from mortm.utils.de_convert import ct_token_to_midi


def pre_train_generate(model: MORTM, tokenizer: Tokenizer, save_directory: str,
                       midi_path: str | List[str], program: List[int], split_measure: int = 999,
                       temperature: float = 1.0, p=0.95, print_log = True) -> PrettyMIDI | List[PrettyMIDI]:
    if isinstance(midi_path, str):
        midi_path = [midi_path]

    src_list = []
    for path in midi_path:
        if not os.path.exists(path):
            assert FileNotFoundError(f"MIDI file not found: {path}")
        converter = MIDI2Seq(tokenizer, os.path.dirname(path), os.path.basename(path), split_measure=split_measure, program_list=program)
        converter.convert()
        if not converter.is_error:
            src_list.append(torch.tensor(converter.aya_node[1][:-1]))
        else:
            print(f"Error in converting MIDI file: {path}. Skipping this file.")
            continue

    src_list = pad_sequence(src_list, batch_first=True, padding_value=tokenizer.get("<PAD>"))
    src_list = src_list.to(model.progress.get_device())
    #print(src_list)
    all_seq, _ = model.top_sampling_measure_kv_cache(src_list, temperature=temperature, p=p, print_log=print_log)

    tokenizer.mode(to=TO_MUSIC)
    midi = []
    count = 0
    for seq in all_seq:
        m = ct_token_to_midi(tokenizer, seq, os.path.join(save_directory, f"generated_{os.path.basename(midi_path[count])}"))
        midi.append(m)
        count += 1

    return midi if len(midi) != 1 else midi[0]
