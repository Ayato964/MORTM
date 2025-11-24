import os.path
from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from pretty_midi import PrettyMIDI

from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, TO_MUSIC
from mortm.utils.convert import *
from mortm.utils.de_convert import ct_token_to_midi

########### タスク ############
MELODY_GEM = "melody_gem"
CHORD_GEM = "chord_gem"


def _create_midi_prompt(tokenizer: Tokenizer, midi_path: str | List[str], split_measure, program, is_add_query_symbol=False) -> List[torch.Tensor]:

    if isinstance(midi_path, str):
        midi_list = [midi_path]
    else:
        midi_list = midi_path

    src_list = []
    for path in midi_list:
        if not os.path.exists(path):
            assert FileNotFoundError(f"MIDI file not found: {path}")
        converter = MIDI2Seq(tokenizer, os.path.dirname(path), os.path.basename(path), split_measure=split_measure, program_list=program, is_include_special_token=False)
        converter.convert()
        if not converter.is_error:
            if is_add_query_symbol:
                src_list.append(torch.cat([torch.tensor([tokenizer.get("<QUERY_M>")]), torch.tensor(converter.aya_node[1][:-1]), torch.tensor([tokenizer.get("</QUERY_M>")])]))
            else:
                src_list.append(torch.tensor(converter.aya_node[1][:-1]))
        else:
            print(f"Error in converting MIDI file: {path}. Skipping this file.")
            continue

    return src_list

def _print_gen(seq, tokenizer):
    for s in seq:
        for token in s:
            print(token, tokenizer.rev_get(token))

def create_chord_prompt(tokenizer: Tokenizer, chord_prompt: List[np.ndarray]) -> List[torch.Tensor]:
    if isinstance(chord_prompt, np.ndarray):
        chord_prompt = [torch.cat([
            torch.tensor([tokenizer.get("<QUERY_C>")]), torch.tensor(chord_prompt), torch.tensor([tokenizer.get("</QUERY_C>")])])]
    elif chord_prompt is not None:
        chord_prompt = [torch.cat([torch.tensor(tokenizer.get("<QUERY_C>")), torch.tensor(cp), tokenizer.get("</QUERY_C>")]) for cp in chord_prompt]

    return chord_prompt

def create_system_prompt(tokenizer: Tokenizer, program: List[List[str]], key: List[str]) -> List[torch.Tensor]:
    prompt = []
    if isinstance(program[0], str):
        program = [program]
    if isinstance(key, str):
        key = [key]
    for i, k in enumerate(key):
        t = []
        t.append(tokenizer.get("<EOS>"))
        t.append(tokenizer.get("<SYSTEM>"))
        if program[i] is not None:
            for p in program[i]:
                t.append(tokenizer.get(f"<INST_{p}>"))
        if k is not None:
            t.append(tokenizer.get(f"k_{k}"))
        t.append(tokenizer.get("<TAG_END>"))
        t.append(tokenizer.get("<MGEN>"))
        prompt.append(torch.tensor(t))
    return prompt



def pre_train_generate(model: MORTM, tokenizer: Tokenizer, save_directory: str,
                       midi_path: str | List[str], program: List[str] | List[List[str]], key:str | List[str], end_tokens: tuple, split_measure: int = 999,
                       temperature: float = 1.0, p=0.95, print_log = True) -> PrettyMIDI | List[PrettyMIDI]:

    system_prompt = create_system_prompt(tokenizer, program, key)
    if midi_path is not None:
        src_list = _create_midi_prompt(tokenizer, midi_path, split_measure, program)
        for i in range(len(src_list)):
            src_list[i] = torch.cat([system_prompt[i], src_list[i]])
    else:
        src_list = system_prompt

    src_list = pad_sequence(src_list, batch_first=True, padding_value=tokenizer.get("<PAD>")).to(model.progress.get_device())
    all_seq, _ = model.top_sampling_measure_kv_cache(tokenizer, src_list, temperature=temperature, p=p, print_log=print_log)

    tokenizer.mode(to=TO_MUSIC)
    _print_gen(all_seq, tokenizer)
    midi = []
    for i, seq in enumerate(all_seq):
        if midi_path is not None:
            m = ct_token_to_midi(tokenizer, seq, os.path.join(save_directory, f"generated_{os.path.basename(midi_path[i])}_{i}.mid"))
        else:
            m = ct_token_to_midi(tokenizer, seq, os.path.join(save_directory, f"generated_system_prompt_{i}.mid"))
        midi.append(m)

    return midi if len(midi) != 1 else midi[0]


def task_trained_generate(model: MORTM, tokenizer: Tokenizer, save_directory: str,
                          input_prompt_midi: str | List[str], chord_prompt: np.ndarray | List[np.ndarray],
                          program: List[int], task: "MELODY_GEM" or "CHORD_GEM",
                          split_measure: int = 999, temperature: float = 1.0, p=0.95, print_log=True
                          ) -> PrettyMIDI | List[PrettyMIDI] | str | List[str]:
    if input_prompt_midi is not None:
        midi_prompt = _create_midi_prompt(tokenizer, input_prompt_midi, split_measure, program, is_add_query_symbol=True)
    else:
        midi_prompt = None

    if chord_prompt is not None:
        chord_prompt = create_chord_prompt(tokenizer, chord_prompt)

    prompt = []
    out_d = torch.tensor([tokenizer.get("<MGEN>")]) if task == MELODY_GEM else torch.tensor([tokenizer.get("<CGEN>")])
    if midi_prompt is not None:
        for i in range(len(midi_prompt)):
            if chord_prompt is not None:
                prompt.append(torch.cat([midi_prompt[i], chord_prompt[i], out_d]))
            else:
                prompt.append(torch.cat([midi_prompt[i], out_d]))
    else:
        for i in range(len(chord_prompt)):
            prompt.append(torch.cat([chord_prompt[i], out_d]))

    src_prompt = pad_sequence(prompt, batch_first=True, padding_value=tokenizer.get("<PAD>")).to(model.progress.get_device())
    all_seq, pack = model.top_sampling_measure_kv_cache(src_prompt, temperature=temperature, p=p, print_log=print_log) #生成
    prompt, generated = pack

    tokenizer.mode(to=TO_MUSIC)
    midi = []
    chord = []
    for i in range(len(all_seq)):
        if task == MELODY_GEM:
            print(f"PROMPT: {generated[i]}")
            m = torch.cat([midi_prompt[i],  torch.tensor(generated[i])])
            midi.append(ct_token_to_midi(tokenizer, m, os.path.join(save_directory, f"melody_generated_{i}.mid"), program=program[0]))
        else:
            c = torch.cat([chord_prompt[i], generated[i]])
            chord.append(c)

    return (midi if len(midi) != 1 else midi[0]) if task == MELODY_GEM else (chord if len(chord) != 1 else chord[0])
