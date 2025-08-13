import os.path
from typing import List

from pretty_midi import PrettyMIDI

from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer
from mortm.utils.convert import MIDI2Seq

def pre_train_generate(model: MORTM, tokenizer: Tokenizer,
                       midi_path: str | List[str], program: List[int], split_measure: int,
                       temperature: float = 1.0, p=0.95, ) -> PrettyMIDI | List[PrettyMIDI]:
    if isinstance(midi_path, str):
        midi_path = [midi_path]

    src_list = []
    for path in midi_path:
        if not os.path.exists(path):
            assert FileNotFoundError(f"MIDI file not found: {path}")
        converter = MIDI2Seq(tokenizer, os.path.dirname(path), os.path.basename(path), split_measure=split_measure, program_list=program)
        converter.convert()

