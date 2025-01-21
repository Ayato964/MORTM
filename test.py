from pretty_midi import PrettyMIDI, Instrument, Note

def convert(inst: Instrument, v: int):
    new_inst = Instrument(inst.program)

    for note in inst.notes:
        note: Note
        new_inst.notes.append(Note(pitch=note.pitch + v, velocity=note.velocity, start=note.start, end=note.end))

    return new_inst


midi = PrettyMIDI("./data/generate/Sample.mid")
inst: list = midi.instruments
new_midi = PrettyMIDI()

for i in inst:
    i: Instrument
    if not i.is_drum:
        new_inst = convert(i, 2)
        new_midi.instruments.append(new_inst)
    else:
        new_midi.instruments.append(i)

new_midi.write("./data/generate/Sample_add2.mid")


