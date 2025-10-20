from mortm.utils.pianoroll_convert import get_pianoroll, pianoroll_to_midi


pianoroll = get_pianoroll("C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI/9/d/6/9d6db1fc171f8270c2233d8f848abd82.mid", ticks_per_measure=96, instrument_programs=[1, 65])
print(pianoroll.shape)
pianoroll_to_midi(pianoroll, "./out/test_pianoroll.mid", ticks_per_measure=96, instrument_programs=[1, 65], tempo=120)