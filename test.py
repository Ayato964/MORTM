from convert.ConvertMidi import ConvertMidi
import pretty_midi as pm


cm = ConvertMidi("data/other/4brosrvg.mid", [57, 58, 65, 66, 67, 68], 120)
cm.save()
aya_node = cm.convert()

print(aya_node[17])
