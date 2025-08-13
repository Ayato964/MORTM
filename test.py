import json

from mortm.utils.convert import MetaData2Chord
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN

EX = 821
def find_midi_files_with_json(root_folder):
    midi_files = []
    direct = []
    with open(f'{root_folder}/json/train.json', 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
        for i in range(len(data)):
            location = data[i]["location"]
            file_name = location.split("/")[-1]
            d = "/".join(location.split("/")[:-1])

            midi_files.append(file_name)
            direct.append(f"{root_folder}/{d}")

    return direct, midi_files, data

datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MIDI_Caps"
directory, md_file, system_file = find_midi_files_with_json(datasets)
tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

print(system_file[EX]["location"])
conv = MetaData2Chord(tokenizer, system_file[EX]["key"], system_file[EX]["all_chords"], system_file[EX]["all_chords_timestamps"],
                      system_file[EX]["tempo"], directory[EX], file_name=md_file[EX])
conv.convert()

print(conv.aya_node)
tokenizer.save("out/vocab/")