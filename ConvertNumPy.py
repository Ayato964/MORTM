"""
要確認
"""
#from mortm import gmail_messanger
from mortm.tokenizer import Tokenizer, get_token_converter, TO_MUSIC, TO_TOKEN
from mortm.convert import MidiToAyaNode
import os
#from mortm.messager import Messenger


def find_midi_files(root_folder):
    midi_files = []
    direc = []
    # Walk through the directory
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            # Check if the file is a MIDI file
            if file.lower().endswith(('.mid', '.midi')):
                # Get the full path and add it to the list
                midi_files.append(file)
                direc.append(defpath)

    return direc, midi_files



BRASS = [65, 66]
PIANO = [0, 1, 2, 3, 4, 5, 6, 7, 8]
GUITAR = [25, 26, 27, 28, 29, 30, 31, 32]

ALL = [1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, 30, 31, 32, 57, 58, 65, 66, 67, 68]



datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"
#datasets = "G:\情報科学科\研究室\datasets\MMD_MIDI"
#datasets = "data/other"
directory, md_file = find_midi_files(datasets)


#mes: Messenger = gmail_messanger.GmailMessanger()

tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

count = 0
reasons = dict()
for i in range(len(md_file)):
    con = MidiToAyaNode(tokenizer, directory[i], md_file[i], BRASS)
    con.convert()
    is_saved, reason = con.save("out/np/datasets")
    if is_saved:
        count += 1

    print(f"\r Save Count:{count} Step;{i}/{len(md_file)} Result:{reason}  Loaded:[{md_file[i]}] ", end="")

    #if count - 1 >= 20000:
    #    break

tokenizer.save("out/vocab/")
print(len(tokenizer.tokens))
print(tokenizer.token_max)

#mes.send_message("データセットの前処理が完了しました。", f"ボキャブラリーサイズは{len(tokenizer.tokens)}です")
