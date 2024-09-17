'''
要確認
'''
#from mortm import gmail_messanger
from mortm.ConvertMidi import ConvertMidi
from mortm.tokenizer import Tokenizer
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



BRASS = [57, 58, 65, 66, 67, 68]
PIANO = [1, 2, 3, 4, 5, 6, 7, 8]
GUITAR = [25, 26, 27, 28, 29, 30, 31, 32]

ALL = [1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, 30, 31, 32, 57, 58, 65, 66, 67, 68]

tokenizer = Tokenizer("out/vocab")


datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"
directory, md_file = find_midi_files(datasets)


#mes: Messenger = gmail_messanger.GmailMessanger()

count = 0
for i in range(len(md_file)):
    con = MidiToAyaNode(tokenizer, directory[i], md_file[i], BRASS)
    con.convert()
    is_saved = con.save("out/np/datasets")
    if is_saved:
        count += 1
    print(count)
    if count - 1 >= 12000:
        break

tokenizer.save()
print(len(tokenizer.tokens))
print(tokenizer.token_max)

#mes.send_message("データセットの前処理が完了しました。", f"ボキャブラリーサイズは{len(tokenizer.tokens)}です")
