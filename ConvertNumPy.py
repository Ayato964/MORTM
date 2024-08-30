from convert.ConvertMidi import ConvertMidi
from transformer.tokenizer import Tokenizer
import os
from messager import Messenger
def find_midi_files(root_folder):
    midi_files = []

    # Walk through the directory
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            # Check if the file is a MIDI file
            if file.lower().endswith(('.mid', '.midi')):
                # Get the full path and add it to the list
                full_path = os.path.join(defpath, file)
                midi_files.append(full_path)

    return midi_files



BRASS = [57, 58, 65, 66, 67, 68]
PIANO = [1, 2, 3, 4, 5, 6, 7, 8]
GUITAR = [25, 26, 27, 28, 29, 30, 31, 32]

ALL = [1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, 30, 31, 32, 57, 58, 65, 66, 67, 68]

tokenizer = Tokenizer()


datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets"
md_file = find_midi_files(datasets)


mes = Messenger()
count = 0
for file in md_file:
    con = ConvertMidi(tokenizer, file, BRASS, 120)
    con.convert()
    is_saved = con.save()
    if is_saved:
        count += 1
    print(count)
    if count - 1 >= 15000:
        break

tokenizer.save()
print(len(tokenizer.tokens))
print(tokenizer.token_max)

mes.send_mail("データセットの前処理が完了しました。", f"ボキャブラリーサイズは{len(tokenizer.tokens)}です")
