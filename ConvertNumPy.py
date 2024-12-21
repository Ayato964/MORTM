import os
import numpy as np
from multiprocessing import Process, Manager
from mortm.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import MidiToSequece

def find_midi_files(root_folder):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files

def convert(pid, tokenizer, directory, md_file, program, progress):
    local_count = 0
    for i in range(len(md_file)):
        con = MidiToSequece(tokenizer, directory[i], md_file[i], program)
        con.convert()
        is_saved, reason = con.save("out/np/datasets/")
        if is_saved:
            local_count += 1
        print(f"Process#{pid}: Running... {local_count}  {reason}")
    progress[pid] = local_count

if __name__ == "__main__":
    THREAD_VALUE = 10
    SAX = [65, 66]

    datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"
    directory, md_file = find_midi_files(datasets)

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

    with Manager() as manager:
        progress = manager.dict()  # 共有辞書
        processes = []
        for t in range(THREAD_VALUE):
            p = Process(target=convert, args=(t, tokenizer, directory[t].tolist(), md_file[t].tolist(), SAX, progress))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_count = sum(progress.values())
        print(f"Total converted files: {total_count}")

    tokenizer.save("out/vocab/")
    print(len(tokenizer.tokens))
    print(tokenizer.token_max)
