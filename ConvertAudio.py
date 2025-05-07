import os

import numpy as np
from multiprocessing import Process, Manager

from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import Audio2MelSpectrogramALL
from midi2audio import FluidSynth

def find_midi_files(root_folder):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.wav')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


def convert_spect(pid, directory, md_file, output):
    local_count = 0
    for i in range(len(md_file)):
        con = Audio2MelSpectrogramALL(directory[i], md_file[i], n_mels=128, split_time=10)
        con.convert()
        is_saved, reason = con.save(output)
        #if is_saved:
        #    local_count += 1
        print(f"Process#{pid}: Running... {local_count}  {reason}")
    #progress[pid] = local_count


if __name__ == "__main__":
    THREAD_VALUE = 5
    PIANO = [i + 1 for i in range(5)]
    SAX = [65, 66]

    datasets = "./out/audio/datasets_small/"
    directory, md_file = find_midi_files(datasets)

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

    with Manager() as manager:
        progress = manager.dict()  # 共有辞書
        processes = []
        for t in range(THREAD_VALUE):
            p = Process(target=convert_spect, args=(t, directory[t].tolist(), md_file[t].tolist(), "./out/spect/datasets_small"))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_count = sum(progress.values())
        print(f"Total converted files: {total_count}")


    tokenizer.save("out/vocab/")
    print(len(tokenizer.tokens))
    print(tokenizer.token_max)
