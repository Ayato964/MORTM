import os
import numpy as np
from multiprocessing import Process, Manager
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import MIDI2Seq, PackSeq, MidiExpantion

def find_midi_files(root_folder):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files




def convert(pid, tokenizer, directory, md_file, program, progress, save_path):
    local_count = 0
    for i in range(len(md_file)):
        con = MIDI2Seq(tokenizer, directory[i], md_file[i], program)
        con.convert()
        is_saved, reason = con.save(save_path)
        #if is_saved:
        #    local_count += 1
        print(f"Process#{pid}: Running... {local_count}  {reason}")
    progress[pid] = local_count


def expansion(pid, tokenizer, directory, md_file, program, progress, save_path):
    local_count = 0
    for i in range(len(md_file)):
        con = MidiExpantion(tokenizer, directory[i], md_file[i], program)
        ex_midi = con.expansion_midi()
        con.convert()
        is_saved, reason = con.save(save_path)

        for ex in ex_midi:
            ex.convert()
            is_saved, reason = ex.save(save_path)
            print(f"Process#{pid}: データ拡張中...{is_saved}  {reason}")

        if is_saved:
            local_count += 1
        #if local_count >= 30:
        #    break
        print(f"Process#{pid}: Running... {local_count}  {reason}")
    progress[pid] = local_count


def convert_ex(pid, tokenizer, directory, md_file, program, progress, save_path):
    local_count = 0
    for i in range(len(md_file)):
        con = MIDI2Seq(tokenizer, directory[i], md_file[i], program)
        ex_midi = con.expansion_midi()
        con.convert()
        is_saved, reason = con.save(save_path)

        for ex in ex_midi:
            ex.convert()
            is_saved, reason = ex.save(save_path)
            print(f"Process#{pid}: データ拡張中...{is_saved}  {reason}")

        if is_saved:
            local_count += 1
        #if local_count >= 30:
        #    break
        print(f"Process#{pid}: Running... {local_count}  {reason}")
    progress[pid] = local_count

if __name__ == "__main__":
    THREAD_VALUE = 10
    PIANO = [i + 1 for i in range(5)]
    SAX = [65, 66]

    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"
    datasets = "./data/other"
    directory, md_file = find_midi_files(datasets)

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

    with Manager() as manager:
        progress = manager.dict()  # 共有辞書
        processes = []
        for t in range(THREAD_VALUE):
            p = Process(target=expansion, args=(t, tokenizer, directory[t].tolist(),
                                              md_file[t].tolist(), PIANO, progress, "out/midi/datasets_small"))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_count = sum(progress.values())
        print(f"Total converted files: {total_count}")

    tokenizer.save("out/vocab/")
    print(len(tokenizer.tokens))
    print(tokenizer.token_max)
