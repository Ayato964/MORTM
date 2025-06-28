import json
import os
from typing import List

import numpy as np
from multiprocessing import Process, Manager
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import *

def find_midi_files(root_folder):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


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



def convert(pid, tokenizer, directory, md_file, program, progress, save_path):
    '''
    MIDIデータを受け取り、シーケンス生成を行う。
    :param pid:
    :param tokenizer:
    :param directory:
    :param md_file:
    :param program:
    :param progress:
    :param save_path:
    :return:
    '''

    local_count = 0
    is_error = False
    for i in range(len(md_file)):
        if not is_error:
            con = MIDI2Seq(tokenizer, directory[i], md_file[i], program)
            con.convert()
            is_saved, reason = con.save(save_path)
            if is_saved:
                local_count += 1
                #if local_count >= 50:
                #    break

            for a in con.aya_node[1:]:
                a:np.ndarray
                if np.sum(a == 0) != 0:
                    print(f"Error!!  {directory[i]}/{md_file[i]}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    is_error = True
                    break
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
    '''
    MIDIデータを受け取り、全てのキーに変換を行いながらシーケンス生成を行う。
    :param pid:
    :param tokenizer:
    :param directory:
    :param md_file:
    :param program:
    :param progress:
    :param save_path:
    :return:
    '''
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


def convert_with_chord(pid, tokenizer, directory: List[str], md_file: List[str], system_file: List[dict], program, progress, save_path):
    '''
    MIDIデータと楽曲情報を含むJSONを受け取り、コード進行を考慮したシーケンス生成を行う。
    :param pid:
    :param tokenizer:
    :param directory: 楽曲のディレクトリが格納されている。
    :param md_file:　楽曲名が格納されている。
    :param system_file: 楽曲情報が格納されている
    :param program:
    :param progress:
    :param save_path:
    :return:
    '''
    local_count = 0

    for i in range(len(md_file)):
        if (system_file[i]["key"] and system_file[i]["all_chords"] and not ("N" in system_file[i]["all_chords"])
                and system_file[i]["all_chords_timestamps"] and system_file[i]["tempo"]):

            con = Midi2SeqWithChord(tokenizer, directory[i], md_file[i],key=system_file[i]["key"], all_chords=system_file[i]["all_chords"],
                                    all_chord_timestamps=system_file[i]["all_chords_timestamps"],  program_list=program)
            con.convert()
            print("SSS?")
            is_saved, reason = con.save(save_path)
            for a in con.aya_node[1:]:
                a:np.ndarray
                if np.sum(a == 0) != 0:
                    print(f"Error!!  : {system_file[i]["location"]}")
                    is_error = True
                    break
            if is_saved:
                local_count += 1
                #if local_count >= 10:
                #    break

            print(f"Process#{pid}: Running... {local_count}  {reason}")

    progress[pid] = local_count

def convert_chord(pid, tokenizer, directory, md_file, system_file, SAX, progress, save_directory):
    local_count = 0

    for i in range(len(md_file)):
        if (system_file[i]["key"] and system_file[i]["all_chords"] and not ("N" in system_file[i]["all_chords"])
                and system_file[i]["all_chords_timestamps"] and system_file[i]["tempo"]):
            is_error = False
            con = MetaData2Chord(tokenizer, key=system_file[i]["key"], all_chords=system_file[i]["all_chords"],
                                 all_chord_timestamps=system_file[i]["all_chords_timestamps"], tempo=system_file[i]["tempo"],
                                 directory=directory[i], file_name=md_file[i])
            con.convert()
            print("は？")

            for a in con.aya_node[1:]:
                a:np.ndarray
                if np.sum(a == 0) != 0:
                    print(f"\033[31m 警告, 今すぐ処理を中断してください！！！  : {system_file[i]["location"]}  {np.where(a == 0)}")
                    is_error = True
                    break
            if not is_error:
                is_saved, reason = con.save(save_directory)
                if is_saved:
                    local_count += 1

                print(f"Process#{pid}: Running... {local_count}  Reason: {reason}")

def convert_task_seq(pid, tokenizer, directory, md_file, system_file, SAX, progress, save_directory):
    local_count = 0
    for i in range(len(md_file)):
        if (system_file[i]["key"] and len(system_file[i]["all_chords"]) > 10 and not ("N" in system_file[i]["all_chords"])
                and len(system_file[i]["all_chords_timestamps"]) != 0 and system_file[i]["tempo"]):
            #print(system_file[i]["location"], system_file[i]["all_chords"], system_file[i]["all_chords_timestamps"])
            is_error = False
            con = MIDI2TaskSeq(tokenizer, system=system_file[i], split_measure=8, out_measure=12,
                                 directory=directory[i], file_name=md_file[i], program_list=SAX)
            con.convert()

            for a in con.aya_node[1:]:
                a:np.ndarray
                if np.sum(a == 0) != 0:
                    print(f"\033[31m 警告, 今すぐ処理を中断してください！！！  : {system_file[i]["location"]}  {np.where(a == 0)} ")
                    is_error = True
                    break

            if not is_error:
                is_saved, reason = con.save(save_directory)
                if is_saved:
                    local_count += 1
                print(f"Process#{pid}: Running... {local_count}  Reason: {reason}")


if __name__ == "__main__":
    THREAD_VALUE = 10
    PIANO = [i + 1 for i in range(5)]
    SAX = [65, 66]

    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"
    datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MIDI_Caps"
    #datasets = "./data/other"
    """
    directory, md_file = find_midi_files(datasets)
    """
    #"""
    print("データ整理中・・・・")
    directory, md_file, system_file = find_midi_files_with_json(datasets)
    print("完了！！")
    #"""

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)
    #"""
    system_file = np.array_split(system_file, THREAD_VALUE)
    #"""
    tokenizer = Tokenizer(get_token_converter(TO_TOKEN))

    with Manager() as manager:
        progress = manager.dict()  # 共有辞書
        processes = []
        for t in range(THREAD_VALUE):
            """
            p = Process(target=convert, args=(t, tokenizer, directory[t].tolist(),
                                              md_file[t].tolist(), SAX, progress, "out/np/Sax/pre-train/Phase1/mel_large"))
            """

            #"""
            p = Process(target=convert_task_seq, args=(t, tokenizer, directory[t].tolist(),
                                                         md_file[t].tolist(), system_file[t], SAX, progress, "out/np/Sax/task_train"))
            #"""

            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_count = sum(progress.values())
        print(f"Total converted files: {total_count}")

    tokenizer.save("out/vocab/")
    print(len(tokenizer.tokens))
    print(tokenizer.token_max)
