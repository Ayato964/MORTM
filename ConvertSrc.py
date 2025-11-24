import json
from typing import List

from multiprocessing import Process, Manager
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, get_token_converter_melody_only
from mortm.utils.convert import *

def find_midi_files(root_folder):
    """
    Finds all MIDI files in the specified root folder.

    Args:
        root_folder (str): Path to the root folder.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of directories and a list of MIDI file names.
    """
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


def find_midi_files_with_json(root_folder):
    """
    Finds all MIDI files and their corresponding JSON data in the specified root folder.

    Args:
        root_folder (str): Path to the root folder.

    Returns:
        Tuple[List[str], List[str], List[dict]]: A tuple containing a list of directories, a list of MIDI file names, and a list of JSON data.
    """
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
    """
    Converts MIDI files to sequences using the specified tokenizer.

    Args:
        pid (int): Process ID.
        tokenizer (Tokenizer): Tokenizer instance for conversion.
        directory (List[str]): List of directories containing MIDI files.
        md_file (List[str]): List of MIDI file names.
        program (List[str]): List of MIDI program numbers.
        progress (Manager.dict): Shared dictionary to track progress.
        save_path (str): Directory to save the converted sequences.

    Returns:
        None
    """
    local_count = 0
    is_error = False
    for i in range(len(md_file)):
        if not is_error:
            con = MIDI2Seq(tokenizer, directory[i], md_file[i], program)
            con.convert()
            is_saved, reason = con.save(save_path)
            if is_saved:
                local_count += 1

            for a in con.aya_node[1:]:
                a:np.ndarray
                if np.sum(a == 0) != 0:
                    print(f"\033[31m Error!!  {directory[i]}/{md_file[i]}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    is_error = True
                    break
            print(f"Process#{pid}: Running... {local_count}  {reason}")
    progress[pid] = local_count


def expansion(pid, tokenizer, directory, md_file, program, progress, save_path):
    """
    Expands and converts MIDI files to sequences using the specified tokenizer.

    Args:
        pid (int): Process ID.
        tokenizer (Tokenizer): Tokenizer instance for conversion.
        directory (List[str]): List of directories containing MIDI files.
        md_file (List[str]): List of MIDI file names.
        program (List[int]): List of MIDI program numbers.
        progress (Manager.dict): Shared dictionary to track progress.
        save_path (str): Directory to save the converted sequences.

    Returns:
        None
    """
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
    """
    Converts MIDI files to sequences in all keys using the specified tokenizer.

    Args:
        pid (int): Process ID.
        tokenizer (Tokenizer): Tokenizer instance for conversion.
        directory (List[str]): List of directories containing MIDI files.
        md_file (List[str]): List of MIDI file names.
        program (List[int]): List of MIDI program numbers.
        progress (Manager.dict): Shared dictionary to track progress.
        save_path (str): Directory to save the converted sequences.

    Returns:
        None
    """
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
    """
    Converts MIDI files to sequences considering chord progressions using the specified tokenizer.

    Args:
        pid (int): Process ID.
        tokenizer (Tokenizer): Tokenizer instance for conversion.
        directory (List[str]): List of directories containing MIDI files.
        md_file (List[str]): List of MIDI file names.
        system_file (List[dict]): List of JSON data containing song information.
        program (List[int]): List of MIDI program numbers.
        progress (Manager.dict): Shared dictionary to track progress.
        save_path (str): Directory to save the converted sequences.

    Returns:
        None
    """
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
    """
    Converts MIDI files to chord sequences using the specified tokenizer.

    Args:
        pid (int): Process ID.
        tokenizer (Tokenizer): Tokenizer instance for conversion.
        directory (List[str]): List of directories containing MIDI files.
        md_file (List[str]): List of MIDI file names.
        system_file (List[dict]): List of JSON data containing song information.
        SAX (List[int]): List of MIDI program numbers for saxophone.
        progress (Manager.dict): Shared dictionary to track progress.
        save_directory (str): Directory to save the converted sequences.

    Returns:
        None
    """
    local_count = 0

    for i in range(len(md_file)):
        if (system_file[i]["key"] and system_file[i]["all_chords"] and not ("N" in system_file[i]["all_chords"])
                and system_file[i]["all_chords_timestamps"] and system_file[i]["tempo"]):
            is_error = False
            con = MetaData2Chord(tokenizer, key=system_file[i]["key"], all_chords=system_file[i]["all_chords"],
                                 all_chord_timestamps=system_file[i]["all_chords_timestamps"], tempo=system_file[i]["tempo"],
                                 directory=directory[i], file_name=md_file[i])
            con.convert()

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
    """
    Converts MIDI files to task sequences using the specified tokenizer.

    Args:
        pid (int): Process ID.
        tokenizer (Tokenizer): Tokenizer instance for conversion.
        directory (List[str]): List of directories containing MIDI files.
        md_file (List[str]): List of MIDI file names.
        system_file (List[dict]): List of JSON data containing song information.
        SAX (List[int]): List of MIDI program numbers for saxophone.
        progress (Manager.dict): Shared dictionary to track progress.
        save_directory (str): Directory to save the converted sequences.

    Returns:
        None
    """
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
                    print(a)
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
    PROGRAM = ['PIANO', 'SAX']

    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MMD_MIDI"
    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/LMD/lmd_full"
    datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MIDI_Caps"
    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/midi_hawthorne/midi/live"
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
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))

    with Manager() as manager:
        progress = manager.dict()  # 共有辞書
        processes = []
        for t in range(THREAD_VALUE):
            """
            p = Process(target=convert, args=(t, tokenizer, directory[t].tolist(),
                                              md_file[t].tolist(), PROGRAM, progress, "out/np/research/midi"))
            """

            #"""
            p = Process(target=convert_chord, args=(t, tokenizer, directory[t].tolist(),
                                                         md_file[t].tolist(), system_file[t], PROGRAM, progress, "out/np/research/chord"))
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

