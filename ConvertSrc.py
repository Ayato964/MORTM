import json
from pathlib import Path
from typing import List

from multiprocessing import Process, Manager
from mortm.train.tokenizer import *
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

def extract_midi_npz_paths(json_file_path: str) -> Tuple[List[str], List[str]]:
    """
    Extracts paths for .npz files located within a 'midi' directory from a JSON file.

    Args:
        json_file_path (str): Path to the target JSON file.

    Returns:
        List[str]: A flattened list of filtered file paths.
    """
    base_paths = []
    file_paths = []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sublist in data:
            for path_str in sublist:
                path = Path(path_str)

                if path.suffix == '.npz' and 'midi' in path.parts:
                    base_paths.append(os.path.dirname(path_str))
                    file_paths.append(os.path.basename(path_str))

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON.")
        return [], []

    return base_paths, file_paths

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
            con = MIDIConverter(tokenizer, directory[i], md_file[i], program)
            con.convert()

            if con.is_error:
                continue

            maker = PreTrainDataMaker(con, 12)
            maker.convert()

            is_saved, reason = maker.save(save_path)
            if is_saved:
                local_count += 1

            for a in maker.aya_node[1:]:
                a:np.ndarray
                if np.sum(a == 0) != 0:
                    print(f"\033[31m Error!!  {directory[i]}/{md_file[i]}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    is_error = True
                    break
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


def convert_task_seq(pid, tokenizer, directory, md_file, system_file, program, progress, save_directory):
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
    datamaker = [Task1DataMaker,Task2DataMaker, Task3DataMaker, Task4DataMaker, Task5DataMaker,
                 Task6DataMaker, Task7DataMaker, Task8DataMaker, Task9DataMaker, Task10DataMaker]
    for i in range(len(md_file)):
        if (system_file[i]["key"] and len(system_file[i]["all_chords"]) > 10 and not ("N" in system_file[i]["all_chords"])
                and len(system_file[i]["all_chords_timestamps"]) != 0 and system_file[i]["tempo"]):
            #print(system_file[i]["location"], system_file[i]["all_chords"], system_file[i]["all_chords_timestamps"])
            is_error = False
            con = MIDIConverter(tokenizer, directory[i], md_file[i], program, key=system_file[i]["key"],
                                all_chords=system_file[i]["all_chords"], all_chord_timestamps=system_file[i]["all_chords_timestamps"],
                                use_midi2seq=True, use_midi2seq_with_chord=True)
            con.convert()

            if con.is_error:
                continue
            print("コード----------------")
            print(con.midi2seq_with_chord.aya_node)

            for c, maker, in enumerate(datamaker):
                m = maker(converter=con, measure_max=8)
                m.convert()


                for a in m.aya_node[1:]:
                    a:np.ndarray
                    if np.sum(a == 0) != 0:
                        print(a)
                        print(f"\033[31m 警告, 今すぐ処理を中断してください！！！  : {system_file[i]["location"]}  {np.where(a == 0)} ")
                        is_error = True
                        break

                if not is_error:
                    is_saved, reason = m.save(os.path.join(save_directory, f"task{c+1}/"))
                    if is_saved:
                        local_count += 1
                    print(f"Process#{pid}: Running... {local_count}  Reason: {reason}")


if __name__ == "__main__":
    print("やあっほう！変換開始だよ！！")
    THREAD_VALUE = 10
    PROGRAM = ['PIANO', 'SAX']
    tokenizer = Tokenizer(omega_converter(TO_TOKEN))

    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/GMD/training/all-instruments-with-drums/"
    datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/MIDI_Caps"
    #datasets = "C:/Users/Nagoshi Takaaki.KTHRLab/MIDIdatasets/midi_hawthorne/midi/live"
    #datasets = "./data/other"
    #datasets = "./out/model/mortm/45_research/datasets/eval.json"

    """
    directory, md_file = find_midi_files(datasets)
    #directory, md_file = extract_midi_npz_paths(datasets)
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

    with Manager() as manager:
        progress = manager.dict()  # 共有辞書
        processes = []
        for t in range(THREAD_VALUE):
            """
            p = Process(target=convert, args=(t, tokenizer, directory[t].tolist(),
                                              md_file[t].tolist(), PROGRAM, progress, "C:/Users/Nagoshi Takaaki.KTHRLab/MORTM/pre_train/music"))
            """

            """
            p = Process(target=convert_class_seq, args=(t, tokenizer, directory[t], md_file[t], "HUMAN", "out/np/bertm/"))
            """

            #"""
            p = Process(target=convert_task_seq, args=(t, tokenizer, directory[t].tolist(),
                                                         md_file[t].tolist(), system_file[t], PROGRAM, progress, "C:/Users/Nagoshi Takaaki.KTHRLab/MORTM/post_train/omega/"))
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

