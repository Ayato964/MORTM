import os
from multiprocessing import Process, Manager

import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker


def find_midi_files(root_folder: str):
    midi_files = []
    direc = []
    for defpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


def convert_foundation(pid, tokenizer, directory, md_file, program, min_measure, max_measure, progress, save_path):
    local_count = 0
    local_tokens = 0

    for i in range(len(md_file)):
        try:
            con = MIDIConverter(tokenizer, directory[i], md_file[i], program)
            con.convert()

            if con.is_error:
                continue

            # 原曲を変換
            maker = FoundationDataMaker(con, min_measure, max_measure)
            stats = maker.convert()

            has_zero = any(np.sum(a == 0) != 0 for a in maker.aya_node[1:])
            if has_zero:
                print(f"\033[31m Error!! zero token detected: {directory[i]}/{md_file[i]}\033[0m")
                continue

            is_saved, reason = maker.save(save_path)
            if is_saved:
                local_count += 1
                if stats is not None:
                    local_tokens += stats.get("total_tokens", 0)

            print(f"Process#{pid}: {md_file[i]}  saved={is_saved}  samples={stats['total_samples'] if stats else '-'}  tokens={stats.get('total_tokens', 0) if stats else '-'}  reason={reason}")

            # 移調拡張版を変換
            con_list = con.expansion_midi()
            for cl in con_list:
                cl.convert()
                maker = FoundationDataMaker(cl, min_measure, max_measure)
                stats = maker.convert()

                has_zero = any(np.sum(a == 0) != 0 for a in maker.aya_node[1:])
                if has_zero:
                    print(f"\033[31m Error!! zero token detected: {cl.file_name}\033[0m")
                    continue

                is_saved, reason = maker.save(save_path)
                if is_saved:
                    local_count += 1
                    if stats is not None:
                        local_tokens += stats.get("total_tokens", 0)

                print(f"Process#{pid}: {cl.file_name}  saved={is_saved}  tokens={stats.get('total_tokens', 0) if stats else '-'}  reason={reason}")

        except Exception as e:
            print(f"Process#{pid} error: {directory[i]}/{md_file[i]}: {e}")
            continue

    progress[pid] = {"count": local_count, "tokens": local_tokens}


if __name__ == "__main__":
    THREAD_VALUE = 10
    MIN_MEASURE = 1
    MAX_MEASURE = 8
    PROGRAM = ['PIANO', 'SAX']
    DATASETS = "/home/ubuntu/nagoshi/music_generation/data/GMD/training"
    SAVE_PATH = "/home/ubuntu/nagoshi/music_generation/dataset/music"

    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))

    directory, md_file = find_midi_files(DATASETS)
    print(f"Found {len(md_file)} MIDI files.")

    directory = np.array_split(directory, THREAD_VALUE)
    md_file = np.array_split(md_file, THREAD_VALUE)

    with Manager() as manager:
        progress = manager.dict()
        processes = []
        for t in range(THREAD_VALUE):
            p = Process(
                target=convert_foundation,
                args=(t, tokenizer, directory[t].tolist(), md_file[t].tolist(),
                      PROGRAM, MIN_MEASURE, MAX_MEASURE, progress, SAVE_PATH),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        total_count = sum(v["count"] for v in progress.values())
        total_tokens = sum(v["tokens"] for v in progress.values())
        print(f"変換完了: {total_count} ファイル保存  総トークン数: {total_tokens}")
