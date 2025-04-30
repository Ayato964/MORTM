import os

from mortm.convert import PackSeq


def find_seq_files(root_folder):
    midi_files = []
    direc = []
    for defpath, surnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.npz')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


def convert_pack(directory, md_file):

    sp = PackSeq(directory, md_file)
    sp.convert()
    sp.save("./out/np/Piano/pack_small/", "Datasets_large")

    pass



if __name__ == "__main__":
    datasets = "./out/np/Piano/datasets_large/"
    directory, md_file = find_seq_files(datasets)
    convert_pack(directory[0], md_file)
