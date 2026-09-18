"""SFT分析タスク用の train.json / eval.json インデックスファイルを生成する。
docs/splits/ の曲分割テキストに基づき、data/sft/analysis/music/ の全 npz パスを振り分ける。
"""
import os
import json
import glob

SPLIT_DIR = "/home/takaaki-nagoshi/PycharmProjects/MORTM/docs/splits"
MUSIC_DIR = "/home/takaaki-nagoshi/data/sft/analysis/music"
OUT_DIR = "/home/takaaki-nagoshi/data/sft/analysis"

def load_songs(name):
    path = os.path.join(SPLIT_DIR, f"{name}_songs.txt")
    if not os.path.exists(path):
        return set()
    return set(open(path).read().split())

def main():
    train_songs = load_songs("train")
    val_songs = load_songs("val")
    test_songs = load_songs("test")
    print(f"Loaded splits: train={len(train_songs)}, val={len(val_songs)}, test={len(test_songs)}")

    train_paths = []
    val_paths = []
    test_paths = []

    # music 以下の全 npz ファイルを検索
    pattern = os.path.join(MUSIC_DIR, "**", "*.npz")
    npz_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(npz_files)} npz files in {MUSIC_DIR}")

    for path in npz_files:
        filename = os.path.basename(path)
        # e.g., 01d9f42b358c1af3ddd972a28300d047.mid.npz -> 01d9f42b358c1af3ddd972a28300d047
        song_hash = filename.replace(".mid.npz", "").replace(".midi.npz", "")
        if song_hash in train_songs:
            train_paths.append(path)
        elif song_hash in val_songs:
            val_paths.append(path)
        elif song_hash in test_songs:
            test_paths.append(path)
        else:
            # 万が一どのスプリットにも属さない曲があれば警告
            pass

    print(f"Assigned paths: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # 書き出し
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train.json"), "w") as f:
        json.dump(train_paths, f, indent=2)
    with open(os.path.join(OUT_DIR, "eval.json"), "w") as f:
        json.dump(val_paths, f, indent=2)
    
    print("Done generating train.json and eval.json for SFT analysis!")

if __name__ == "__main__":
    main()
