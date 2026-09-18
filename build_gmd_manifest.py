import os
import json
import random

GMD_DIRS = [
    ("/mnt/nvme0n1p3/MIDIdatasets/GMD/training/all-instruments-with-drums", "all-instruments-with-drums"),
    ("/mnt/nvme0n1p3/MIDIdatasets/GMD/training/no-drums", "no-drums"),
    ("/mnt/nvme0n1p3/MIDIdatasets/Final_GigaMIDI_V2.0_Final/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/training-V1.1-80%/drums-only", "drums-only"),
]

OUT_DIR = "data/gmd"
EVAL_PER_CATEGORY = {
    "all-instruments-with-drums": 2000,
    "no-drums": 2000,
    "drums-only": 1000,
}
SEED = 42

def collect_midi_files(directory):
    midi_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, f))
    return sorted(midi_files)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    train_all = []
    eval_all = []

    print("=== Scanning GMD Dataset Directories ===")
    for dir_path, category in GMD_DIRS:
        if not os.path.exists(dir_path):
            print(f"[Warning] Directory not found: {dir_path}")
            continue

        print(f"Scanning {category} at {dir_path}...")
        files = collect_midi_files(dir_path)
        print(f"  Found {len(files):,} files in {category}")

        rng.shuffle(files)
        n_eval = EVAL_PER_CATEGORY.get(category, 1000)
        n_eval = min(n_eval, len(files))

        eval_subset = files[:n_eval]
        train_subset = files[n_eval:]

        eval_all.extend(eval_subset)
        train_all.extend(train_subset)

        print(f"  Split {category}: {len(train_subset):,} train / {len(eval_subset):,} eval")

    # 全体をシャッフル
    rng.shuffle(train_all)
    rng.shuffle(eval_all)

    train_json_path = os.path.join(OUT_DIR, "train.json")
    eval_json_path = os.path.join(OUT_DIR, "eval.json")

    print(f"\nWriting {len(train_all):,} files to {train_json_path}...")
    with open(train_json_path, "w") as f:
        json.dump(train_all, f, indent=2)

    print(f"Writing {len(eval_all):,} files to {eval_json_path}...")
    with open(eval_json_path, "w") as f:
        json.dump(eval_all, f, indent=2)

    print(f"\n=== Successfully Created Manifests ===")
    print(f"Total: {len(train_all) + len(eval_all):,} files")
    print(f"Train: {len(train_all):,} files -> {train_json_path}")
    print(f"Eval : {len(eval_all):,} files -> {eval_json_path}")

if __name__ == "__main__":
    main()
