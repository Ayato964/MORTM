import json
import os
import random
import signal
from multiprocessing import Process, Manager, Queue
from pathlib import Path

import numpy as np

from mortm.utils.convert_pianoroll import (
    midi_to_chunks,
    midi_to_roll,
    save_chunks_npz,
    save_npz,
    make_synthetic_test_midi,
    verify_roundtrip,
    roll_to_chunks,
    chunks_to_roll,
)


def find_midi_files(root_folder: str):
    midi_files = []
    direc = []
    for defpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(file)
                direc.append(defpath)
    return direc, midi_files


_QUEUE_SENTINEL = None  # ライタープロセスの停止シグナル


def writer_worker(write_queue: Queue, result_queue: Queue):
    """
    NTFS への書き込みを単一プロセスで直列化する専用ライター。
    ntfs-3g の並列 write/close/rename による D クラスデッドロックを防ぐ。
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    count = 0
    units = 0

    while True:
        task = write_queue.get()
        if task is _QUEUE_SENTINEL:
            break

        out_path, data, meta, save_as_chunks, unit_count = task
        try:
            if save_as_chunks:
                save_chunks_npz(out_path, data, meta)
            else:
                save_npz(out_path, data, meta)
            count += 1
            units += unit_count
        except Exception as e:
            print(f"[writer] ERROR  {out_path}  {e}")

    result_queue.put({"count": count, "units": units})


def convert_pianoroll(pid, directory, md_file, save_path, save_chunks, progress, write_queue):
    random.seed(os.urandom(32))
    local_count = 0
    local_units = 0
    total = len(md_file)

    for i, (d, f) in enumerate(zip(directory, md_file), 1):
        midi_path = os.path.join(d, f)
        base_name = os.path.splitext(f)[0]
        # 16個のサブディレクトリ (0-f) に分散
        sub_dir = f"{abs(hash(f)) & 0xf:x}"
        out_file = os.path.join(save_path, sub_dir, f"{base_name}.npz")

        try:
            if save_chunks:
                chunks, meta = midi_to_chunks(midi_path)
                unit_count = chunks.shape[0]
                write_queue.put((out_file, chunks, meta, True, unit_count))
            else:
                roll, meta = midi_to_roll(midi_path)
                unit_count = roll.shape[0]
                write_queue.put((out_file, roll, meta, False, unit_count))

            local_count += 1
            local_units += unit_count
            unit_label = "chunks" if save_chunks else "bars"
            print(f"[#{pid}] ({i}/{total}) QUEUED  {f}  {unit_label}={unit_count}")

        except Exception as e:
            print(f"[#{pid}] ({i}/{total}) SKIP  {f}  ({e})")
            continue

    progress[pid] = {"count": local_count, "units": local_units}


def run_selftest(output_dir: str = "./out/pianoroll_selftest") -> bool:
    """
    合成MIDIを用いたエンコード・デコード・チャンク化のRound-trip自己診断テスト。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / "synthetic_original.mid"
    dst = out / "synthetic_roundtrip.mid"
    arr = out / "synthetic_roll.npz"

    print("=== [Self-Test] Running synthetic MIDI roundtrip verification ===")
    make_synthetic_test_midi(src)
    roll, meta = midi_to_roll(src)
    save_npz(arr, roll, meta)
    verify_roundtrip(src, dst)
    print(f"PASS: {src} -> {arr} -> {dst}")

    chunks = roll_to_chunks(roll)
    restored_roll = chunks_to_roll(chunks)
    assert np.array_equal(roll, restored_roll), "Chunks roundtrip mismatch!"
    print(f"PASS: Chunks roundtrip verified! (bars={roll.shape[0]}, chunks={chunks.shape[0]})")
    print(f"Roll shape={roll.shape}, dtype={roll.dtype}, range=[{roll.min():.4f}, {roll.max():.4f}]")
    print(
        f"Chunks shape={chunks.shape}, dtype={chunks.dtype}, "
        f"range=[{chunks.min():.4f}, {chunks.max():.4f}]"
    )
    print("Slot configuration:")
    for slot in meta.slots:
        print(f"  {slot}")
    print("=== [Self-Test] All tests passed successfully! ===\n")
    return True


if __name__ == "__main__":
    THREAD_VALUE = 25
    SAVE_CHUNKS = True  # True: 24-step chunks [N, 8, 24, 128] / False: bar-wise roll [bars, 8, 192, 128]
    RUN_SELFTEST = True
    DATASETS = "/media/takaaki-nagoshi/C8DCDBF4DCDBDB30/MIDIdatasets/GMD/training"
    SAVE_PATH = "/media/takaaki-nagoshi/C8DCDBF4DCDBDB30/MORTM/pre_train/ver5/pianoroll"
    SELFTEST_DIR = "./out/pianoroll_selftest"

    # 1. 自己診断テスト (合成MIDIのエンコード・復元・整合性検証)
    if RUN_SELFTEST:
        run_selftest(SELFTEST_DIR)

    # 2. 実データセットが存在する場合は並列バッチ変換
    if not os.path.exists(DATASETS):
        print(f"[Info] DATASETS パスが見つかりません: {DATASETS}")
        print("実データセットの変換を行う場合は、DATASETS および SAVE_PATH を設定してください。")
    else:
        # NTFS (ntfs-3g) は並列 mkdir で ENOSPC を返すバグがあるため、
        # プロセス起動前にサブディレクトリを一括作成しておく（16個に抑える）
        os.makedirs(SAVE_PATH, exist_ok=True)
        for h in "0123456789abcdef":
            os.makedirs(os.path.join(SAVE_PATH, h), exist_ok=True)

        directory, md_file = find_midi_files(DATASETS)
        print(f"Found {len(md_file)} MIDI files in {DATASETS}.")

        if len(md_file) == 0:
            print("変換対象のMIDIファイルが見つかりませんでした。")
        else:
            num_threads = min(THREAD_VALUE, len(md_file))
            directory_split = np.array_split(directory, num_threads)
            md_file_split = np.array_split(md_file, num_threads)

            # 書き込みキュー: maxsize でバックプレッシャーをかけてメモリを抑制
            write_queue = Queue(maxsize=200)
            result_queue = Queue()

            # ライタープロセスを先に起動（全 NTFS 書き込みをここで直列化）
            writer = Process(
                target=writer_worker,
                args=(write_queue, result_queue),
                daemon=False,
            )
            writer.start()

            with Manager() as manager:
                progress = manager.dict()
                processes = []
                for t in range(num_threads):
                    p = Process(
                        target=convert_pianoroll,
                        args=(
                            t,
                            directory_split[t].tolist(),
                            md_file_split[t].tolist(),
                            SAVE_PATH,
                            SAVE_CHUNKS,
                            progress,
                            write_queue,
                        ),
                    )
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                total_queued = sum(v["count"] for v in progress.values())

            # 全ワーカー終了後にライターへ停止シグナルを送る
            write_queue.put(_QUEUE_SENTINEL)
            writer.join()

            writer_result = result_queue.get()
            unit_label = "チャンク" if SAVE_CHUNKS else "小節"
            print(
                f"変換完了: {writer_result['count']} ファイル保存  "
                f"総{unit_label}数: {writer_result['units']}"
            )
            if total_queued != writer_result["count"]:
                print(f"[警告] キュー投入数({total_queued}) != 書き込み完了数({writer_result['count']})")
