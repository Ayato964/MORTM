import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import sys
import subprocess

"""
MORTM5 (MidiVision) - GMD (GigaMIDI Dataset, 81.4万曲) 本番学習エントリーポイント
run_train.py と同様に単体実行および torchrun DDP 実行に対応しています。

実行例:
    .venv/bin/python run_gmd.py
または
    .venv/bin/torchrun --nproc_per_node=2 run_gmd.py
"""

if __name__ == "__main__":
    # torchrun 経由でない場合は利用可能な GPU 数に応じて自動起動
    if "WORLD_SIZE" not in os.environ:
        import torch
        nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            __file__,
        ] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))

    # DDP 実行部
    from run_midi_vision import main

    default_defaults = {
        "--model_config": "configs/models/mortm5/default.json",
        "--train_config": "configs/train/mortm5/default.json",
        "--data_dir": "data/gmd/train.json",
        "--eval_json": "data/gmd/eval.json",
        "--save_dir": "out/models/mortm5/",
        "--version": "MORTM5_GMD_full810k",
        "--project_name": "MORTM5_MidiVision",
    }

    # 指定されていないオプションのみデフォルト構成で補完
    for opt, val in default_defaults.items():
        if opt not in sys.argv:
            sys.argv.extend([opt, val])

    main()
