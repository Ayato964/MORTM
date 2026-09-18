#!/usr/bin/env python
"""300M行を既存json_v5データで学習し、N軸に高N点を追加(α同定用)。
新データセットは不要(α測定はD固定でNを振るだけ。D項は定数に吸収)。
- 全セル 2GPU, batch16×accum32 → eff global 1024 (160Mと同一=1.6Bでもデッドロックしない安全構成)
- LR=6e-4 (1/√N トレンド: 160M=8e-4基準で 300M≈5.9e-4)
- total_steps はデータから事前算出。version=300M_D{label}_lr6.00e-04、保存=既存gridディレクトリ。
"""
import os, sys, json, glob, datetime, subprocess
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from run_train_many_project import estimate_total_steps, DATA_BASE, GRID_SAVE_DIR, GRID_PROJECT, SWEEP_EVAL

LR = 6e-4
MODEL_CFG = "configs/models/mortm/foundation/scaling/300M.json"
BASE_TRAIN = "configs/train/mortm/foundation/scaling/160M.json"   # 構成流用(batch16/accum32)
CELLS = ["200M", "400M", "800M", "1.6B", "3.2B"]
BATCH, ACCUM, NPROC = 4, 128, 2  # 300Mはbatch16/8共にOOM(seq5000のattnが重い) → 4に(eff 4*128*2=1024維持)
RESOLVED_DIR = os.path.join(PROJECT_ROOT, "../configs/train/mortm/foundation/scaling/production/_resolved/grid")
SPEC_DIR = os.path.join(PROJECT_ROOT, "../out/models/mortm/scaling/lr_sweep/_specs")
SAVE_DIR = os.path.join(PROJECT_ROOT, GRID_SAVE_DIR)


def done(version):
    return len(glob.glob(os.path.join(SAVE_DIR, f"MORTM.{version}_*.pth"))) > 0


def main():
    dry = "--dry-run" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1:]
    with open(os.path.join(PROJECT_ROOT, BASE_TRAIN)) as f:
        base = json.load(f)
    os.makedirs(RESOLVED_DIR, exist_ok=True); os.makedirs(SPEC_DIR, exist_ok=True)

    cells = [c for c in CELLS if (only is None or c in only)]
    plan = []
    for label in cells:
        dataset = [f"{DATA_BASE}/{label}/train.json"]
        steps, info = estimate_total_steps(dataset, BATCH, ACCUM, NPROC, 40, 5000)
        version = f"300M_D{label}_lr{LR:.2e}"
        plan.append((label, dataset, steps, version))
        print(f"  300M D{label:>4}: nproc={NPROC} batch={BATCH} accum={ACCUM} eff={BATCH*ACCUM*NPROC} "
              f"steps={steps} {version} {'DONE' if done(version) else '-'}")
    if dry:
        print("[dry-run]"); return

    for i, (label, dataset, steps, version) in enumerate(plan):
        if done(version):
            print(f">>> SKIP {version}"); continue
        cfg = dict(base)
        cfg["lr_param"] = LR; cfg["batch_size"] = BATCH; cfg["accumulation_steps"] = ACCUM
        sched = dict(cfg.get("scheduler", {"type": "cos"})); sched["total_steps"] = steps
        cfg["scheduler"] = sched; cfg.pop("_note", None)
        cfg["_resolved_note"] = f"300M rerun lr={LR:g} steps={steps}"
        resolved = os.path.join(RESOLVED_DIR, f"{version}.json")
        with open(resolved, "w") as f: json.dump(cfg, f, indent=2)
        spec = {"model_config": MODEL_CFG, "train_config": os.path.relpath(resolved, PROJECT_ROOT),
                "dataset": dataset, "eval": SWEEP_EVAL,
                "save_directory": GRID_SAVE_DIR, "version": version, "project_name": GRID_PROJECT}
        spec_path = os.path.join(SPEC_DIR, f"{version}.json")
        with open(spec_path, "w") as f: json.dump(spec, f, indent=2)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(NPROC),
               "--master_port", str(29590 + i), os.path.join(PROJECT_ROOT, "run_train_many_project.py"),
               "--worker", spec_path]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#'*70}\n# [{i+1}/{len(plan)}] {stamp} RUN {version}\n{'#'*70}", flush=True)
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        print(f">>> {'done' if ret.returncode == 0 else 'FAILED('+str(ret.returncode)+')'}: {version}")
    print("\n300M行 終了。")


if __name__ == "__main__":
    main()
