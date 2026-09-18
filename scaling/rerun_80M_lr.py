#!/usr/bin/env python
"""
80M行だけ LR=1.2e-3 で撮り直す (v5格子のN軸ノイズ補正)。
旧80MはLR=8e-4(トレンド~1.1e-3に対し低すぎ)でundertrained。他4サイズはトレンド適正なので触らない。

- 4セル(200M/400M/800M/3.2B): 2GPU, batch32×accum16 → eff 1024
- 1セル(1.6B): DDP uneven-batchデッドロック回避で単一GPU, batch32×accum32 → eff 1024
total_stepsはデータから事前算出(eff globalは両方1024なのでnproc非依存で一致)。
新versionは lr1.20e-03 で旧 lr8.00e-04 と衝突しない。保存先は既存gridディレクトリ。
"""
import os, sys, json, glob, datetime, subprocess
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from run_train_many_project import estimate_total_steps, DATA_BASE, GRID_SAVE_DIR, GRID_PROJECT, SWEEP_EVAL

LR = 1.2e-3
MODEL_CFG = "configs/models/mortm/foundation/scaling/80M.json"
BASE_TRAIN = "configs/train/mortm/foundation/scaling/80M.json"
CELLS = ["200M", "400M", "800M", "1.6B", "3.2B"]
SINGLE_GPU = {"1.6B"}                      # デッドロック回避
RESOLVED_DIR = os.path.join(PROJECT_ROOT, "../configs/train/mortm/foundation/scaling/production/_resolved/grid")
SPEC_DIR = os.path.join(PROJECT_ROOT, "../out/models/mortm/scaling/lr_sweep/_specs")
SAVE_DIR = os.path.join(PROJECT_ROOT, GRID_SAVE_DIR)


def done(version):
    return len(glob.glob(os.path.join(SAVE_DIR, f"MORTM.{version}_*.pth"))) > 0


def main():
    dry = "--dry-run" in sys.argv
    with open(os.path.join(PROJECT_ROOT, BASE_TRAIN)) as f:
        base = json.load(f)
    os.makedirs(RESOLVED_DIR, exist_ok=True); os.makedirs(SPEC_DIR, exist_ok=True)

    plan = []
    for label in CELLS:
        nproc = 1 if label in SINGLE_GPU else 2
        batch = 32
        accum = 32 if nproc == 1 else 16          # eff global = batch*accum*nproc = 1024 共通
        dataset = [f"{DATA_BASE}/{label}/train.json"]
        total_steps, info = estimate_total_steps(dataset, batch, accum, nproc, 40, 5000)
        version = f"80M_D{label}_lr{LR:.2e}"
        plan.append((label, nproc, batch, accum, dataset, total_steps, version))
        print(f"  80M D{label:>4}: nproc={nproc} batch={batch} accum={accum} eff={batch*accum*nproc} "
              f"steps={total_steps} {version} {'DONE' if done(version) else '-'}")
    if dry:
        print("[dry-run]"); return

    for i, (label, nproc, batch, accum, dataset, total_steps, version) in enumerate(plan):
        if done(version):
            print(f">>> SKIP {version}"); continue
        cfg = dict(base)
        cfg["lr_param"] = LR; cfg["batch_size"] = batch; cfg["accumulation_steps"] = accum
        sched = dict(cfg.get("scheduler", {"type": "cos"})); sched["total_steps"] = total_steps
        cfg["scheduler"] = sched; cfg.pop("_note", None)
        cfg["_resolved_note"] = f"80M rerun lr={LR:g} steps={total_steps} nproc={nproc}"
        resolved = os.path.join(RESOLVED_DIR, f"{version}_n{nproc}.json")
        with open(resolved, "w") as f: json.dump(cfg, f, indent=2)
        spec = {"model_config": MODEL_CFG, "train_config": os.path.relpath(resolved, PROJECT_ROOT),
                "dataset": dataset, "eval": SWEEP_EVAL,
                "save_directory": GRID_SAVE_DIR, "version": version, "project_name": GRID_PROJECT}
        spec_path = os.path.join(SPEC_DIR, f"{version}.json")
        with open(spec_path, "w") as f: json.dump(spec, f, indent=2)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
               "--master_port", str(29580 + i), os.path.join(PROJECT_ROOT, "run_train_many_project.py"),
               "--worker", spec_path]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#'*70}\n# [{i+1}/{len(plan)}] {stamp} RUN {version} (nproc={nproc})\n{'#'*70}", flush=True)
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        print(f">>> {'done' if ret.returncode == 0 else 'FAILED('+str(ret.returncode)+')'}: {version}")
    print("\n80M行 撮り直し 終了。")


if __name__ == "__main__":
    main()
