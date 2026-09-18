"""no-aug格子の 80M/160M を OOM回避のバッチ縮小で再実行。
no-aug(ブロック削除なし=長い系列)では v5のバッチ(80M=32,160M=16)がOOM。
flash_attnなのでメモリは系列長に線形 → バッチ半減で収まる。eff global=1024は維持。
- 80M : batch16/accum32 (=1024), lr1.2e-3
- 160M: batch8 /accum64 (=1024), lr8e-4
1.6B/3.2B はDDP uneven-batchデッドロックの可能性 → 落ちたら単一GPUで(別途)。
"""
import os, sys, json, glob, datetime, subprocess
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from run_train_many_project import estimate_total_steps, DATA_BASE, GRID_SAVE_DIR, GRID_PROJECT, SWEEP_EVAL

# (size, model_cfg, base_train, lr, batch, accum)
# no-augの長系列+ランク間偏りで 80M(b16)が15.7GB/16GBギリギリ→OOMリスク。安全側に半減。
PLAN = [
    ("80M",  "configs/models/mortm/foundation/scaling/80M.json",
     "configs/train/mortm/foundation/scaling/80M.json",  1.2e-3,  8,  64),
    ("160M", "configs/models/mortm/foundation/scaling/160M.json",
     "configs/train/mortm/foundation/scaling/160M.json", 8e-4,    4, 128),
]
CELLS = ["200M", "400M", "800M", "1.6B", "3.2B"]
RESOLVED_DIR = os.path.join(PROJECT_ROOT, "../configs/train/mortm/foundation/scaling/production/_resolved/grid")
SPEC_DIR = os.path.join(PROJECT_ROOT, "../out/models/mortm/scaling/lr_sweep/_specs")
SAVE_DIR = os.path.join(PROJECT_ROOT, GRID_SAVE_DIR)
NPROC = 2


def done(version):
    return len(glob.glob(os.path.join(SAVE_DIR, f"MORTM.{version}_*.pth"))) > 0


def main():
    dry = "--dry-run" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    os.makedirs(RESOLVED_DIR, exist_ok=True); os.makedirs(SPEC_DIR, exist_ok=True)

    runs = []
    for size, mcfg, base, lr, batch, accum in PLAN:
        if only and size not in only:
            continue
        with open(os.path.join(PROJECT_ROOT, base)) as f:
            bcfg = json.load(f)
        for label in CELLS:
            dataset = [f"{DATA_BASE}/{label}/train.json"]
            steps, _ = estimate_total_steps(dataset, batch, accum, NPROC, 40, 5000)
            version = f"{size}_D{label}_lr{lr:.2e}"
            runs.append((size, mcfg, bcfg, lr, batch, accum, label, dataset, steps, version))
            print(f"  {size} D{label:>4}: batch{batch} accum{accum} eff{batch*accum*NPROC} "
                  f"steps={steps} {version} {'DONE' if done(version) else '-'}")
    if dry:
        print("[dry-run]"); return

    for i, (size, mcfg, bcfg, lr, batch, accum, label, dataset, steps, version) in enumerate(runs):
        if done(version):
            print(f">>> SKIP {version}"); continue
        cfg = dict(bcfg)
        cfg["lr_param"] = lr; cfg["batch_size"] = batch; cfg["accumulation_steps"] = accum
        sch = dict(cfg.get("scheduler", {"type": "cos"})); sch["total_steps"] = steps
        cfg["scheduler"] = sch; cfg.pop("_note", None)
        cfg["_resolved_note"] = f"noaug {size} lr={lr:g} batch={batch} steps={steps}"
        resolved = os.path.join(RESOLVED_DIR, f"noaug_{version}.json")
        with open(resolved, "w") as f: json.dump(cfg, f, indent=2)
        spec = {"model_config": mcfg, "train_config": os.path.relpath(resolved, PROJECT_ROOT),
                "dataset": dataset, "eval": SWEEP_EVAL, "save_directory": GRID_SAVE_DIR,
                "version": version, "project_name": GRID_PROJECT}
        spec_path = os.path.join(SPEC_DIR, f"noaug_{version}.json")
        with open(spec_path, "w") as f: json.dump(spec, f, indent=2)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(NPROC),
               "--master_port", str(29610 + i), os.path.join(PROJECT_ROOT, "run_train_many_project.py"),
               "--worker", spec_path]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#'*70}\n# [{i+1}/{len(runs)}] {stamp} RUN {version} (batch{batch})\n{'#'*70}", flush=True)
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        print(f">>> {'done' if ret.returncode==0 else 'FAILED('+str(ret.returncode)+')'}: {version}")
    print("\nno-aug 80M/160M 再実行 終了。")


if __name__ == "__main__":
    main()
