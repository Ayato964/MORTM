"""トークナイザ比較 基盤モデル学習 (珠玉3点, プランII)。
共通130,500曲。A提案60M/B REMI86M/C提案127M。LRは1/√Nトレンド経験則。
eff global batch = batch×accum×2 = 1024 で統一。run_train_many_project の worker を流用。
"""
import os, sys, json, glob, datetime, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_train_many_project import estimate_total_steps  # noqa

DATA = "/home/takaaki-nagoshi/data/scaling/tokencmp"
SAVE_DIR = "out/models/mortm/tokencmp"
PROJECT = "MORTM_TokenizerCompare"
RESOLVED = os.path.join(HERE, "configs_resolved_tokencmp");
SPEC_DIR = os.path.join(PROJECT_ROOT, SAVE_DIR, "_specs")

# (version, model_config, arm, lr, batch, accum)
RUNS = [
    ("A_prop_60M",  "configs/models/mortm/tokencmp/A_prop_60M.json",  "proposed", 1.3e-3, 16, 32),
    ("B_remi_86M",  "configs/models/mortm/tokencmp/B_remi_86M.json",  "remi",     1.2e-3,  8, 64),
    ("C_prop_127M", "configs/models/mortm/tokencmp/C_prop_127M.json", "proposed", 9.0e-4,  8, 64),
]
NPROC = 2
STEP_MARGIN = 1.08


def done(v):
    return len(glob.glob(os.path.join(PROJECT_ROOT, SAVE_DIR, f"MORTM.{v}_*.pth"))) > 0


def main():
    dry = "--dry-run" in sys.argv
    only = sys.argv[sys.argv.index("--only")+1:] if "--only" in sys.argv else None
    os.makedirs(RESOLVED, exist_ok=True); os.makedirs(SPEC_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, SAVE_DIR), exist_ok=True)

    plan = []
    for ver, mcfg, arm, lr, batch, accum in RUNS:
        if only and ver not in only: continue
        ds = [f"{DATA}/{arm}/train.json"]; ev = [f"{DATA}/{arm}/eval.json"]
        steps, info = estimate_total_steps(ds, batch, accum, NPROC, 40, 5000)
        plan.append((ver, mcfg, arm, lr, batch, accum, ds, ev, steps))
        print(f"  {ver:<12} arm={arm:<8} lr={lr:.1e} batch{batch}/accum{accum} eff={batch*accum*NPROC} "
              f"steps={steps} {'DONE' if done(ver) else '-'}")
    if dry:
        print("[dry-run]"); return

    for i, (ver, mcfg, arm, lr, batch, accum, ds, ev, steps) in enumerate(plan):
        if done(ver): print(f">>> SKIP {ver}"); continue
        with open(os.path.join(PROJECT_ROOT, "configs/train/mortm/foundation/scaling/80M.json")) as f:
            cfg = json.load(f)
        cfg["lr_param"] = lr; cfg["batch_size"] = batch; cfg["accumulation_steps"] = accum
        cfg["scheduler"] = {"type": "cos", "warmup_steps": round(0.05 * steps),
                            "warmup_ratio": 0.05, "total_steps": steps}
        cfg.pop("_note", None); cfg["_resolved_note"] = f"tokencmp {ver} lr={lr:g} b{batch}/a{accum} steps={steps}"
        rp = os.path.join(RESOLVED, f"{ver}.json"); json.dump(cfg, open(rp, "w"), indent=2)
        spec = {"model_config": mcfg, "train_config": os.path.relpath(rp, PROJECT_ROOT),
                "dataset": ds, "eval": ev, "save_directory": SAVE_DIR, "version": ver, "project_name": PROJECT}
        sp = os.path.join(SPEC_DIR, f"{ver}.json"); json.dump(spec, open(sp, "w"), indent=2)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(NPROC),
               "--master_port", str(29650+i), os.path.join(HERE, "run_train_many_project.py"), "--worker", sp]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#'*70}\n# [{i+1}/{len(plan)}] {stamp} RUN {ver} (arm={arm}, lr={lr:.1e}, b{batch})\n{'#'*70}", flush=True)
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        print(f">>> {'done' if ret.returncode==0 else 'FAILED('+str(ret.returncode)+')'}: {ver}")
    print("\nトークナイザ比較 学習 終了。")


if __name__ == "__main__":
    main()
