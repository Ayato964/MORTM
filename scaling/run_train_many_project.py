#!/usr/bin/env python
"""
run_train_many_project.py
=========================
複数モデル / 複数ハイパーパラメータを順番に学習させる汎用トレーニング・プロトコル。
`run_train.py` を「1モデル1回」から「N条件を逐次実行」へ拡張したもの。

主な用途は **スケーリング則のための学習率スイープ**:
  各パラメータサイズについて、バッチ条件（実質512バッチ = batch_size × accumulation_steps）を
  固定したまま学習率だけを振り、最良の val_loss を出す学習率を求める。

設計
----
* このファイルは 2 モードを持つ単一スクリプト:
    - **orchestrator**（引数なし / --dry-run）: 各 run の設定を解決し、
      run ごとに **新しい torchrun サブプロセス** を起動する。
      run ごとにプロセスを分けることで、CUDA/NCCL の状態が毎回クリーンになり、
      NCCL の再初期化ハングや GPU メモリリークを避けられる（夜間連続実行でも堅牢）。
    - **worker**（--worker SPEC.json）: torchrun 配下で 1 条件だけを学習する。
      内部で `train_mortm` を呼ぶ（run_train.py と同等）。

* total_steps は「データセットを 1 epoch 流したときに実際に発生する optimizer step 数」を
  NPZ をサンプリングして **事前算出** し、cosine スケジューラに渡す。
  同一スイープ内の全 run は同じデータ・同じバッチ構成なので total_steps は共通になり、
  学習率比較は公平になる。

使い方
------
    # 計画だけ確認（学習はしない）
    python run_train_many_project.py --dry-run

    # スイープ実行（各 run を torchrun --nproc_per_node=NPROC で起動）
    python run_train_many_project.py
    MORTM_NPROC=2 python run_train_many_project.py     # GPU 枚数を明示

注意
----
* このプロジェクトの config は「GPU 1 枚あたり」のハイパーパラメータ。
  有効バッチ = batch_size × accumulation_steps（= 512）を全 run で固定し、学習率のみ変える。
* DDP は GPU 2 枚構成（NPROC=2）を既定とする。
"""

import os
import sys
import json
import glob
import random
import argparse
import datetime
import subprocess

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_BASE = os.path.dirname(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# スイープ定義（汎用: dict を増やせば他サイズも回せる）
# ---------------------------------------------------------------------------
# v5: 12転調データ (221,709曲×12=11.78Bトークン) から再構築した json_v5 を使う。
# 全ラベル実体通り (1.6B/3.2B も真サイズ)。eval は v5 で新規分割 (旧 json/ の L 値と比較不可)。
DATA_BASE = "/home/takaaki-nagoshi/data/scaling/json_v5_noaug"

# 学習率スイープ対象。1 エントリ = 1 パラメータサイズ。
# base_train_config の lr_param をベース学習率として lr_multipliers を掛ける。
# LR スイープは全サイズ共通で 200M データを使う（短い固定ステップ予算 ≈350 steps）。
# 有効バッチ(batch×accum)=512 を全サイズ固定しているので total_steps も全サイズ共通になり、
# サイズ間も同一トークン/ステップ予算で LR を公平に比較できる。
# （各サイズのフル学習は Chinchilla ペア 10M↔200M/20M↔400M/40M↔800M/80M↔1.6B で別途実施する）
SWEEP_DATASET = [f"{DATA_BASE}/200M/train.json"]
SWEEP_EVAL = [f"{DATA_BASE}/eval.json"]
SWEEP_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]

SWEEPS = [
    {
        "size": "10M",
        "model_config":      "configs/models/mortm/foundation/scaling/10M.json",
        "base_train_config": "configs/train/mortm/foundation/scaling/10M.json",
        "dataset": SWEEP_DATASET,
        "eval":    SWEEP_EVAL,
        "save_directory": "out/models/mortm/scaling/lr_sweep/10M",
        "project_name":   "MORTM_Scaling_LR_Sweep_10M",
        "lr_multipliers": SWEEP_MULTS,
    },
    {
        "size": "20M",
        "model_config":      "configs/models/mortm/foundation/scaling/20M.json",
        "base_train_config": "configs/train/mortm/foundation/scaling/20M.json",
        "dataset": SWEEP_DATASET,
        "eval":    SWEEP_EVAL,
        "save_directory": "out/models/mortm/scaling/lr_sweep/20M",
        "project_name":   "MORTM_Scaling_LR_Sweep_20M",
        "lr_multipliers": SWEEP_MULTS,
    },
    {
        "size": "40M",
        "model_config":      "configs/models/mortm/foundation/scaling/40M.json",
        "base_train_config": "configs/train/mortm/foundation/scaling/40M.json",
        "dataset": SWEEP_DATASET,
        "eval":    SWEEP_EVAL,
        "save_directory": "out/models/mortm/scaling/lr_sweep/40M",
        "project_name":   "MORTM_Scaling_LR_Sweep_40M",
        "lr_multipliers": SWEEP_MULTS,
    },
    {
        "size": "80M",
        "model_config":      "configs/models/mortm/foundation/scaling/80M.json",
        "base_train_config": "configs/train/mortm/foundation/scaling/80M.json",
        "dataset": SWEEP_DATASET,
        "eval":    SWEEP_EVAL,
        "save_directory": "out/models/mortm/scaling/lr_sweep/80M",
        "project_name":   "MORTM_Scaling_LR_Sweep_80M",
        "lr_multipliers": SWEEP_MULTS,
    },
    {
        "size": "160M",
        "model_config":      "configs/models/mortm/foundation/scaling/160M.json",
        "base_train_config": "configs/train/mortm/foundation/scaling/160M.json",
        # LR スイープは固定 200M データで実施（3.2B は本番フル学習用で、スイープには不要）。
        "dataset": SWEEP_DATASET,
        "eval":    SWEEP_EVAL,
        "save_directory": "out/models/mortm/scaling/lr_sweep/160M",
        "project_name":   "MORTM_Scaling_LR_Sweep_160M",
        "lr_multipliers": SWEEP_MULTS,
    },
]

# ---------------------------------------------------------------------------
# 本実験（フル Chinchilla 学習）定義
# ---------------------------------------------------------------------------
# LR スイープで求めた各サイズのベスト学習率を使い、Chinchilla ペア(D≈20xN)の
# データセットでフル 1 epoch 学習する。これで L(N,D) スケーリング則を実測する。
# wandb の optional_logging が val_loss / tokens / flops(6*active_params*tokens) を
# 記録するので、それを使って L(N) / L(C) をフィットできる。
PRODUCTION = [
    {"size": "10M", "lr": 3.2e-3,
     "model_config":      "configs/models/mortm/foundation/scaling/10M.json",
     "base_train_config": "configs/train/mortm/foundation/scaling/10M.json",
     "dataset": [f"{DATA_BASE}/200M/train.json"]},
    {"size": "20M", "lr": 2.4e-3,
     "model_config":      "configs/models/mortm/foundation/scaling/20M.json",
     "base_train_config": "configs/train/mortm/foundation/scaling/20M.json",
     "dataset": [f"{DATA_BASE}/400M/train.json"]},
    {"size": "40M", "lr": 1.6e-3,
     "model_config":      "configs/models/mortm/foundation/scaling/40M.json",
     "base_train_config": "configs/train/mortm/foundation/scaling/40M.json",
     "dataset": [f"{DATA_BASE}/800M/train.json"]},
    {"size": "80M", "lr": 1.2e-3,   # v5で確定した補正LR(旧8e-4はundertrained)
     "model_config":      "configs/models/mortm/foundation/scaling/80M.json",
     "base_train_config": "configs/train/mortm/foundation/scaling/80M.json",
     "dataset": [f"{DATA_BASE}/1.6B/train.json"]},
    {"size": "160M", "lr": 8e-4,
     "model_config":      "configs/models/mortm/foundation/scaling/160M.json",
     "base_train_config": "configs/train/mortm/foundation/scaling/160M.json",
     "dataset": [f"{DATA_BASE}/3.2B/train.json"]},
]
PRODUCTION_SAVE_DIR = "out/models/mortm/scaling_v5_noaug/production"
PRODUCTION_PROJECT = "MORTM_Scaling_Production_V5_noaug"

# ---------------------------------------------------------------------------
# 格子実験 (full grid) 定義
# ---------------------------------------------------------------------------
# 分離可能な L(N,D) = E + A/N^α + B/D^β を求めるには、N と D を独立に振る格子が要る。
# 各モデルサイズ(best LR 固定) × 全データセット の総当たり。対角(D≈20N)は production で
# 学習済みなので、その .pth を grid 命名でシードしておけば already_done でスキップされる。
# 各セルの LR はそのサイズのベスト LR を使う（セル毎の LR 再チューニングはしない簡略化）。
GRID_DATASETS = [
    ("200M", [f"{DATA_BASE}/200M/train.json"]),
    ("400M", [f"{DATA_BASE}/400M/train.json"]),
    ("800M", [f"{DATA_BASE}/800M/train.json"]),
    ("1.6B", [f"{DATA_BASE}/1.6B/train.json"]),
    ("3.2B", [f"{DATA_BASE}/3.2B/train.json"]),
]
GRID_SAVE_DIR = "../out/models/mortm/scaling_v5_noaug/grid"
GRID_PROJECT = "MORTM_Scaling_Grid_V5_noaug"

NPROC = int(os.environ.get("MORTM_NPROC", "2"))
RESOLVED_DIR = os.path.join(PROJECT_ROOT, "../configs/train/mortm/foundation/scaling/lr_sweep/_resolved")
RESOLVED_DIR_PROD = os.path.join(PROJECT_ROOT, "../configs/train/mortm/foundation/scaling/production/_resolved")
SPEC_DIR = os.path.join(PROJECT_ROOT, "../out/models/mortm/scaling/lr_sweep/_specs")
SAMPLE_NPZ = 400          # total_steps 推定に使う NPZ サンプル数
SAMPLE_SEED = 0           # 再現性のための固定シード
STEP_MARGIN = 1.08        # total_steps に乗せる安全マージン。
                          # データ epoch の端数(部分 outer バッチ)とサンプリング誤差で
                          # 実 optimizer step が推定をわずかに超え、末尾が cosine 下限(5%LR)で
                          # 平坦化するのを防ぐ。total_steps を実ステップ数より気持ち大きめにして
                          # cosine を全区間で効かせる。

MODEL_NAME = "MORTM"      # MORTMArgs.name（保存ファイル名の接頭辞）


# ---------------------------------------------------------------------------
# total_steps の事前算出
# ---------------------------------------------------------------------------
def estimate_total_steps(dataset_jsons, batch_size, accumulation_steps, world_size,
                         min_length, position_length, sample=SAMPLE_NPZ, seed=SAMPLE_SEED):
    """データセットを 1 epoch 流したときの optimizer step 数を推定する。

    各 NPZ には複数の系列 (array1, array2, ...) が入っており、
    min_length < len < position_length を満たす系列だけが学習に使われる。
    NPZ をサンプリングして「NPZ あたり平均有効系列数」を出し、
      総系列数 ≈ 平均系列数 × NPZ 総数
      optimizer step ≈ 総系列数 / (batch_size × accumulation_steps × world_size)
    で見積もる。同一スイープ内では同じ値になるので学習率比較は公平。
    """
    all_paths = []
    for dj in dataset_jsons:
        with open(dj) as f:
            all_paths.extend(json.load(f))
    n_total = len(all_paths)
    if n_total == 0:
        raise RuntimeError(f"dataset is empty: {dataset_jsons}")

    rng = random.Random(seed)
    picked = rng.sample(all_paths, min(sample, n_total))

    seqs = 0
    files_ok = 0
    for p in picked:
        try:
            with np.load(p, allow_pickle=True) as d:
                i = 1
                while f"array{i}" in d.files:
                    a = np.asarray(d[f"array{i}"])
                    if a.ndim > 0 and min_length < len(a) < position_length:
                        seqs += 1
                    i += 1
            files_ok += 1
        except FileNotFoundError:
            pass

    if files_ok == 0:
        raise RuntimeError(f"no readable NPZ found for {dataset_jsons}")

    avg_seq_per_npz = seqs / files_ok
    est_total_seq = avg_seq_per_npz * n_total
    eff_batch = batch_size * accumulation_steps
    raw_steps = est_total_seq / (eff_batch * world_size)
    steps = max(int(raw_steps * STEP_MARGIN), 1)
    info = {
        "n_npz": n_total,
        "sampled_npz": files_ok,
        "avg_seq_per_npz": round(avg_seq_per_npz, 3),
        "est_total_seq": int(est_total_seq),
        "eff_batch_per_rank": eff_batch,
        "world_size": world_size,
        "raw_steps": int(raw_steps),
        "step_margin": STEP_MARGIN,
        "total_steps": steps,
    }
    return steps, info


def resolve_train_config(base_train_config, lr, total_steps, out_path):
    """ベース学習 config をコピーし、lr_param と total_steps を上書きして保存する。"""
    with open(base_train_config) as f:
        cfg = json.load(f)
    cfg["lr_param"] = lr
    sched = dict(cfg.get("scheduler", {"type": "cos"}))
    sched["total_steps"] = total_steps
    # warmup_steps が無ければ warmup_ratio から config.py 側で計算される。
    cfg["scheduler"] = sched
    cfg.pop("_note", None)
    cfg["_resolved_note"] = (
        f"auto-resolved by run_train_many_project.py: lr={lr:g}, total_steps={total_steps}"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def build_runs():
    """SWEEPS を展開して run のリストを作る（学習はまだしない）。"""
    runs = []
    for sw in SWEEPS:
        with open(os.path.join(PROJECT_BASE, sw["base_train_config"])) as f:
            base_cfg = json.load(f)
        with open(os.path.join(PROJECT_BASE, sw["model_config"])) as f:
            model_cfg = json.load(f)

        base_lr = base_cfg["lr_param"]
        batch_size = base_cfg["batch_size"]
        accum = base_cfg["accumulation_steps"]
        min_length = model_cfg.get("min_length", 40)
        position_length = model_cfg.get("position_length", 5000)

        total_steps, est_info = estimate_total_steps(
            sw["dataset"], batch_size, accum, NPROC, min_length, position_length
        )

        for mult in sw["lr_multipliers"]:
            lr = base_lr * mult
            version = f"{sw['size']}_x{mult:g}_lr{lr:.2e}"
            resolved = os.path.join(RESOLVED_DIR, sw["size"], f"{version}.json")
            runs.append({
                "size": sw["size"],
                "version": version,
                "mult": mult,
                "lr": lr,
                "model_config": sw["model_config"],
                "base_train_config": sw["base_train_config"],
                "resolved_train_config": resolved,
                "dataset": sw["dataset"],
                "eval": sw["eval"],
                "save_directory": sw["save_directory"],
                "project_name": sw["project_name"],
                "total_steps": total_steps,
                "batch_size": batch_size,
                "accumulation_steps": accum,
                "est_info": est_info,
            })
    return runs


def build_production_runs():
    """PRODUCTION を展開して本実験 run のリストを作る（各サイズ best LR × Chinchilla ペア）。"""
    runs = []
    for p in PRODUCTION:
        with open(os.path.join(PROJECT_BASE, p["base_train_config"])) as f:
            base_cfg = json.load(f)
        with open(os.path.join(PROJECT_BASE, p["model_config"])) as f:
            model_cfg = json.load(f)

        batch_size = base_cfg["batch_size"]
        accum = base_cfg["accumulation_steps"]
        min_length = model_cfg.get("min_length", 40)
        position_length = model_cfg.get("position_length", 5000)

        total_steps, est_info = estimate_total_steps(
            p["dataset"], batch_size, accum, NPROC, min_length, position_length
        )

        lr = p["lr"]
        # データセット規模をディレクトリ名から取得して version に含める
        # 例: ".../200M/train.json" -> "200M" -> version "10M_D200M_full_lr3.20e-03"
        data_label = os.path.basename(os.path.dirname(p["dataset"][0]))
        version = f"{p['size']}_D{data_label}_full_lr{lr:.2e}"
        resolved = os.path.join(RESOLVED_DIR_PROD, f"{version}.json")
        runs.append({
            "size": p["size"],
            "version": version,
            "mult": lr / base_cfg["lr_param"],  # 表示用（base比）
            "lr": lr,
            "model_config": p["model_config"],
            "base_train_config": p["base_train_config"],
            "resolved_train_config": resolved,
            "dataset": p["dataset"],
            "eval": SWEEP_EVAL,
            "save_directory": PRODUCTION_SAVE_DIR,
            "project_name": PRODUCTION_PROJECT,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "accumulation_steps": accum,
            "est_info": est_info,
        })
    return runs


def build_grid_runs():
    """格子(各サイズ best LR × 全データセット)の run リストを作る。"""
    runs = []
    for p in PRODUCTION:  # モデル仕様と best LR は PRODUCTION を流用
        with open(os.path.join(PROJECT_BASE, p["base_train_config"])) as f:
            base_cfg = json.load(f)
        with open(os.path.join(PROJECT_BASE, p["model_config"])) as f:
            model_cfg = json.load(f)
        batch_size = base_cfg["batch_size"]
        accum = base_cfg["accumulation_steps"]
        min_length = model_cfg.get("min_length", 40)
        position_length = model_cfg.get("position_length", 5000)
        lr = p["lr"]

        for data_label, dataset in GRID_DATASETS:
            total_steps, est_info = estimate_total_steps(
                dataset, batch_size, accum, NPROC, min_length, position_length
            )
            version = f"{p['size']}_D{data_label}_lr{lr:.2e}"
            resolved = os.path.join(RESOLVED_DIR_PROD, "grid", f"{version}.json")
            runs.append({
                "size": p["size"],
                "version": version,
                "mult": lr / base_cfg["lr_param"],
                "lr": lr,
                "model_config": p["model_config"],
                "base_train_config": p["base_train_config"],
                "resolved_train_config": resolved,
                "dataset": dataset,
                "eval": SWEEP_EVAL,
                "save_directory": GRID_SAVE_DIR,
                "project_name": GRID_PROJECT,
                "total_steps": total_steps,
                "batch_size": batch_size,
                "accumulation_steps": accum,
                "est_info": est_info,
            })
    return runs


def already_done(run):
    pattern = os.path.join(PROJECT_ROOT, run["save_directory"], f"{MODEL_NAME}.{run['version']}_*.pth")
    return len(glob.glob(pattern)) > 0


def print_plan(runs):
    print("=" * 78)
    print(f"LR スイープ計画  (NPROC={NPROC}, 有効バッチ=batch×accum×nproc)")
    print("=" * 78)
    cur_size = None
    for r in runs:
        if r["size"] != cur_size:
            cur_size = r["size"]
            ei = r["est_info"]
            print(f"\n[{cur_size}]  base_train={r['base_train_config']}")
            print(f"   dataset={r['dataset']}")
            print(f"   est: {ei['n_npz']} npz, {ei['avg_seq_per_npz']} seq/npz, "
                  f"~{ei['est_total_seq']:,} seq  ->  total_steps={ei['total_steps']}")
            print(f"   eff_batch/rank={r['batch_size']}x{r['accumulation_steps']}="
                  f"{r['batch_size'] * r['accumulation_steps']}  (global x{NPROC})")
            print(f"   {'version':<28} {'mult':>5} {'lr':>11}  done?")
        mark = "DONE" if already_done(r) else "-"
        print(f"   {r['version']:<28} {r['mult']:>5g} {r['lr']:>11.3e}  {mark}")
    print("\n" + "=" * 78)


def write_spec(run):
    os.makedirs(SPEC_DIR, exist_ok=True)
    spec = {
        "model_config": run["model_config"],
        "train_config": os.path.relpath(run["resolved_train_config"], PROJECT_ROOT),
        "dataset": run["dataset"],
        "eval": run["eval"],
        "save_directory": run["save_directory"],
        "version": run["version"],
        "project_name": run["project_name"],
    }
    spec_path = os.path.join(SPEC_DIR, f"{run['version']}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    return spec_path


def orchestrate(dry_run=False, only=None, production=False, grid=False):
    if grid:
        runs = build_grid_runs()
    elif production:
        runs = build_production_runs()
    else:
        runs = build_runs()
    if only:
        runs = [r for r in runs if r["version"] in only or r["size"] in only]
    print_plan(runs)

    if dry_run:
        print("[dry-run] 学習は実行しません。")
        return

    for i, run in enumerate(runs):
        if already_done(run):
            print(f"\n>>> SKIP (already done): {run['version']}")
            continue

        os.makedirs(os.path.join(PROJECT_ROOT, run["save_directory"]), exist_ok=True)
        resolve_train_config(
            os.path.join(PROJECT_ROOT, run["base_train_config"]),
            run["lr"], run["total_steps"], run["resolved_train_config"],
        )
        spec_path = write_spec(run)

        port = 29500 + i
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(NPROC),
            "--master_port", str(port),
            os.path.abspath(__file__),
            "--worker", spec_path,
        ]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#' * 78}")
        print(f"# [{i + 1}/{len(runs)}] {stamp}  RUN {run['version']}  "
              f"(lr={run['lr']:.3e}, total_steps={run['total_steps']})")
        print(f"# {' '.join(cmd)}")
        print(f"{'#' * 78}", flush=True)

        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            print(f"!!! run failed (returncode={ret.returncode}): {run['version']}")
            print("!!! 次の run に進みます。")
        else:
            print(f">>> done: {run['version']}")

    print("\nすべての run が終了しました。")


# ---------------------------------------------------------------------------
# worker（torchrun 配下で 1 条件だけ学習）
# ---------------------------------------------------------------------------
def run_worker(spec_path):
    from mortm.train.train import train_mortm
    from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
    from mortm.utils.messager import _DefaultMessenger

    with open(spec_path) as f:
        spec = json.load(f)

    tokenizer = Tokenizer(get_token_converter_pro(TO_MUSIC))
    train_mortm(
        tokenizer,
        spec["model_config"],
        spec["train_config"],
        tuple(spec["dataset"]),
        spec["save_directory"],
        spec["version"],
        log_scale=True,
        project_name=spec["project_name"],
        message=_DefaultMessenger(),
        # インストール版 train_mortm は eval_list_json の tuple を反復しない(find_files_with_json
        # に丸ごと渡す)ため、単一eval jsonは str で渡す。複数時のみ tuple。
        eval_list_json=(spec["eval"][0] if len(spec["eval"]) == 1 else tuple(spec["eval"])),
    )


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", metavar="SPEC.json", default=None,
                        help="（内部用）torchrun 配下で 1 条件だけ学習する")
    parser.add_argument("--dry-run", action="store_true",
                        help="計画とtotal_steps推定だけ表示して終了")
    parser.add_argument("--only", nargs="*", default=None,
                        help="指定した version / size だけ実行")
    parser.add_argument("--production", action="store_true",
                        help="本実験モード: 各サイズ best LR × Chinchilla ペアでフル学習")
    parser.add_argument("--grid", action="store_true",
                        help="格子モード: 各サイズ best LR × 全データセット総当たり(L(N,D)分離用)")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker)
    else:
        orchestrate(dry_run=args.dry_run, only=args.only,
                    production=args.production, grid=args.grid)


if __name__ == "__main__":
    main()
