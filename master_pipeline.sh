#!/bin/bash
# MORTM 自律実験パイプライン(設計書 v1.8)。E1評価を先に出し、その後 学習が必要な全モデルを順次学習・評価する。
# 2GPU DDP。各ステージ set +e で継続。完了マーカー: "PIPELINE ALL DONE"。
set +e
cd /home/takaaki-nagoshi/PycharmProjects/MORTM
export CUDA_VISIBLE_DEVICES=0,1
PORT=29720
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
wait_gpu_free(){ while pgrep -f "distributed.run|eval_e1_genaxes|eval_e2_|eval_analysis|eval_h2" >/dev/null 2>&1; do sleep 20; done; sleep 8; }
train_e1(){ PORT=$((PORT+1)); log "TRAIN E1 $1 $2 $3 $4"; .venv/bin/python -m torch.distributed.run --nproc_per_node=2 --max-restarts=0 --master_port=$PORT run_e1.py $1 $2 $3 $4 > "log_e1_$1_$2_$3_$4.log" 2>&1; log "done E1 $1 $4 exit=$?"; }
train_e2(){ PORT=$((PORT+1)); log "TRAIN E2 $1"; .venv/bin/python -m torch.distributed.run --nproc_per_node=2 --max-restarts=0 --master_port=$PORT run_e2.py $1 > "log_e2_$1.log" 2>&1; log "done E2 $1 exit=$?"; }

log "===== PIPELINE START ====="

# --- ステージ1: E1 確証評価(既存モデルで即実行. 生成軸/分析/継ぎ目) ---
log "### STAGE 1: E1 evaluation (既存モデル) ###"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_e1_genaxes.py 80M 200 > log_eval_genaxes80M.log 2>&1; log "genaxes80M exit=$?"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_e1_genaxes.py 10M 200 > log_eval_genaxes10M.log 2>&1; log "genaxes10M exit=$?"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_analysis_acc.py 80M > log_eval_anacc80M.log 2>&1; log "anacc80M exit=$?"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_analysis_acc.py 10M > log_eval_anacc10M.log 2>&1; log "anacc10M exit=$?"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_e2_boundary.py 1000 > log_eval_boundary.log 2>&1; log "boundary exit=$?"
log "### STAGE 1 DONE (E1 evaluation complete) ###"

# --- ステージ2: A3 3シード化(E1 §7行D, 統計). 10M/400M s1337/s2024 ---
log "### STAGE 2: A3 3-seed training (10M) ###"
for arm in A3a A3b A3c; do for seed in s1337 s2024; do
  d=out/models/paper/E1/${arm}_10M_400M_${seed}
  if ls $d/*_[1-9]*.pth >/dev/null 2>&1; then log "skip $arm $seed"; else train_e1 $arm 10M 400M $seed; wait_gpu_free; fi
done; done
log "### STAGE 2 DONE (A3 3-seed complete = E1 fully done) ###"

# --- ステージ3: E2 残アーム(H2主図). 80M FT ---
log "### STAGE 3: E2 remaining arms (80M FT) ###"
for arm in noaug_lr0.3 noaug_lr0.1 noaug_replay aug_lr1.0 aug_lr0.1; do
  d=out/models/paper/E2/${arm}
  if ls $d/*_[1-9]*.pth >/dev/null 2>&1; then log "skip E2 $arm"; else train_e2 $arm; wait_gpu_free; fi
done
log "### STAGE 3 DONE (E2 training complete) ###"

# --- ステージ4: E2 評価(補完NLL曲線 全アーム) ---
log "### STAGE 4: E2 evaluation ###"
for arm in noaug_lr1.0 noaug_lr0.3 noaug_lr0.1 noaug_replay aug_lr1.0 aug_lr0.1; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_e2_infill_nll.py $arm 500 > log_eval_e2nll_$arm.log 2>&1; log "e2_infill_nll $arm exit=$?"
done
log "===== PIPELINE ALL DONE ====="
