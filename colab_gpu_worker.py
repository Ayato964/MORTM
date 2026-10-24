#!/usr/bin/env python3
"""
colab_gpu_worker.py — Google Colab FastAPI Remote GPU Worker (MORTM v5.0.0b3 Beta)
===================================================================================
モデルのアーキテクチャ (`MidiVision`, `AblationTransformer1`, `AblationTransformer2`,
`MORTM5TrainSet`, `HybridMuonOptimizer`, `MORTM5MemmapDataset`) はすべて pip パッケージ
`mortm` (`pip install --pre -U mortm`) から直接呼び出します。

【Google Colab での起動手順】
1. ランタイムを GPU (A100 / L4 / T4) に設定し、以下を実行:
   !pip install -q --pre -U "mortm==5.0.0b3" vector-quantize-pytorch pretty_midi mido fastapi uvicorn pycloudflared
2. Google Drive をマウント (トークンデータやチェックポイントを Drive に置く場合):
   from google.colab import drive
   drive.mount('/content/drive')
3. サーバースクリプトを起動して Cloudflare Tunnel URL を取得:
   !wget -q -O colab_gpu_worker.py https://raw.githubusercontent.com/Ayato964/MORTM/mortm5-tokenizer-mac/colab_gpu_worker.py
   !nohup python3 -u colab_gpu_worker.py > worker.log 2>&1 &
   !sleep 5 && cat worker.log
"""

import copy
import datetime
import io
import json
import math
import os
import shutil
import threading
import zlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from fastapi import FastAPI, Request, Response
    import uvicorn
except ImportError:
    uvicorn = None  # type: ignore

    class Request:  # type: ignore
        pass

    class Response:  # type: ignore
        def __init__(self, content=b"", status_code: int = 200, media_type: str = "application/octet-stream"):
            self.content = content
            self.status_code = status_code
            self.media_type = media_type

    class FastAPI:  # type: ignore
        def get(self, _path: str):
            return lambda fn: fn

        def post(self, _path: str):
            return lambda fn: fn

import mortm
from mortm import (
    MORTM5Args,
    TrainArgs,
    PAD_TOKEN_ID,
    MidiVision,
    AblationTransformer1,
    AblationTransformer2,
    build_mortm5_ablation_model,
    MORTM5TrainSet,
    MORTM5MemmapDataset,
    MORTM5Collate,
    MORTM5WindowSubsetDataset,
)
from mortm.train.train import _compute_exact_epoch_optimizer_steps

torch.set_float32_matmul_precision("high")


# ============================================================
# 1. MidiVision (CNN+RVQ Tokenizer) 用損失関数 (後方互換維持)
# ============================================================
class MidiVisionLoss(nn.Module):
    def __init__(self, pos_weight: float = 12.0, onset_weight: float = 2.0, vel_weight: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.onset_weight = onset_weight
        self.vel_weight = vel_weight
        self.last_metrics: Dict[str, Any] = {}

    def forward(self, recon_logits: torch.Tensor, target: torch.Tensor, commit_loss: torch.Tensor) -> torch.Tensor:
        logits_active = recon_logits[:, 0::2].float()
        pred_vel = torch.sigmoid(recon_logits[:, 1::2].float())
        target_active = (target[:, 0::2] > 0.5).float()
        target_vel = target[:, 1::2].float()

        bce_active = F.binary_cross_entropy_with_logits(
            logits_active, target_active, pos_weight=self.pos_weight.to(logits_active.device)
        )
        prob_active = torch.sigmoid(logits_active)
        inter = (prob_active * target_active).sum(dim=(1, 2, 3))
        union = prob_active.sum(dim=(1, 2, 3)) + target_active.sum(dim=(1, 2, 3))
        dice_active = (1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)).mean()
        loss_active = bce_active + dice_active

        target_onset = (target_vel > 0.25).float() * target_active
        onset_mask = target_active > 0.5
        if onset_mask.any():
            pred_onset_logit = (pred_vel - 0.375) * 12.0
            loss_onset = F.binary_cross_entropy_with_logits(pred_onset_logit[onset_mask], target_onset[onset_mask])
        else:
            loss_onset = logits_active.new_tensor(0.0)

        onset_sum = target_onset.sum()
        if onset_sum > 0:
            vel_diff = ((pred_vel - target_vel) * 2.0).pow(2) * target_onset
            loss_vel = vel_diff.sum() / (onset_sum + 1e-5)
        else:
            loss_vel = logits_active.new_tensor(0.0)

        commit_scalar = commit_loss.float().mean() if isinstance(commit_loss, torch.Tensor) else float(commit_loss)
        loss_commit = 0.25 * commit_scalar

        with torch.no_grad():
            pred_bin = (prob_active > 0.5).float()
            tp = (pred_bin * target_active).sum().item()
            fp = (pred_bin * (1.0 - target_active)).sum().item()
            fn = ((1.0 - pred_bin) * target_active).sum().item()
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall + 1e-8)) if (precision + recall) > 0 else 0.0

            pred_onset_bin = ((pred_vel > 0.375).float() * pred_bin)
            otp = (pred_onset_bin * target_onset).sum().item()
            ofp = (pred_onset_bin * (1.0 - target_onset)).sum().item()
            ofn = ((1.0 - pred_onset_bin) * target_onset).sum().item()
            onset_recall = otp / (otp + ofn + 1e-8) if (otp + ofn) > 0 else 0.0
            onset_precision = otp / (otp + ofp + 1e-8) if (otp + ofp) > 0 else 0.0
            onset_f1 = (
                2.0 * onset_precision * onset_recall / (onset_precision + onset_recall + 1e-8)
                if (onset_precision + onset_recall) > 0
                else 0.0
            )
            vel_mae = (
                (((pred_vel.clamp(0.5, 1.0) - target_vel).abs() * 2.0 * target_onset).sum().item() / (onset_sum.item() + 1e-5))
                if onset_sum > 0
                else 0.0
            )

        total_loss = loss_active + self.onset_weight * loss_onset + self.vel_weight * loss_vel + loss_commit
        self.last_metrics = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "onset_f1": onset_f1,
            "onset_recall": onset_recall,
            "onset_precision": onset_precision,
            "vel_mae": vel_mae,
            "bce": bce_active.item(),
            "dice": dice_active.item(),
            "onset_loss": loss_onset.item(),
            "vel_loss": loss_vel.item(),
            "commit": commit_scalar.item() if isinstance(commit_scalar, torch.Tensor) else float(commit_scalar),
            "total": total_loss.item(),
        }
        return total_loss


# ============================================================
# 2. FastAPI Remote GPU Worker
# ============================================================
app = FastAPI()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
STATE: Dict[str, Any] = {
    "model": None,
    "optimizer": None,
    "criterion": None,
    "step": 0,
    "lr": 4e-4,
    # MORTM5 Autoregressive Transformer (`MORTM5TrainSet`) 用状態
    "trainer": None,
    "m_args": None,
    "t_args": None,
    "train_ds": None,
    "val_ds": None,
    "collate_fn": None,
    "total_steps": 0,
    "epoch": 0,
    "outer_batch_idx": 0,
    "mortm5_job_results": [],
    "mortm5_job_done": False,
    "mortm5_job_error": None,
    "upload_parts": [],
}


@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(DEVICE)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0
    trainer: Optional[MORTM5TrainSet] = STATE.get("trainer")
    return {
        "status": "ok",
        "mortm_version": getattr(mortm, "__version__", "unknown"),
        "device": str(DEVICE),
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "step": int(STATE.get("step", 0)),
        "all_tokens": int(trainer.get_synced_tokens()) if trainer is not None else 0,
        "model_class": type(trainer.model).__name__ if trainer is not None else (
            type(STATE["model"]).__name__ if STATE.get("model") is not None else None
        ),
    }


# ============================================================
# 3. MORTM5 Autoregressive Transformer (Ablation #1-#8) Endpoints
# ============================================================
_DRIVE_SAVE_LOCK = threading.Lock()

_ABLATION_SUBDIR_MAP: Dict[str, str] = {
    "#1": "1", "1": "1", "m3_flat_shared": "1", "mortm5_m3_flat_shared": "1",
    "#2": "2", "2": "2", "m3_flat_separated": "2", "mortm5_m3_flat_separated": "2",
    "#3": "3", "3": "3", "m4_flat_shared": "3", "mortm5_m4_flat_shared": "3",
    "#4": "4", "4": "4", "m4_flat_separated": "4", "mortm5_m4_flat_separated": "4",
    "#5": "5", "5": "5", "m3_grouped_shared": "5", "mortm5_m3_grouped_shared": "5",
    "#6": "6", "6": "6", "m3_grouped_separated": "6", "mortm5_m3_grouped_separated": "6",
    "#7": "7", "7": "7", "m4_grouped_shared": "7", "mortm5_m4_grouped_shared": "7",
    "#8": "8", "8": "8", "m4_grouped_separated": "8", "mortm5_m4_grouped_separated": "8",
}


def _resolve_ablation_subdir(model_name: str = "", ablation_id: str = "") -> str:
    """アブレーション識別子 ('#1'〜'#8', '1'〜'8') またはモデル名から Google Drive サブディレクトリ名 ('1'〜'8') を返す。"""
    for raw in (ablation_id, model_name):
        key = str(raw or "").strip().lower()
        if not key:
            continue
        if key in _ABLATION_SUBDIR_MAP:
            return _ABLATION_SUBDIR_MAP[key]
        stripped = key.lstrip("#")
        if stripped in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            return stripped
    return "default"


def _get_drive_ckpt_dir(ablation_subdir: str) -> str:
    base_dir = os.environ.get("MORTM5_DRIVE_CKPT_DIR", "/content/drive/MyDrive/MORTM5_checkpoints")
    return os.path.join(base_dir, ablation_subdir) if ablation_subdir else base_dir


def _get_worker_dir() -> str:
    """Colab 上では `/content`、ローカルテスト環境では `MORTM5_WORKER_DIR` または一時ディレクトリを返す。"""
    import tempfile
    work_dir = os.environ.get(
        "MORTM5_WORKER_DIR",
        "/content" if os.path.isdir("/content") else os.path.join(tempfile.gettempdir(), "mortm5_worker"),
    )
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


@app.post("/mortm5/init")
async def mortm5_init(request: Request):
    """
    pip パッケージ `mortm` の `MORTM5Args`, `TrainArgs`, `MORTM5TrainSet` (内部で
    `build_mortm5_ablation_model` + `build_muon_optimizer` + `get_cosine_schedule_with_warmup` を使用)
    をそのまま呼び出して Colab GPU 上にモデルとオプティマイザを構築する。
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        raw = await request.body()
        payload = torch.load(io.BytesIO(zlib.decompress(raw)), map_location="cpu", weights_only=False)

    # 既存モデルが GPU メモリを占有している場合は先に解放する
    if STATE.get("trainer") is not None or STATE.get("model") is not None:
        STATE["trainer"] = None
        STATE["model"] = None
        STATE["optimizer"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_cfg = payload.get("model_config", "m4_grouped_shared")
    train_cfg = payload.get("train_config", "muon_cos")
    m_args = MORTM5Args(model_cfg)
    t_args = TrainArgs(train_cfg)
    if "optimizer" not in getattr(t_args, "_raw_config", {}):
        t_args.optimizer_type = "muon"

    ablation_subdir = _resolve_ablation_subdir(
        model_name=m_args.name,
        ablation_id=str(payload.get("ablation_id", "")),
    )
    STATE["ablation_subdir"] = ablation_subdir

    # 1エポック学習で二度と使わないトークン列のローカル SSD キャッシュは行わず、Google Drive から直接 Memmap 読み込みする
    colab_split_dir: Optional[str] = payload.get("colab_split_dir")
    stride: Optional[int] = payload.get("stride")
    val_stride: Optional[int] = payload.get("val_stride")

    train_ds: Optional[MORTM5MemmapDataset] = None
    val_ds: Optional[MORTM5MemmapDataset] = None
    if colab_split_dir:
        if not os.path.isdir(colab_split_dir):
            return Response(
                content=json.dumps({"error": f"colab_split_dir not found: {colab_split_dir}"}),
                status_code=400,
                media_type="application/json",
            )
        train_ds = MORTM5MemmapDataset.from_split_dir(
            token_dir=colab_split_dir, split="train", args=m_args, stride=stride
        )
        val_tokens_path = os.path.join(colab_split_dir, "val_tokens.bin")
        if os.path.isfile(val_tokens_path):
            val_ds = MORTM5MemmapDataset.from_split_dir(
                token_dir=colab_split_dir, split="val", args=m_args, stride=val_stride
            )

    # MORTM5TrainSet (AblationTransformer1/2 + HybridMuonOptimizer + Cosine/WSD Scheduler) の構築
    trainer = MORTM5TrainSet(m_args=m_args, t_args=t_args, device=DEVICE)

    # total_steps の決定とスケジューラ再構築
    explicit_total_steps = payload.get("total_steps")
    if explicit_total_steps is not None and int(explicit_total_steps) > 0:
        total_steps = int(explicit_total_steps)
    elif train_ds is not None:
        steps_per_epoch = _compute_exact_epoch_optimizer_steps(
            num_windows=len(train_ds),
            big_batch_size=int(t_args.big_batch_size),
            batch_size=int(t_args.batch_size),
            accumulation_steps=int(t_args.accumulation_steps),
        )
        total_steps = max(1, steps_per_epoch * max(1, int(t_args.num_epochs)))
    else:
        total_steps = int(t_args.scheduler.get("total_steps", 17500))

    trainer.rebuild_scheduler(total_steps)

    # チェックポイントの復元 (指定パス または Google Drive の 1-8 サブディレクトリ内チェックポイント)
    resumed_from: Optional[str] = None
    resume_ckpt_path: Optional[str] = payload.get("resume_checkpoint_path")
    version_str = str(payload.get("version") or f"{m_args.name}_muon_cos").strip()
    if resume_ckpt_path and not os.path.isfile(resume_ckpt_path):
        # サブディレクトリ (1-8) 側および旧フラットディレクトリ側をフォールバック探索
        cand_in_sub = os.path.join(_get_drive_ckpt_dir(ablation_subdir), os.path.basename(resume_ckpt_path))
        if os.path.isfile(cand_in_sub):
            resume_ckpt_path = cand_in_sub

    ckpt_data: Optional[Dict[str, Any]] = None
    if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
        ckpt_data = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        resumed_from = resume_ckpt_path
    elif payload.get("model_state_dict") is not None:
        ckpt_data = payload
        resumed_from = "uploaded_payload"

    if ckpt_data is not None:
        sd = ckpt_data.get("model_state_dict", ckpt_data)
        trainer.model.load_state_dict(sd, strict=True)
        if ckpt_data.get("optimizer_state_dict") is not None:
            trainer.optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        if ckpt_data.get("scheduler_state_dict") is not None and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
        STATE["step"] = int(ckpt_data.get("optimizer_step", ckpt_data.get("step", 0)))
        STATE["epoch"] = int(ckpt_data.get("epoch", 0))
        STATE["outer_batch_idx"] = int(ckpt_data.get("outer_batch_idx", 0))
        STATE["verification_loss"] = float(ckpt_data.get("verification_loss", 0.0))
        trainer.all_tokens.fill_(int(ckpt_data.get("all_tokens", 0)))
        trainer.last_val_calc_tokens = int(ckpt_data.get("last_val_calc_tokens", 0))
    else:
        STATE["step"] = 0
        STATE["epoch"] = 0
        STATE["outer_batch_idx"] = 0
        STATE["verification_loss"] = 0.0
        if bool(payload.get("clean_old_checkpoints", False)) and version_str:
            prefix = f"MORTM5.{version_str}"
            drive_ckpt_dir = _get_drive_ckpt_dir(ablation_subdir)
            for target_dir in (_get_worker_dir(), drive_ckpt_dir):
                if os.path.isdir(target_dir):
                    for fname in os.listdir(target_dir):
                        if fname.startswith(prefix) and fname.endswith((".pt", ".pth", ".tmp")):
                            try:
                                os.remove(os.path.join(target_dir, fname))
                            except OSError:
                                pass

    trainer.model.train()
    STATE["trainer"] = trainer
    STATE["model"] = trainer.model
    STATE["optimizer"] = trainer.optimizer
    STATE["criterion"] = trainer.criterion
    STATE["m_args"] = m_args
    STATE["t_args"] = t_args
    STATE["train_ds"] = train_ds
    STATE["val_ds"] = val_ds
    STATE["collate_fn"] = MORTM5Collate(
        mode=m_args.model_mode,
        vq_groups=m_args.vq_groups,
        pad_token_id=PAD_TOKEN_ID,
    )
    STATE["total_steps"] = total_steps

    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    muon_params = (
        sum(p.numel() for g in trainer.optimizer.muon.param_groups for p in g["params"])
        if hasattr(trainer.optimizer, "muon") and trainer.optimizer.muon is not None
        else 0
    )
    adamw_params = (
        sum(p.numel() for g in trainer.optimizer.adamw.param_groups for p in g["params"])
        if hasattr(trainer.optimizer, "adamw") and trainer.optimizer.adamw is not None
        else total_params
    )

    return {
        "status": "initialized",
        "mortm_version": getattr(mortm, "__version__", "unknown"),
        "device": str(DEVICE),
        "model_name": m_args.name,
        "ablation_subdir": ablation_subdir,
        "drive_ckpt_dir": _get_drive_ckpt_dir(ablation_subdir),
        "model_class": type(trainer.model).__name__,
        "model_mode": m_args.model_mode,
        "separate_rvq_layers": m_args.separate_rvq_layers,
        "vocab_size": m_args.vocab_size,
        "position_length": m_args.position_length,
        "total_params": total_params,
        "muon_params": muon_params,
        "adamw_params": adamw_params,
        "total_steps": total_steps,
        "train_windows": len(train_ds) if train_ds is not None else 0,
        "val_windows": len(val_ds) if val_ds is not None else 0,
        "step": STATE["step"],
        "epoch": STATE["epoch"],
        "outer_batch_idx": STATE["outer_batch_idx"],
        "all_tokens": int(trainer.get_synced_tokens()),
        "verification_loss": float(STATE.get("verification_loss", 0.0)),
        "resumed_from": resumed_from,
    }


def _extract_lrs(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    lrs = [float(pg["lr"]) for pg in optimizer.param_groups if "lr" in pg]
    if not lrs:
        return 0.0, 0.0
    return lrs[0], lrs[-1]


def _run_mortm5_outer_batch_worker(
    window_indices: Optional[List[int]],
    uploaded_comp: Optional[bytes],
    outer_idx: int,
    epoch: int,
):
    """
    Colab GPU 上で 1 Outer Batch (例: 2048 windows = 64 mini-batches = 32 Muon optimizer steps)
    を実行するバックグラウンドワーカー。MORTM5TrainSet.epoch_fc を直接呼び出す。
    """
    try:
        trainer: MORTM5TrainSet = STATE["trainer"]
        t_args: TrainArgs = STATE["t_args"]
        collate_fn: MORTM5Collate = STATE["collate_fn"]
        model = trainer.model
        optimizer = trainer.optimizer
        scheduler = trainer.scheduler
        criterion = trainer.criterion
        model.train()

        batch_size = max(1, int(t_args.batch_size))
        accum_steps = max(1, int(t_args.accumulation_steps))
        amp_dtype = torch.bfloat16 if DEVICE.type in ("cuda", "cpu") else torch.float16

        # データソースの構築 (Mode A: Colab 上の MemmapDataset / Mode B: Mac からアップロードされたトークン配列)
        if window_indices is not None and STATE.get("train_ds") is not None:
            sub_ds = MORTM5WindowSubsetDataset(STATE["train_ds"], window_indices)
            inner_loader = DataLoader(
                sub_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=2 if DEVICE.type == "cuda" else 0,
                pin_memory=(DEVICE.type == "cuda"),
            )
            batches_iter = inner_loader
            total_micro_batches = len(inner_loader)
        elif uploaded_comp:
            header_len = int.from_bytes(uploaded_comp[:4], byteorder="big")
            header = json.loads(uploaded_comp[4 : 4 + header_len].decode("utf-8"))
            decomp = zlib.decompress(uploaded_comp[4 + header_len :])
            np_dtype = np.dtype(header.get("dtype", "int32"))
            arr = np.frombuffer(decomp, dtype=np_dtype).copy().reshape(header["shape"])
            del uploaded_comp, decomp
            perm = np.random.permutation(len(arr))
            arr = arr[perm]
            micro_list = [
                torch.from_numpy(arr[i : i + batch_size]).to(dtype=torch.long)
                for i in range(0, len(arr), batch_size)
            ]
            batches_iter = micro_list
            total_micro_batches = len(micro_list)
        else:
            raise RuntimeError("Neither window_indices nor uploaded token tensor was provided.")

        optimizer.zero_grad(set_to_none=True)
        accum_loss_sum = 0.0
        accum_micro_count = 0
        accum_valid_tokens = 0

        for b_idx, pack in enumerate(batches_iter):
            tokens_before = int(trainer.get_synced_tokens())
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype):
                logits_flat, target_flat = trainer.epoch_fc(model, pack, None)
            tokens_after = int(trainer.get_synced_tokens())
            valid_tokens = max(0, tokens_after - tokens_before)
            if int((target_flat != PAD_TOKEN_ID).sum().item()) == 0:
                continue

            is_last_micro = (b_idx + 1) == total_micro_batches
            will_step = ((accum_micro_count + 1) >= accum_steps) or is_last_micro
            effective_div = accum_steps if (accum_micro_count + 1) >= accum_steps else (accum_micro_count + 1)

            with torch.autocast(device_type=DEVICE.type, enabled=False):
                raw_loss = criterion(logits_flat.float(), target_flat)

            loss_val = float(raw_loss.detach().item())
            scaled_loss = raw_loss / float(effective_div if will_step and accum_micro_count == 0 else accum_steps)
            if will_step and (accum_micro_count + 1) < accum_steps and accum_micro_count > 0:
                scaled_loss = raw_loss / float(accum_steps)

            scaled_loss.backward()
            if will_step and (accum_micro_count + 1) < accum_steps and accum_micro_count > 0:
                scale_fix = float(accum_steps) / float(accum_micro_count + 1)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale_fix)

            accum_loss_sum += loss_val
            accum_micro_count += 1
            accum_valid_tokens += valid_tokens

            if will_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                STATE["step"] += 1
                mean_step_loss = accum_loss_sum / max(1, accum_micro_count)
                ppl = math.exp(min(20.0, mean_step_loss))
                lr_muon, lr_adamw = _extract_lrs(optimizer)
                total_toks_now = int(trainer.get_synced_tokens())

                step_metric = {
                    "step": int(STATE["step"]),
                    "outer_idx": int(outer_idx),
                    "epoch": int(epoch),
                    "loss": mean_step_loss,
                    "ppl": ppl,
                    "lr_muon": lr_muon,
                    "lr_adamw": lr_adamw,
                    "step_tokens": accum_valid_tokens,
                    "all_tokens": total_toks_now,
                }
                STATE["mortm5_job_results"].append(step_metric)
                if STATE["step"] % 20 == 0:
                    print(
                        f"[MORTM5 Step {STATE['step']}/{STATE['total_steps']}] "
                        f"Loss:{mean_step_loss:.4f} | PPL:{ppl:.2f} | "
                        f"LR(Muon):{lr_muon:.2e} | Tokens:{total_toks_now/1e6:.2f}M",
                        flush=True,
                    )
                accum_loss_sum = 0.0
                accum_micro_count = 0
                accum_valid_tokens = 0

        STATE["outer_batch_idx"] = int(outer_idx)
        STATE["epoch"] = int(epoch)
        STATE["mortm5_job_done"] = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE["mortm5_job_error"] = str(e)
        STATE["mortm5_job_done"] = True


@app.post("/mortm5/start_outer_batch")
async def mortm5_start_outer_batch(request: Request):
    content_type = request.headers.get("content-type", "")
    body_json: Dict[str, Any] = {}
    if "application/json" in content_type:
        body_json = await request.json()

    outer_idx = int(body_json.get("outer_idx", request.query_params.get("outer_idx", "0")))
    epoch = int(body_json.get("epoch", request.query_params.get("epoch", "0")))
    window_indices: Optional[List[int]] = body_json.get("window_indices")

    # window_indices が直接渡されず outer_idx & big_batch_size 指定の場合、Colab 側でスライス可能
    if window_indices is None and STATE.get("train_ds") is not None and not STATE.get("upload_parts"):
        big_bs = int(STATE["t_args"].big_batch_size)
        n_win = len(STATE["train_ds"])
        start = outer_idx * big_bs
        end = min(n_win, start + big_bs)
        window_indices = list(range(start, end))

    uploaded_comp: Optional[bytes] = None
    if STATE.get("upload_parts"):
        uploaded_comp = b"".join(STATE["upload_parts"])
        STATE["upload_parts"] = []

    STATE["mortm5_job_results"] = []
    STATE["mortm5_job_done"] = False
    STATE["mortm5_job_error"] = None

    t = threading.Thread(
        target=_run_mortm5_outer_batch_worker,
        args=(window_indices, uploaded_comp, outer_idx, epoch),
        daemon=True,
    )
    t.start()
    return {
        "status": "started",
        "outer_idx": outer_idx,
        "epoch": epoch,
        "num_windows": len(window_indices) if window_indices is not None else 0,
        "uploaded_bytes": len(uploaded_comp) if uploaded_comp is not None else 0,
    }


@app.get("/mortm5/poll_outer_batch")
async def mortm5_poll_outer_batch(request: Request):
    cursor = int(request.query_params.get("cursor", "0"))
    res = STATE.get("mortm5_job_results", [])
    new_items = res[cursor:]
    trainer: Optional[MORTM5TrainSet] = STATE.get("trainer")
    return {
        "results": new_items,
        "next_cursor": cursor + len(new_items),
        "done": bool(STATE.get("mortm5_job_done", False)),
        "error": STATE.get("mortm5_job_error"),
        "step": int(STATE.get("step", 0)),
        "all_tokens": int(trainer.get_synced_tokens()) if trainer is not None else 0,
    }


@app.post("/mortm5/validate")
async def mortm5_validate(request: Request):
    """Colab GPU 上で検証セット (`val_ds` またはアップロードされたトークン) の CrossEntropy 損失と PPL を計算する。"""
    trainer: MORTM5TrainSet = STATE["trainer"]
    t_args: TrainArgs = STATE["t_args"]
    collate_fn: MORTM5Collate = STATE["collate_fn"]
    model = trainer.model
    criterion = trainer.criterion

    max_batches = int(request.query_params.get("max_batches", "0"))
    amp_dtype = torch.bfloat16 if DEVICE.type in ("cuda", "cpu") else torch.float16

    val_ds: Optional[MORTM5MemmapDataset] = STATE.get("val_ds")
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(t_args.batch_size),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2 if DEVICE.type == "cuda" else 0,
            pin_memory=(DEVICE.type == "cuda"),
        )
        batches_iter = val_loader
    elif STATE.get("upload_parts"):
        raw_comp = b"".join(STATE["upload_parts"])
        STATE["upload_parts"] = []
        header_len = int.from_bytes(raw_comp[:4], byteorder="big")
        header = json.loads(raw_comp[4 : 4 + header_len].decode("utf-8"))
        decomp = zlib.decompress(raw_comp[4 + header_len :])
        np_dtype = np.dtype(header.get("dtype", "int32"))
        arr = np.frombuffer(decomp, dtype=np_dtype).copy().reshape(header["shape"])
        bs = int(t_args.batch_size)
        batches_iter = [torch.from_numpy(arr[i : i + bs]).to(dtype=torch.long) for i in range(0, len(arr), bs)]
    else:
        return {"val_loss": 0.0, "val_ppl": 1.0, "val_tokens": 0, "step": int(STATE["step"])}

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    val_tokens = 0

    with torch.no_grad():
        for b_idx, pack in enumerate(batches_iter):
            if 0 < max_batches <= b_idx:
                break
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype):
                logits_flat, target_flat = trainer.epoch_fc(model, pack, None)
            valid = int((target_flat != PAD_TOKEN_ID).sum().item())
            if valid == 0:
                continue
            with torch.autocast(device_type=DEVICE.type, enabled=False):
                loss = criterion(logits_flat.float(), target_flat)
            loss_sum += float(loss.item())
            n_batches += 1
            val_tokens += valid

    if was_training:
        model.train()

    val_loss = loss_sum / max(1, n_batches)
    val_ppl = math.exp(min(20.0, val_loss))
    trainer.last_val_calc_tokens = int(trainer.get_synced_tokens())
    return {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_tokens": val_tokens,
        "val_batches": n_batches,
        "step": int(STATE["step"]),
        "all_tokens": int(trainer.get_synced_tokens()),
    }


@app.post("/mortm5/save_ckpt")
async def mortm5_save_ckpt(request: Request):
    """
    MORTM5 チェックポイントを Google Drive のアブレーション番号別サブディレクトリ
    (`/content/drive/MyDrive/MORTM5_checkpoints/1/` 〜 `8/`) に直接保存する。
    Google Drive マウント時は `/content` ローカル SSD への二重保存を行わず、
    検証バッチ実行中にバックグラウンドスレッド（排他ロック付き）で Drive へ書き込む。
    """
    trainer: MORTM5TrainSet = STATE["trainer"]
    m_args: MORTM5Args = STATE["m_args"]
    t_args: TrainArgs = STATE["t_args"]

    outer_idx = int(request.query_params.get("outer", str(STATE.get("outer_batch_idx", 0))))
    epoch = int(request.query_params.get("epoch", str(STATE.get("epoch", 0))))
    total_outer = int(request.query_params.get("total_outer", "0"))
    val_loss = float(request.query_params.get("val_loss", "0.0"))
    w_id = request.query_params.get("wandb_id", "")
    version = request.query_params.get("version", m_args.name)
    ablation_id = request.query_params.get("ablation_id", str(STATE.get("ablation_subdir", "")))
    ablation_subdir = _resolve_ablation_subdir(model_name=m_args.name, ablation_id=ablation_id)
    prepare_download = int(request.query_params.get("prepare_download", "0"))
    milestone_tag = request.query_params.get("milestone", "")
    sync_drive = int(request.query_params.get("sync_drive", "0"))

    step_now = int(STATE["step"])
    tokens_now = int(trainer.get_synced_tokens())
    cpu_model_sd = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}

    full_ckpt = {
        "model_state_dict": cpu_model_sd,
        "optimizer_state_dict": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(trainer.scheduler.state_dict()) if trainer.scheduler is not None else None,
        "epoch": epoch,
        "outer_batch_idx": outer_idx,
        "total_outer_batches": total_outer,
        "optimizer_step": step_now,
        "all_tokens": tokens_now,
        "last_val_calc_tokens": int(trainer.last_val_calc_tokens),
        "verification_loss": val_loss,
        "wandb_run_id": w_id,
        "model_name": m_args.name,
        "ablation_subdir": ablation_subdir,
        "model_config": m_args.to_dict(),
        "train_config": t_args.to_dict(),
        "version": version,
        "mortm_version": getattr(mortm, "__version__", "unknown"),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    drive_mounted = os.path.exists("/content/drive/MyDrive") or bool(os.environ.get("MORTM5_DRIVE_CKPT_DIR"))
    drive_dir = _get_drive_ckpt_dir(ablation_subdir)
    dst_ckpt_path = os.path.join(drive_dir, f"MORTM5.{version}.checkpoint.pt")

    if drive_mounted:
        def _write_to_drive_subdir(ckpt_obj, model_sd, target_dir, ver, s_now, m_tag):
            with _DRIVE_SAVE_LOCK:
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    dst_ckpt = os.path.join(target_dir, f"MORTM5.{ver}.checkpoint.pt")
                    tmp_ckpt = os.path.join(target_dir, f"MORTM5.{ver}.checkpoint.{s_now}.tmp")
                    torch.save(ckpt_obj, tmp_ckpt)
                    os.replace(tmp_ckpt, dst_ckpt)

                    dst_pth = os.path.join(target_dir, f"MORTM5.{ver}_latest.pth")
                    tmp_pth = os.path.join(target_dir, f"MORTM5.{ver}_latest.{s_now}.tmp")
                    torch.save(model_sd, tmp_pth)
                    os.replace(tmp_pth, dst_pth)

                    if m_tag:
                        ms_pth = os.path.join(target_dir, f"MORTM5.{ver}_{m_tag}_step{s_now}.pth")
                        shutil.copy2(dst_pth, ms_pth)
                except Exception as e:
                    print(f"[Drive Save Error ({target_dir})] {e}", flush=True)

        if sync_drive:
            _write_to_drive_subdir(full_ckpt, cpu_model_sd, drive_dir, version, step_now, milestone_tag)
        else:
            threading.Thread(
                target=_write_to_drive_subdir,
                args=(full_ckpt, cpu_model_sd, drive_dir, version, step_now, milestone_tag),
                daemon=True,
            ).start()
        saved_path = dst_ckpt_path
        local_saved = False
    else:
        # ローカルテスト環境 (Google Drive 非マウント時のみ)
        local_sub_dir = os.path.join(_get_worker_dir(), ablation_subdir)
        os.makedirs(local_sub_dir, exist_ok=True)
        local_ckpt = os.path.join(local_sub_dir, f"MORTM5.{version}.checkpoint.pt")
        torch.save(full_ckpt, local_ckpt + ".tmp")
        os.replace(local_ckpt + ".tmp", local_ckpt)
        # 旧テスト互換のためルートにも配置
        legacy_ckpt = os.path.join(_get_worker_dir(), f"MORTM5.{version}.checkpoint.pt")
        shutil.copy2(local_ckpt, legacy_ckpt)
        saved_path = local_ckpt
        local_saved = True

    total_dl_bytes = 0
    if prepare_download:
        buf = io.BytesIO()
        torch.save(
            {
                "model_state_dict": cpu_model_sd,
                "optimizer_step": step_now,
                "outer_batch_idx": outer_idx,
                "all_tokens": tokens_now,
                "verification_loss": val_loss,
            },
            buf,
        )
        comp = zlib.compress(buf.getvalue(), level=1)
        STATE["ckpt_bytes"] = comp
        total_dl_bytes = len(comp)

    return {
        "status": "saved",
        "colab_path": saved_path,
        "drive_dir": drive_dir,
        "ablation_subdir": ablation_subdir,
        "local_saved": local_saved,
        "drive_saved": drive_mounted,
        "download_bytes": total_dl_bytes,
        "step": step_now,
        "outer": outer_idx,
        "tokens": tokens_now,
        "version": version,
    }


# ============================================================
# 4. 共通アップロード & MidiVision 後方互換エンドポイント
# ============================================================
@app.post("/init")
async def init_model(request: Request):
    raw = await request.body()
    payload = torch.load(io.BytesIO(zlib.decompress(raw)), map_location="cpu", weights_only=False)
    args = SimpleNamespace(**payload["model_config"])
    lr = float(payload.get("lr", 4e-4))

    model = MidiVision(args)
    if payload.get("model_state_dict") is not None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-2)
    if payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    criterion = MidiVisionLoss().to(DEVICE)
    STATE["model"] = model
    STATE["optimizer"] = optimizer
    STATE["criterion"] = criterion
    STATE["step"] = int(payload.get("optimizer_step", 0))
    STATE["lr"] = lr

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"status": "initialized", "params": params, "step": STATE["step"], "lr": lr, "device": str(DEVICE)}


@app.post("/train_multi_step")
async def train_multi_step(request: Request):
    raw = await request.body()
    batch_size = int(request.headers.get("X-Batch-Size", "1024"))

    header_len = int.from_bytes(raw[:4], byteorder="big")
    header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
    decomp = zlib.decompress(raw[4 + header_len :])
    arr_u8 = np.frombuffer(decomp, dtype=np.uint8).copy().reshape(header["shape"])

    model = STATE["model"]
    optimizer = STATE["optimizer"]
    criterion = STATE["criterion"]
    model.train()

    results = []
    n_total = len(arr_u8)
    for start in range(0, n_total, batch_size):
        sub_u8 = arr_u8[start : start + batch_size]
        x = torch.from_numpy(sub_u8).to(device=DEVICE, dtype=torch.float32, non_blocking=True).div_(254.0)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            recon, _tokens, commit = model(x)
            loss = criterion(recon, x, commit)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        STATE["step"] += 1

        m = dict(criterion.last_metrics)
        m["batch_chunks"] = len(sub_u8)
        m["step"] = STATE["step"]
        results.append(m)

    return {"results": results}


def _run_big_batch_worker(raw_comp: bytes, batch_size: int):
    try:
        header_len = int.from_bytes(raw_comp[:4], byteorder="big")
        header = json.loads(raw_comp[4 : 4 + header_len].decode("utf-8"))
        decomp = zlib.decompress(raw_comp[4 + header_len :])
        arr_u8 = np.frombuffer(decomp, dtype=np.uint8).copy().reshape(header["shape"])
        del raw_comp, decomp

        model = STATE["model"]
        optimizer = STATE["optimizer"]
        criterion = STATE["criterion"]
        model.train()

        n_total = len(arr_u8)
        for start in range(0, n_total, batch_size):
            sub_u8 = arr_u8[start : start + batch_size]
            x = torch.from_numpy(sub_u8).to(device=DEVICE, dtype=torch.float32, non_blocking=True).div_(254.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                recon, _tokens, commit = model(x)
                loss = criterion(recon, x, commit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            STATE["step"] += 1
            m = dict(criterion.last_metrics)
            m["batch_chunks"] = len(sub_u8)
            m["step"] = STATE["step"]
            STATE["job_results"].append(m)
            if STATE["step"] % 25 == 0:
                print(
                    f"[Step {m['step']}] Loss:{m['total']:.4f} | NoteF1:{m['f1']*100:.1f}% | "
                    f"OnsetF1:{m['onset_f1']*100:.1f}% | VelMAE:{m['vel_mae']:.4f}",
                    flush=True,
                )
        STATE["job_done"] = True
    except Exception as e:
        STATE["job_error"] = str(e)
        STATE["job_done"] = True


@app.post("/start_big_batch")
async def start_big_batch(request: Request):
    batch_size = int(request.query_params.get("batch_size", "1024"))
    raw_comp = b"".join(STATE.get("upload_parts", []))
    STATE["upload_parts"] = []
    STATE["job_results"] = []
    STATE["job_done"] = False
    STATE["job_error"] = None
    t = threading.Thread(target=_run_big_batch_worker, args=(raw_comp, batch_size), daemon=True)
    t.start()
    return {"status": "started", "compressed_bytes": len(raw_comp)}


@app.get("/poll_big_batch")
async def poll_big_batch(request: Request):
    cursor = int(request.query_params.get("cursor", "0"))
    res = STATE.get("job_results", [])
    new_items = res[cursor:]
    return {
        "results": new_items,
        "next_cursor": cursor + len(new_items),
        "done": bool(STATE.get("job_done", False)),
        "error": STATE.get("job_error"),
    }


@app.post("/save_colab_ckpt")
async def save_colab_ckpt(request: Request):
    outer_idx = int(request.query_params.get("outer", "0"))
    tokens = int(request.query_params.get("tokens", "0"))
    epoch = int(request.query_params.get("epoch", "0"))
    total_outer = int(request.query_params.get("total_outer", "1581"))
    w_id = request.query_params.get("wandb_id", "7vkpvh37")
    version = request.query_params.get("version", "M2_Compact_G4L2_D128_K2048")
    refresh_buf = int(request.query_params.get("refresh_buffer", "1"))
    model = STATE["model"]
    optimizer = STATE["optimizer"]
    step_now = int(STATE["step"])
    cpu_model_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    colab_path = os.path.join(_get_worker_dir(), f"MORTM5.{version}.checkpoint.pt")
    full_ckpt = {
        "model_state_dict": cpu_model_sd,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": None,
        "epoch": epoch,
        "outer_batch_idx": outer_idx,
        "total_outer_batches": total_outer,
        "optimizer_step": step_now,
        "all_tokens": tokens,
        "last_val_calc_tokens": 0,
        "verification_loss": 0.0,
        "rng_states": None,
        "wandb_run_id": w_id,
        "model_name": "MORTM5",
        "version": version,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    torch.save(full_ckpt, colab_path + ".tmp")
    os.replace(colab_path + ".tmp", colab_path)
    STATE["version"] = version

    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        def _bg_save_to_drive(src_path, model_sd, ver, o_idx, s_now):
            try:
                drive_dir = "/content/drive/MyDrive/MORTM5_checkpoints"
                os.makedirs(drive_dir, exist_ok=True)
                dst_ckpt = os.path.join(drive_dir, f"MORTM5.{ver}.checkpoint.pt")
                shutil.copy2(src_path, dst_ckpt + ".tmp")
                os.replace(dst_ckpt + ".tmp", dst_ckpt)
                dst_pth = os.path.join(drive_dir, f"MORTM5.{ver}_latest.pth")
                torch.save(model_sd, dst_pth + ".tmp")
                os.replace(dst_pth + ".tmp", dst_pth)
                if o_idx % 20 == 0 or o_idx in (215, 220, 256):
                    ms_pth = os.path.join(drive_dir, f"MORTM5.{ver}_outer{o_idx}_step{s_now}.pth")
                    shutil.copy2(dst_pth, ms_pth)
            except Exception as e:
                print(f"[Drive Save Warning] {e}", flush=True)

        threading.Thread(
            target=_bg_save_to_drive,
            args=(colab_path, cpu_model_sd, version, outer_idx, step_now),
            daemon=True,
        ).start()
        STATE["ckpt_meta"] = {
            "drive_saved": True,
            "total_bytes": 0,
            "step": step_now,
            "outer": outer_idx,
            "tokens": tokens,
            "version": version,
        }
        return STATE["ckpt_meta"]

    if refresh_buf or "ckpt_bytes" not in STATE:
        buf = io.BytesIO()
        torch.save(
            {
                "model_state_dict": cpu_model_sd,
                "outer_batch_idx": outer_idx,
                "step": step_now,
                "all_tokens": tokens,
            },
            buf,
        )
        comp = zlib.compress(buf.getvalue(), level=1)
        STATE["ckpt_bytes"] = comp
        STATE["ckpt_meta"] = {
            "drive_saved": False,
            "total_bytes": len(comp),
            "step": step_now,
            "outer": outer_idx,
            "tokens": tokens,
        }
    return STATE["ckpt_meta"]


@app.get("/get_state")
async def get_state():
    model = STATE["model"]
    optimizer = STATE["optimizer"]
    cpu_model_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(
        {
            "model_state_dict": cpu_model_sd,
            "optimizer_state_dict": optimizer.state_dict(),
            "step": STATE["step"],
        },
        buf,
    )
    comp = zlib.compress(buf.getvalue(), level=1)
    return Response(content=comp, media_type="application/octet-stream")


@app.post("/clear_upload")
async def clear_upload():
    STATE["upload_parts"] = []
    return {"status": "cleared"}


@app.post("/append_upload")
async def append_upload(request: Request):
    body = await request.body()
    STATE.setdefault("upload_parts", []).append(body)
    return {"parts": len(STATE["upload_parts"]), "bytes": len(body)}


@app.get("/get_state_raw_chunk")
async def get_state_raw_chunk(request: Request):
    start = int(request.query_params.get("start", "0"))
    end = int(request.query_params.get("end", "2097152"))
    b = STATE.get("ckpt_bytes", b"")[start:end]
    return Response(content=b, media_type="application/octet-stream")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MORTM5 Remote GPU Worker (Colab)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-tunnel", action="store_true", help="Disable Cloudflare tunnel (for local testing)")
    cli_args = parser.parse_args()

    if not cli_args.no_tunnel:
        try:
            from pycloudflared import try_cloudflare

            tunnel_url = try_cloudflare(port=cli_args.port)
            print("\n" + "=" * 72, flush=True)
            print(f"🚀 COLAB REMOTE GPU WORKER READY (mortm v{getattr(mortm, '__version__', '5.0.0b1')})!", flush=True)
            print(f"🔗 Tunnel URL: {tunnel_url.tunnel}", flush=True)
            print("=" * 72 + "\n", flush=True)
        except Exception as e:
            print(f"[Info] Running without pycloudflared ({e})", flush=True)

    uvicorn.run(app, host="0.0.0.0", port=cli_args.port, log_level="warning")

