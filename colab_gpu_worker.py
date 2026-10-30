#!/usr/bin/env python3
"""
colab_gpu_worker.py — Google Colab FastAPI Remote GPU Worker (MORTM v5.0.0b4 Beta)
===================================================================================
モデルのアーキテクチャ (`MidiVision`, `AblationTransformer1`, `AblationTransformer2`,
`MORTM5TrainSet`, `HybridMuonOptimizer`, `MORTM5MemmapDataset`) はすべて pip パッケージ
`mortm` (`pip install --pre -U mortm`) から直接呼び出します。

【Google Colab での起動手順】
1. ランタイムを GPU (RTX PRO 6000 / A100 / L4) に設定し、以下を実行:
   !pip install -q -U "git+https://github.com/Ayato964/MORTM.git@mortm5-tokenizer-mac" fastapi uvicorn pycloudflared
   # (PyPI 公開版を使う場合: !pip install -q --pre -U "mortm==5.0.0b4" vector-quantize-pytorch pretty_midi mido fastapi uvicorn pycloudflared)
2. Google Drive をマウント (トークンデータやチェックポイントを Drive に置く場合):
   from google.colab import drive
   drive.mount('/content/drive')
3. サーバースクリプトを起動して Cloudflare Tunnel URL を取得:
   !wget -q -O colab_gpu_worker.py https://raw.githubusercontent.com/Ayato964/MORTM/mortm5-tokenizer-mac/colab_gpu_worker.py
   !nohup python3 -u colab_gpu_worker.py > worker.log 2>&1 &
   !sleep 5 && cat worker.log
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import datetime
import io
import json
import math
import os
import shutil
import tempfile
import threading
import time
import zlib
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
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
    resolve_ablation_subdir,
)
from mortm.train.train import _compute_exact_epoch_optimizer_steps

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
# 3. MORTM5 Autoregressive Transformer (Ablation & SyMIc Main) Endpoints
# ============================================================
_DRIVE_SAVE_LOCK = threading.Lock()
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mortm5_prefetch")


def _resolve_ablation_subdir(model_name: str = "", ablation_id: str = "") -> str:
    """パッケージ `mortm.resolve_ablation_subdir` へ委譲し、#1-#8 および任意のカスタム実験IDを解決する。"""
    return resolve_ablation_subdir(model_name=model_name, ablation_id=ablation_id)


def _detach_state_to_cpu(obj: Any) -> Any:
    """
    Optimizer / Scheduler の state_dict 内に含まれる GPU Tensor をすべて再帰的に CPU へ退避する。
    `copy.deepcopy` による GPU VRAM の倍増スパイクと、バックグラウンド保存スレッドからの CUDA ストリーム競合を完全に防ぐ。
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(device="cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _detach_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_state_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


class LocalTokenStagingService:
    """
    Google Drive FUSE 上のトークンファイル (`*.bin`, `*.npy`, `*.json`) を Colab ローカル NVMe SSD へ
    ストリーム転送・キャッシュし、数百万回のランダムシークによる I/O ボトルネックを解消するサービス。
    """

    _STAGE_EXTENSIONS: Tuple[str, ...] = (".bin", ".npy", ".json")
    _SAFETY_MARGIN_BYTES: int = 8 * (1024 ** 3)  # 8 GiB safety headroom for checkpoints
    _CHUNK_BYTES: int = 16 * 1024 * 1024         # 16 MiB sequential read/write buffer

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._done = True
        self._error: Optional[str] = None
        self._src_dir: str = ""
        self._local_dir: str = ""
        self._copied_bytes: int = 0
        self._total_bytes: int = 0
        self._started_at: float = 0.0

    @staticmethod
    def _candidate_cache_roots() -> List[str]:
        explicit = os.environ.get("MORTM5_LOCAL_CACHE_DIR", "").strip()
        if explicit:
            return [explicit]
        roots = [os.path.join(tempfile.gettempdir(), "mortm5_token_cache")]
        if os.path.isdir("/content"):
            roots.append("/content/mortm5_token_cache")
        return roots

    @classmethod
    def _list_stageable_files(cls, src_dir: str) -> List[Tuple[str, int]]:
        entries: List[Tuple[str, int]] = []
        if not os.path.isdir(src_dir):
            return entries
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(cls._STAGE_EXTENSIONS):
                continue
            fpath = os.path.join(src_dir, fname)
            if os.path.isfile(fpath):
                entries.append((fname, int(os.path.getsize(fpath))))
        return entries

    @staticmethod
    def _subdir_key(src_dir: str) -> str:
        norm = os.path.normpath(os.path.abspath(src_dir))
        base = os.path.basename(norm) or "tokens"
        parent = os.path.basename(os.path.dirname(norm))
        return f"{parent}_{base}" if parent and parent not in ("", "/", "MyDrive") else base

    def should_stage(self, src_dir: str, enabled: bool) -> bool:
        if not enabled or not src_dir:
            return False
        if os.environ.get("MORTM5_FORCE_STAGE") == "1":
            return True
        norm = os.path.abspath(src_dir)
        return norm.startswith("/content/drive/")

    def resolve_cached_dir(self, src_dir: str) -> Optional[str]:
        files = self._list_stageable_files(src_dir)
        if not files:
            return None
        sub = self._subdir_key(src_dir)
        for root in self._candidate_cache_roots():
            cand = os.path.join(root, sub)
            if not os.path.isdir(cand):
                continue
            if all(
                os.path.isfile(os.path.join(cand, fn))
                and os.path.getsize(os.path.join(cand, fn)) == sz
                for fn, sz in files
            ):
                return cand
        return None

    def _select_target_dir(self, src_dir: str, files: List[Tuple[str, int]]) -> Tuple[Optional[str], int]:
        sub = self._subdir_key(src_dir)
        for root in self._candidate_cache_roots():
            try:
                os.makedirs(root, exist_ok=True)
                cand = os.path.join(root, sub)
                os.makedirs(cand, exist_ok=True)
                needed = sum(
                    sz
                    for fn, sz in files
                    if not (
                        os.path.isfile(os.path.join(cand, fn))
                        and os.path.getsize(os.path.join(cand, fn)) == sz
                    )
                )
                free_bytes = int(shutil.disk_usage(cand).free)
                if free_bytes >= needed + self._SAFETY_MARGIN_BYTES:
                    return cand, needed
            except OSError:
                continue
        return None, 0

    def stage_sync(self, src_dir: str) -> str:
        cached = self.resolve_cached_dir(src_dir)
        if cached is not None:
            return cached
        files = self._list_stageable_files(src_dir)
        if not files:
            return src_dir
        target_dir, _ = self._select_target_dir(src_dir, files)
        if target_dir is None:
            print(f"[TokenStage] Insufficient local SSD space -> falling back to direct FUSE read: {src_dir}", flush=True)
            return src_dir

        total_bytes = sum(sz for _, sz in files)
        copied = 0
        t0 = time.time()
        for fname, expected_sz in files:
            src_p = os.path.join(src_dir, fname)
            dst_p = os.path.join(target_dir, fname)
            if os.path.isfile(dst_p) and os.path.getsize(dst_p) == expected_sz:
                copied += expected_sz
                with self._lock:
                    self._copied_bytes = copied
                continue
            tmp_p = dst_p + ".part"
            with open(src_p, "rb") as fsrc, open(tmp_p, "wb") as fdst:
                while True:
                    buf = fsrc.read(self._CHUNK_BYTES)
                    if not buf:
                        break
                    fdst.write(buf)
                    copied += len(buf)
                    with self._lock:
                        self._copied_bytes = copied
            os.replace(tmp_p, dst_p)
        dt = max(1e-3, time.time() - t0)
        print(
            f"[TokenStage] Staged {total_bytes / (1024**3):.2f} GiB to {target_dir} in {dt:.1f}s "
            f"({(total_bytes / (1024**2)) / dt:.1f} MiB/s)",
            flush=True,
        )
        return target_dir

    def start_async(self, src_dir: str) -> Dict[str, Any]:
        with self._lock:
            if self._running and self._src_dir == src_dir:
                return self.get_status_unlocked()

        cached = self.resolve_cached_dir(src_dir)
        files = self._list_stageable_files(src_dir)
        total_bytes = sum(sz for _, sz in files)
        if cached is not None:
            with self._lock:
                self._running = False
                self._done = True
                self._error = None
                self._src_dir = src_dir
                self._local_dir = cached
                self._copied_bytes = total_bytes
                self._total_bytes = total_bytes
                self._started_at = time.time()
                return self.get_status_unlocked()

        with self._lock:
            self._running = True
            self._done = False
            self._error = None
            self._src_dir = src_dir
            self._local_dir = src_dir
            self._copied_bytes = 0
            self._total_bytes = total_bytes
            self._started_at = time.time()

        def _bg():
            try:
                resolved = self.stage_sync(src_dir)
                with self._lock:
                    self._local_dir = resolved
                    self._copied_bytes = self._total_bytes
                    self._running = False
                    self._done = True
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                    self._local_dir = src_dir
                    self._running = False
                    self._done = True

        threading.Thread(target=_bg, daemon=True, name="mortm5_stage").start()
        return self.get_status()

    def get_status_unlocked(self) -> Dict[str, Any]:
        elapsed = max(1e-3, time.time() - self._started_at) if self._started_at > 0 else 0.0
        mb_s = ((self._copied_bytes / (1024 ** 2)) / elapsed) if elapsed > 0 else 0.0
        return {
            "running": self._running,
            "done": self._done,
            "error": self._error,
            "src_dir": self._src_dir,
            "local_dir": self._local_dir,
            "copied_bytes": int(self._copied_bytes),
            "total_bytes": int(self._total_bytes),
            "mb_per_sec": round(mb_s, 1),
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self.get_status_unlocked()


_STAGING_SERVICE = LocalTokenStagingService()


class PrefetchedMicroBatchLoader:
    """
    常駐スレッドプール (`_PREFETCH_POOL`) を用いて Memmap からマイクロバッチ単位で遅延読み込み・Collate・PinMemory を先読みするイテレータ。
    `DataLoader(num_workers>0)` のような Outer Batch 毎の `os.fork()` コストと、16,384 件一括ロードによる GPU 待機を完全に排除する。
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        batch_size: int,
        collate_fn: Callable[[List[torch.Tensor]], Any],
        shuffle: bool = True,
        pin_memory: bool = False,
        prefetch_batches: int = 8,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.collate_fn = collate_fn
        self.pin_memory = bool(pin_memory)
        self.prefetch_batches = max(1, int(prefetch_batches))

        idx_arr = np.asarray(indices, dtype=np.int64)
        if shuffle and len(idx_arr) > 1:
            idx_arr = idx_arr[np.random.permutation(len(idx_arr))]
        self._batch_slices: List[np.ndarray] = [
            idx_arr[pos : pos + self.batch_size]
            for pos in range(0, len(idx_arr), self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self._batch_slices)

    def _build_one(self, batch_idx_arr: np.ndarray) -> Any:
        items = [self.dataset[int(i)] for i in batch_idx_arr]
        packed = self.collate_fn(items)
        if not self.pin_memory:
            return packed
        if isinstance(packed, tuple):
            return tuple(t.pin_memory() if isinstance(t, torch.Tensor) else t for t in packed)
        if isinstance(packed, torch.Tensor):
            return packed.pin_memory()
        return packed

    def __iter__(self) -> Iterator[Any]:
        total = len(self._batch_slices)
        if total == 0:
            return
        queue: deque[Future] = deque()
        next_submit = 0
        while next_submit < min(total, self.prefetch_batches):
            queue.append(_PREFETCH_POOL.submit(self._build_one, self._batch_slices[next_submit]))
            next_submit += 1

        while queue:
            fut = queue.popleft()
            if next_submit < total:
                queue.append(_PREFETCH_POOL.submit(self._build_one, self._batch_slices[next_submit]))
                next_submit += 1
            yield fut.result()


class ModelExecutionAdapter:
    """
    `trainer.model` 自体（およびその `state_dict()` キー構造）を一切汚染せずに、
    任意の `torch.compile` 実行とエラー時の自動 Eager フォールバックを透過的に提供するアダプタ。
    """

    def __init__(self, base_model: nn.Module, enable_compile: bool = False) -> None:
        self.base_model = base_model
        self.compiled_model: Optional[nn.Module] = None
        if enable_compile and hasattr(torch, "compile"):
            try:
                self.compiled_model = torch.compile(base_model, dynamic=True)
            except Exception as exc:
                print(f"[CompileFallback] torch.compile init skipped ({exc}); using eager mode.", flush=True)
                self.compiled_model = None

    @property
    def is_compiled(self) -> bool:
        return self.compiled_model is not None

    def forward_epoch_fc(self, trainer: MORTM5TrainSet, pack: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.compiled_model is not None:
            try:
                return trainer.epoch_fc(self.compiled_model, pack, None)
            except Exception as exc:
                print(f"[CompileFallback] Runtime fallback to eager model ({exc}).", flush=True)
                self.compiled_model = None
        return trainer.epoch_fc(self.base_model, pack, None)


def _get_drive_ckpt_dir(ablation_subdir: str) -> str:
    base_dir = os.environ.get("MORTM5_DRIVE_CKPT_DIR", "/content/drive/MyDrive/MORTM5_checkpoints")
    return os.path.join(base_dir, ablation_subdir) if ablation_subdir else base_dir


def _get_worker_dir() -> str:
    """Colab 上では `/content`、ローカルテスト環境では `MORTM5_WORKER_DIR` または一時ディレクトリを返す。"""
    work_dir = os.environ.get(
        "MORTM5_WORKER_DIR",
        "/content" if os.path.isdir("/content") else os.path.join(tempfile.gettempdir(), "mortm5_worker"),
    )
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


@app.post("/mortm5/start_stage")
async def mortm5_start_stage(request: Request):
    """Google Drive 上のトークンディレクトリをローカル NVMe SSD へ非同期ステージング開始する。"""
    payload = await request.json()
    colab_split_dir = str(payload.get("colab_split_dir") or "").strip()
    if not colab_split_dir or not os.path.isdir(colab_split_dir):
        return Response(
            content=json.dumps({"error": f"colab_split_dir not found: {colab_split_dir}"}),
            status_code=400,
            media_type="application/json",
        )
    if not _STAGING_SERVICE.should_stage(colab_split_dir, enabled=True):
        return {
            "running": False,
            "done": True,
            "error": None,
            "src_dir": colab_split_dir,
            "local_dir": colab_split_dir,
            "copied_bytes": 0,
            "total_bytes": 0,
            "mb_per_sec": 0.0,
        }
    return _STAGING_SERVICE.start_async(colab_split_dir)


@app.get("/mortm5/poll_stage")
async def mortm5_poll_stage():
    """非同期トークンステージングの進捗状況を返す。"""
    return _STAGING_SERVICE.get_status()


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
        STATE["exec_adapter"] = None
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

    colab_split_dir: Optional[str] = payload.get("colab_split_dir")
    stage_to_local: bool = bool(payload.get("stage_to_local", True))
    use_torch_compile: bool = bool(payload.get("use_torch_compile", False))
    stride: Optional[int] = payload.get("stride")
    val_stride: Optional[int] = payload.get("val_stride")

    effective_split_dir: Optional[str] = colab_split_dir
    train_ds: Optional[MORTM5MemmapDataset] = None
    val_ds: Optional[MORTM5MemmapDataset] = None
    if colab_split_dir:
        if not os.path.isdir(colab_split_dir):
            return Response(
                content=json.dumps({"error": f"colab_split_dir not found: {colab_split_dir}"}),
                status_code=400,
                media_type="application/json",
            )
        cached_local = _STAGING_SERVICE.resolve_cached_dir(colab_split_dir)
        if cached_local is not None:
            effective_split_dir = cached_local
        elif _STAGING_SERVICE.should_stage(colab_split_dir, enabled=stage_to_local):
            effective_split_dir = _STAGING_SERVICE.stage_sync(colab_split_dir)

        assert effective_split_dir is not None
        train_ds = MORTM5MemmapDataset.from_split_dir(
            token_dir=effective_split_dir, split="train", args=m_args, stride=stride
        )
        val_tokens_path = os.path.join(effective_split_dir, "val_tokens.bin")
        val_complete_path = os.path.join(effective_split_dir, "val_tokens_complete.bin")
        if os.path.isfile(val_tokens_path) or os.path.isfile(val_complete_path):
            val_ds = MORTM5MemmapDataset.from_split_dir(
                token_dir=effective_split_dir, split="val", args=m_args, stride=val_stride
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

    # チェックポイントの復元 (指定パス または Google Drive サブディレクトリ内チェックポイント)
    resumed_from: Optional[str] = None
    resumed_big_bs: Optional[int] = None
    resume_ckpt_path: Optional[str] = payload.get("resume_checkpoint_path")
    version_str = str(payload.get("version") or f"{m_args.name}_muon_cos").strip()
    if resume_ckpt_path and not os.path.isfile(resume_ckpt_path):
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
        saved_t_cfg = ckpt_data.get("train_config")
        if isinstance(saved_t_cfg, dict) and saved_t_cfg.get("big_batch_size") is not None:
            resumed_big_bs = int(saved_t_cfg["big_batch_size"])
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
    exec_adapter = ModelExecutionAdapter(trainer.model, enable_compile=use_torch_compile)
    STATE["trainer"] = trainer
    STATE["exec_adapter"] = exec_adapter
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
        "effective_split_dir": effective_split_dir,
        "compiled": exec_adapter.is_compiled,
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
        "resumed_big_batch_size": resumed_big_bs,
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
    Colab GPU 上で 1 Outer Batch を実行するバックグラウンドワーカー。
    - PrefetchedMicroBatchLoader による遅延非同期先読み (fork なし・初期レイテンシゼロ)
    - マイクロバッチごとの `.item()` GPU-CPU 同期を全廃し、オプティマイザ更新時にのみ1回読み出すゼロ同期ループ
    """
    try:
        trainer: MORTM5TrainSet = STATE["trainer"]
        exec_adapter: Optional[ModelExecutionAdapter] = STATE.get("exec_adapter")
        t_args: TrainArgs = STATE["t_args"]
        collate_fn: MORTM5Collate = STATE["collate_fn"]
        model = trainer.model
        optimizer = trainer.optimizer
        scheduler = trainer.scheduler
        model.train()

        batch_size = max(1, int(t_args.batch_size))
        accum_steps = max(1, int(t_args.accumulation_steps))
        amp_dtype = torch.bfloat16 if DEVICE.type in ("cuda", "cpu") else torch.float16

        # データソースの構築 (Mode A: Colab ローカル SSD 上の MemmapDataset / Mode B: Mac からアップロードされたトークン配列)
        if window_indices is not None and STATE.get("train_ds") is not None:
            inner_loader = PrefetchedMicroBatchLoader(
                dataset=STATE["train_ds"],
                indices=window_indices,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                pin_memory=(DEVICE.type == "cuda"),
                prefetch_batches=8,
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
        accum_loss_gpu = torch.zeros((), device=DEVICE, dtype=torch.float32)
        step_start_tokens_gpu = trainer.all_tokens.clone()
        accum_micro_count = 0

        for b_idx, pack in enumerate(batches_iter):
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype):
                if exec_adapter is not None:
                    logits_flat, target_flat = exec_adapter.forward_epoch_fc(trainer, pack)
                else:
                    logits_flat, target_flat = trainer.epoch_fc(model, pack, None)

            is_last_micro = (b_idx + 1) == total_micro_batches
            will_step = ((accum_micro_count + 1) >= accum_steps) or is_last_micro
            effective_div = accum_steps if (accum_micro_count + 1) >= accum_steps else (accum_micro_count + 1)

            with torch.autocast(device_type=DEVICE.type, enabled=False):
                valid_tgt = (target_flat != PAD_TOKEN_ID).sum()
                sum_loss = F.cross_entropy(
                    logits_flat.float(),
                    target_flat,
                    ignore_index=PAD_TOKEN_ID,
                    reduction="sum",
                )
                raw_loss = sum_loss / valid_tgt.clamp(min=1).to(dtype=torch.float32)

            scaled_loss = raw_loss / float(effective_div if will_step and accum_micro_count == 0 else accum_steps)
            if will_step and (accum_micro_count + 1) < accum_steps and accum_micro_count > 0:
                scaled_loss = raw_loss / float(accum_steps)

            scaled_loss.backward()
            if will_step and (accum_micro_count + 1) < accum_steps and accum_micro_count > 0:
                scale_fix = float(accum_steps) / float(accum_micro_count + 1)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale_fix)

            accum_loss_gpu += raw_loss.detach()
            accum_micro_count += 1

            if will_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                STATE["step"] += 1
                # Optimizer Step 境界で 1 回だけまとめて CPU へ同期読み出しする
                step_toks_gpu = (trainer.all_tokens - step_start_tokens_gpu).clamp(min=0)
                loss_sum_val, step_toks_val, total_toks_val = (
                    accum_loss_gpu.item(),
                    int(step_toks_gpu.item()),
                    int(trainer.all_tokens.item()),
                )
                mean_step_loss = float(loss_sum_val) / max(1, accum_micro_count)
                ppl = math.exp(min(20.0, mean_step_loss))
                lr_muon, lr_adamw = _extract_lrs(optimizer)

                step_metric = {
                    "step": int(STATE["step"]),
                    "outer_idx": int(outer_idx),
                    "epoch": int(epoch),
                    "loss": mean_step_loss,
                    "ppl": ppl,
                    "lr_muon": lr_muon,
                    "lr_adamw": lr_adamw,
                    "step_tokens": step_toks_val,
                    "all_tokens": total_toks_val,
                }
                STATE["mortm5_job_results"].append(step_metric)
                if STATE["step"] % 20 == 0:
                    print(
                        f"[MORTM5 Step {STATE['step']}/{STATE['total_steps']}] "
                        f"Loss:{mean_step_loss:.4f} | PPL:{ppl:.2f} | "
                        f"LR(Muon):{lr_muon:.2e} | Tokens:{total_toks_val/1e6:.2f}M",
                        flush=True,
                    )
                accum_loss_gpu.zero_()
                step_start_tokens_gpu.copy_(trainer.all_tokens)
                accum_micro_count = 0

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
    """Colab GPU 上で検証セット (`val_ds` またはアップロードされたトークン) の CrossEntropy 損失と PPL をゼロ同期ループで計算する。"""
    trainer: MORTM5TrainSet = STATE["trainer"]
    exec_adapter: Optional[ModelExecutionAdapter] = STATE.get("exec_adapter")
    t_args: TrainArgs = STATE["t_args"]
    collate_fn: MORTM5Collate = STATE["collate_fn"]
    model = trainer.model

    max_batches = int(request.query_params.get("max_batches", "0"))
    amp_dtype = torch.bfloat16 if DEVICE.type in ("cuda", "cpu") else torch.float16
    bs = max(1, int(t_args.batch_size))

    val_ds: Optional[MORTM5MemmapDataset] = STATE.get("val_ds")
    if val_ds is not None:
        max_items = (max_batches * bs) if max_batches > 0 else len(val_ds)
        val_indices = range(min(len(val_ds), max_items))
        batches_iter: Any = PrefetchedMicroBatchLoader(
            dataset=val_ds,
            indices=val_indices,
            batch_size=bs,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),
            prefetch_batches=8,
        )
    elif STATE.get("upload_parts"):
        raw_comp = b"".join(STATE["upload_parts"])
        STATE["upload_parts"] = []
        header_len = int.from_bytes(raw_comp[:4], byteorder="big")
        header = json.loads(raw_comp[4 : 4 + header_len].decode("utf-8"))
        decomp = zlib.decompress(raw_comp[4 + header_len :])
        np_dtype = np.dtype(header.get("dtype", "int32"))
        arr = np.frombuffer(decomp, dtype=np_dtype).copy().reshape(header["shape"])
        batches_iter = [torch.from_numpy(arr[i : i + bs]).to(dtype=torch.long) for i in range(0, len(arr), bs)]
    else:
        return {"val_loss": 0.0, "val_ppl": 1.0, "val_tokens": 0, "step": int(STATE["step"])}

    was_training = model.training
    model.eval()
    acc_dtype = torch.float64 if DEVICE.type in ("cuda", "cpu") else torch.float32
    loss_sum_gpu = torch.zeros((), device=DEVICE, dtype=acc_dtype)
    n_valid_batches_gpu = torch.zeros((), device=DEVICE, dtype=torch.long)
    val_tokens_gpu = torch.zeros((), device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        for b_idx, pack in enumerate(batches_iter):
            if 0 < max_batches <= b_idx:
                break
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype):
                if exec_adapter is not None:
                    logits_flat, target_flat = exec_adapter.forward_epoch_fc(trainer, pack)
                else:
                    logits_flat, target_flat = trainer.epoch_fc(model, pack, None)
            valid_cnt = (target_flat != PAD_TOKEN_ID).sum()
            with torch.autocast(device_type=DEVICE.type, enabled=False):
                sum_loss = F.cross_entropy(
                    logits_flat.float(),
                    target_flat,
                    ignore_index=PAD_TOKEN_ID,
                    reduction="sum",
                )
                batch_mean_loss = (sum_loss / valid_cnt.clamp(min=1).to(dtype=torch.float32)).to(dtype=acc_dtype)
            has_valid = valid_cnt > 0
            loss_sum_gpu += torch.where(has_valid, batch_mean_loss, torch.zeros_like(batch_mean_loss))
            n_valid_batches_gpu += has_valid.to(dtype=torch.long)
            val_tokens_gpu += valid_cnt

    if was_training:
        model.train()

    n_batches = int(n_valid_batches_gpu.item())
    val_tokens = int(val_tokens_gpu.item())
    val_loss = float(loss_sum_gpu.item()) / max(1, n_batches)
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
    MORTM5 チェックポイントを Google Drive のアブレーション/実験サブディレクトリに直接保存する。
    Optimizer state も事前に CPU メモリへデタッチしてクローンするため、GPU VRAM スパイクや CUDA ストリーム競合が発生しない。
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
    cpu_model_sd = {k: v.detach().to(device="cpu", copy=True) for k, v in trainer.model.state_dict().items()}
    cpu_opt_sd = _detach_state_to_cpu(trainer.optimizer.state_dict())
    cpu_sched_sd = _detach_state_to_cpu(trainer.scheduler.state_dict()) if trainer.scheduler is not None else None

    full_ckpt = {
        "model_state_dict": cpu_model_sd,
        "optimizer_state_dict": cpu_opt_sd,
        "scheduler_state_dict": cpu_sched_sd,
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
            import tempfile
            with _DRIVE_SAVE_LOCK:
                os.makedirs(target_dir, exist_ok=True)
                dst_pth = os.path.join(target_dir, f"MORTM5.{ver}_latest.pth")
                dst_ckpt = os.path.join(target_dir, f"MORTM5.{ver}.checkpoint.pt")

                # 1. 最優先: 推論・評価用の軽量モデル重み (_latest.pth, ~374MB) を最初に独立保存
                tmp_local_pth = os.path.join(tempfile.gettempdir(), f"mortm5_{ver}_latest_{s_now}.pth.tmp")
                try:
                    torch.save(model_sd, tmp_local_pth)
                    shutil.copy2(tmp_local_pth, dst_pth)
                    if m_tag:
                        ms_pth = os.path.join(target_dir, f"MORTM5.{ver}_{m_tag}_step{s_now}.pth")
                        try:
                            shutil.copy2(dst_pth, ms_pth)
                        except Exception as me:
                            print(f"[Drive Milestone Save Warning ({target_dir})] {me}", flush=True)
                except Exception as e:
                    print(f"[Drive Model Save Error ({target_dir})] {e}", flush=True)
                    # Drive 書き込み失敗時は /content に退避してモデル消失を防止
                    fallback_pth = os.path.join(_get_worker_dir(), f"MORTM5.{ver}_latest.pth")
                    try:
                        shutil.copy2(tmp_local_pth, fallback_pth)
                    except Exception:
                        pass
                finally:
                    if os.path.exists(tmp_local_pth):
                        try:
                            os.remove(tmp_local_pth)
                        except OSError:
                            pass

                # 2. 再開用フルチェックポイント (.checkpoint.pt, ~1.15GB) を別ブロックで保存
                tmp_local_ckpt = os.path.join(tempfile.gettempdir(), f"mortm5_{ver}_ckpt_{s_now}.pt.tmp")
                try:
                    torch.save(ckpt_obj, tmp_local_ckpt)
                    shutil.copy2(tmp_local_ckpt, dst_ckpt)
                except Exception as e:
                    print(f"[Drive Full-Ckpt Save Warning ({target_dir})] {e} -> retrying without optimizer state", flush=True)
                    try:
                        lite_ckpt = dict(ckpt_obj)
                        lite_ckpt["optimizer_state_dict"] = None
                        torch.save(lite_ckpt, tmp_local_ckpt)
                        shutil.copy2(tmp_local_ckpt, dst_ckpt)
                    except Exception as e2:
                        print(f"[Drive Lite-Ckpt Save Error ({target_dir})] {e2}", flush=True)
                finally:
                    if os.path.exists(tmp_local_ckpt):
                        try:
                            os.remove(tmp_local_ckpt)
                        except OSError:
                            pass

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
        ablation_subdir = str(STATE.get("ablation_subdir", "") or "")
        def _bg_save_to_drive(src_path, model_sd, ver, o_idx, s_now, sub_dir):
            drive_dir = _get_drive_ckpt_dir(sub_dir) if sub_dir else "/content/drive/MyDrive/MORTM5_checkpoints"
            os.makedirs(drive_dir, exist_ok=True)
            dst_pth = os.path.join(drive_dir, f"MORTM5.{ver}_latest.pth")
            dst_ckpt = os.path.join(drive_dir, f"MORTM5.{ver}.checkpoint.pt")
            try:
                tmp_local_pth = src_path + ".latest.pth.tmp"
                torch.save(model_sd, tmp_local_pth)
                shutil.copy2(tmp_local_pth, dst_pth)
                os.remove(tmp_local_pth)
            except Exception as e:
                print(f"[Drive Model Save Warning] {e}", flush=True)
            try:
                shutil.copy2(src_path, dst_ckpt)
            except Exception as e:
                print(f"[Drive Ckpt Save Warning] {e}", flush=True)

        threading.Thread(
            target=_bg_save_to_drive,
            args=(colab_path, cpu_model_sd, version, outer_idx, step_now, ablation_subdir),
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

