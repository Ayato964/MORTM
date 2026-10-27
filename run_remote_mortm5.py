#!/usr/bin/env python3
"""
run_remote_mortm5.py — Mac <-> Google Colab ハイブリッド MORTM5 アブレーション学習コントローラー
=================================================================================================
Colab 側には `pip install --pre -U mortm` (v5.0.0b1) でインストールされた `mortm` パッケージの
アーキテクチャ (`AblationTransformer1` / `AblationTransformer2` / `MORTM5TrainSet` / `HybridMuonOptimizer`)
を呼び出させ、Mac 側から学習全体のオーケストレーション・W&B ロギング・検証・チェックポイント管理を行います。

【対応する 2 つのハイブリッドデータモード】
1. Colab/Drive 直読モード (`--colab-split-dir /content/drive/MyDrive/MORTM5_tokens/m4_g4l4`):
   - トークン化済みデータ (`train_tokens.bin`, `train_offsets.npy`, `train_slots.npy`) が Google Drive 上にある場合、
     Colab ローカル SSD (`/content/mortm5_tokens_cache`) に自動ステージングし、通信量ゼロで高速学習します。
2. Mac ストリーミングモード (`--mac-split-dir out/midivision_tokens/m4_g4l4`):
   - トークン化済みデータが Mac ローカルにある場合、Mac 側で `MORTM5MemmapDataset` から Outer Batch を切り出し、
     zlib 圧縮して Colab GPU ワーカーへストリーミング送信します。
"""

import argparse
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from mortm import (
    MORTM5Args,
    MORTM5_ABLATION_PRESETS,
    TrainArgs,
    MORTM5MemmapDataset,
    MORTM5Collate,
    PAD_TOKEN_ID,
)
from mortm.train.train import _compute_exact_epoch_optimizer_steps

try:
    import wandb
except ImportError:
    wandb = None


def _pack_token_tensor_zlib(batch_tensor: torch.Tensor) -> bytes:
    """トークン ID テンソル [N, S] または [N, S, G] を int32 + zlib で高速シリアライズする。"""
    arr = batch_tensor.detach().cpu().numpy().astype(np.int32, copy=False)
    raw_bytes = np.ascontiguousarray(arr).tobytes()
    comp = zlib.compress(raw_bytes, level=1)
    header = json.dumps({"shape": list(arr.shape), "dtype": "int32"}).encode("utf-8")
    return len(header).to_bytes(4, byteorder="big") + header + comp


class RemoteColabMORTM5Client:
    """Colab 上の `colab_gpu_worker.py` (`/mortm5/*`) と通信する HTTP クライアント。"""

    def __init__(self, worker_url: str, timeout: float = 600.0):
        self.base_url = worker_url.rstrip("/")
        self.timeout = timeout

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        retries: int = 8,
    ) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        headers = {
            "User-Agent": "MORTM5-Hybrid-Controller/5.0.0b1",
            "Content-Type": "application/json",
        }
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}{path}",
                    data=body if method.upper() == "POST" else None,
                    headers=headers,
                    method=method.upper(),
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="replace")
                if attempt + 1 >= retries:
                    raise RuntimeError(f"HTTP {he.code} on {path}: {err_body}") from he
                time.sleep(min(15.0, 2.0 * (attempt + 1)))
            except Exception as e:
                if attempt + 1 >= retries:
                    raise
                print(f"\n⚠️ [{path} リトライ {attempt+1}/{retries}] {e}", flush=True)
                time.sleep(min(15.0, 2.0 * (attempt + 1)))
        raise RuntimeError(f"Unreachable endpoint: {path}")

    def _post_bytes(self, path: str, data: bytes, retries: int = 8) -> Dict[str, Any]:
        headers = {
            "User-Agent": "MORTM5-Hybrid-Controller/5.0.0b1",
            "Content-Type": "application/octet-stream",
        }
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}{path}",
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                if attempt + 1 >= retries:
                    raise
                print(f"\n⚠️ [{path} Upload リトライ {attempt+1}/{retries}] {e}", flush=True)
                time.sleep(min(15.0, 2.0 * (attempt + 1)))
        raise RuntimeError(f"Upload failed: {path}")

    def health(self) -> Dict[str, Any]:
        return self._request_json("GET", "/health")

    def init_mortm5(self, init_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request_json("POST", "/mortm5/init", payload=init_payload)

    def upload_token_batch(self, batch_tensor: torch.Tensor, chunk_bytes: int = 2 * 1024 * 1024) -> int:
        packed = _pack_token_tensor_zlib(batch_tensor)
        self._request_json("POST", "/clear_upload", payload={})
        for offset in range(0, len(packed), chunk_bytes):
            self._post_bytes("/append_upload", packed[offset : offset + chunk_bytes])
        return len(packed)

    def start_outer_batch(
        self,
        outer_idx: int,
        epoch: int,
        window_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"outer_idx": int(outer_idx), "epoch": int(epoch)}
        if window_indices is not None:
            payload["window_indices"] = window_indices
        return self._request_json("POST", "/mortm5/start_outer_batch", payload=payload)

    def poll_outer_batch(self, cursor: int) -> Dict[str, Any]:
        return self._request_json("GET", f"/mortm5/poll_outer_batch?cursor={cursor}")

    def validate(self, max_batches: int = 0) -> Dict[str, Any]:
        return self._request_json("POST", f"/mortm5/validate?max_batches={max_batches}", payload={})

    def save_checkpoint(
        self,
        outer_idx: int,
        epoch: int,
        total_outer: int,
        val_loss: float,
        version: str,
        ablation_id: str = "",
        wandb_id: str = "",
        prepare_download: bool = False,
        milestone: str = "",
        sync_drive: bool = False,
    ) -> Dict[str, Any]:
        q = urllib.parse.urlencode({
            "outer": outer_idx,
            "epoch": epoch,
            "total_outer": total_outer,
            "val_loss": f"{val_loss:.6f}",
            "version": version,
            "ablation_id": ablation_id,
            "wandb_id": wandb_id,
            "prepare_download": 1 if prepare_download else 0,
            "milestone": milestone,
            "sync_drive": 1 if sync_drive else 0,
        })
        return self._request_json("POST", f"/mortm5/save_ckpt?{q}", payload={})

    def download_prepared_checkpoint(self, total_bytes: int, chunk_bytes: int = 2 * 1024 * 1024, retries: int = 8) -> Dict[str, Any]:
        parts: List[bytes] = []
        for start in range(0, total_bytes, chunk_bytes):
            end = min(total_bytes, start + chunk_bytes)
            expected_len = end - start
            chunk_data: Optional[bytes] = None
            for attempt in range(retries):
                try:
                    req = urllib.request.Request(
                        f"{self.base_url}/get_state_raw_chunk?start={start}&end={end}",
                        headers={"User-Agent": "MORTM5-Hybrid-Controller/5.0.0b2"},
                        method="GET",
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        data = resp.read()
                    if len(data) != expected_len:
                        raise IOError(f"Short read on chunk {start}..{end}: got {len(data)}, expected {expected_len}")
                    chunk_data = data
                    break
                except Exception as e:
                    if attempt + 1 >= retries:
                        raise
                    print(f"\n⚠️ [Download chunk {start}..{end} リトライ {attempt+1}/{retries}] {e}", flush=True)
                    time.sleep(min(15.0, 2.0 * (attempt + 1)))
            assert chunk_data is not None
            parts.append(chunk_data)
        decomp = zlib.decompress(b"".join(parts))
        return torch.load(io.BytesIO(decomp), map_location="cpu", weights_only=False)


_ABLATION_SUBDIR_MAP: Dict[str, str] = {
    "#1": "1", "1": "1", "m3_flat_shared": "1", "mortm5_m3_flat_shared": "1",
    "#2": "2", "2": "2", "m3_flat_separated": "2", "mortm5_m3_flat_separated": "2",
    "#3": "3", "3": "3", "m3_grouped_shared": "3", "mortm5_m3_grouped_shared": "3",
    "#4": "4", "4": "4", "m3_grouped_separated": "4", "mortm5_m3_grouped_separated": "4",
    "#5": "5", "5": "5", "m4_flat_shared": "5", "mortm5_m4_flat_shared": "5",
    "#6": "6", "6": "6", "m4_flat_separated": "6", "mortm5_m4_flat_separated": "6",
    "#7": "7", "7": "7", "m4_grouped_shared": "7", "mortm5_m4_grouped_shared": "7",
    "#8": "8", "8": "8", "m4_grouped_separated": "8", "mortm5_m4_grouped_separated": "8",
}


def resolve_ablation_subdir(model_name: str = "", ablation_id: str = "") -> str:
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


class RemoteMORTM5HybridTrainer:
    """Mac 側で全体の学習進行・検証・W&Bロギング・チェックポイントを統括するハイブリッドトレーナー。"""

    def __init__(
        self,
        client: RemoteColabMORTM5Client,
        m_args: MORTM5Args,
        t_args: TrainArgs,
        version: str,
        save_dir: str,
        ablation_id: str = "",
        colab_split_dir: Optional[str] = None,
        mac_split_dir: Optional[str] = None,
        stage_to_local: bool = False,
        resume_ckpt_path: Optional[str] = None,
        download_ckpt_to_mac: bool = False,
        val_max_batches: int = 0,
        seed: int = 42,
    ):
        self.client = client
        self.m_args = m_args
        self.t_args = t_args
        self.version = version
        self.ablation_subdir = resolve_ablation_subdir(model_name=m_args.name, ablation_id=ablation_id)
        self.save_dir = os.path.join(save_dir, self.ablation_subdir)
        self.colab_split_dir = colab_split_dir
        self.mac_split_dir = mac_split_dir
        self.stage_to_local = stage_to_local
        self.resume_ckpt_path = resume_ckpt_path
        self.download_ckpt_to_mac = download_ckpt_to_mac
        self.val_max_batches = val_max_batches
        self.seed = seed

        os.makedirs(self.save_dir, exist_ok=True)
        self.mac_train_ds: Optional[MORTM5MemmapDataset] = None
        self.mac_val_ds: Optional[MORTM5MemmapDataset] = None
        self.collate_fn = MORTM5Collate(pad_token_id=PAD_TOKEN_ID)

        if self.mac_split_dir:
            self.mac_train_ds = MORTM5MemmapDataset.from_split_dir(
                token_dir=self.mac_split_dir, split="train", args=self.m_args
            )
            val_bin = os.path.join(self.mac_split_dir, "val_tokens.bin")
            if os.path.isfile(val_bin):
                self.mac_val_ds = MORTM5MemmapDataset.from_split_dir(
                    token_dir=self.mac_split_dir, split="val", args=self.m_args
                )

    def run(self, project_name: Optional[str] = "MORTM5-Ablation", wandb_run_id: Optional[str] = None) -> Dict[str, Any]:
        health_info = self.client.health()
        print("=" * 76)
        print("🌐 Connected to Colab Remote GPU Worker")
        print(f"  - Colab mortm ver  : {health_info.get('mortm_version')}")
        print(f"  - GPU Device       : {health_info.get('gpu_name')} ({health_info.get('vram_gb')} GB VRAM)")
        print("=" * 76)

        explicit_total_steps = None
        if self.mac_train_ds is not None and not self.colab_split_dir:
            steps_per_ep = _compute_exact_epoch_optimizer_steps(
                num_windows=len(self.mac_train_ds),
                big_batch_size=int(self.t_args.big_batch_size),
                batch_size=int(self.t_args.batch_size),
                accumulation_steps=int(self.t_args.accumulation_steps),
            )
            explicit_total_steps = max(1, steps_per_ep * max(1, int(self.t_args.num_epochs)))

        init_payload: Dict[str, Any] = {
            "model_config": self.m_args.to_dict(),
            "train_config": self.t_args.to_dict(),
            "ablation_id": self.ablation_subdir,
            "colab_split_dir": self.colab_split_dir,
            "stage_to_local": False,
            "resume_checkpoint_path": self.resume_ckpt_path,
            "version": self.version,
            "clean_old_checkpoints": (self.resume_ckpt_path is None),
        }
        if explicit_total_steps is not None:
            init_payload["total_steps"] = explicit_total_steps

        init_info = self.client.init_mortm5(init_payload)
        if "error" in init_info:
            raise RuntimeError(f"Colab /mortm5/init failed: {init_info['error']}")

        num_train_windows = (
            int(init_info["train_windows"])
            if self.colab_split_dir
            else (len(self.mac_train_ds) if self.mac_train_ds is not None else 0)
        )
        total_steps = int(init_info["total_steps"])
        big_bs = max(1, int(self.t_args.big_batch_size))
        total_outer_batches = max(1, math.ceil(num_train_windows / big_bs))
        drive_ckpt_dir = init_info.get("drive_ckpt_dir", f"/content/drive/MyDrive/MORTM5_checkpoints/{self.ablation_subdir}")

        print(f"🚀 Initialized {init_info['model_class']} ({init_info['model_name']} | Ablation #{self.ablation_subdir}) on Colab GPU:")
        print(f"  - Mode / Vocab     : model_mode={init_info['model_mode']} | separate_rvq_layers={init_info['separate_rvq_layers']} | vocab_size={init_info['vocab_size']}")
        print(f"  - Parameters       : {init_info['total_params']:,} total (Muon: {init_info['muon_params']:,}, AdamW: {init_info['adamw_params']:,})")
        print(f"  - Dataset Windows  : train={num_train_windows:,} ({total_outer_batches:,} outer batches/epoch), val={init_info['val_windows']:,}")
        print(f"  - Total Opt Steps  : {total_steps:,} steps (Resumed at step={init_info['step']}, outer={init_info['outer_batch_idx']})")
        print(f"  - Drive Save Dir   : {drive_ckpt_dir}")
        print("=" * 76)

        use_wandb = bool(project_name and wandb is not None and os.environ.get("WANDB_MODE") != "disabled")
        w_id = wandb_run_id or ""
        if use_wandb:
            run = wandb.init(
                project=project_name,
                name=self.version,
                id=wandb_run_id,
                resume="allow",
                config={
                    "ablation_id": self.ablation_subdir,
                    "model": self.m_args.to_dict(),
                    "train": self.t_args.to_dict(),
                    "colab_gpu": health_info.get("gpu_name"),
                    "total_steps": total_steps,
                },
            )
            w_id = run.id

        current_step = int(init_info.get("step", 0))
        start_epoch = int(init_info.get("epoch", 0))
        start_outer = int(init_info.get("outer_batch_idx", 0))
        if init_info.get("resumed_from") and current_step > 0:
            start_outer += 1
            if start_outer >= total_outer_batches:
                start_epoch += 1
                start_outer = 0

        last_val_tokens = int(init_info.get("all_tokens", 0))
        val_interval_tokens = max(1, int(self.t_args.val_total_tokens))
        last_val_loss = float(init_info.get("verification_loss", 0.0))

        # パーセンテージマイルストーン (25%, 50%, 75%, 100%) — 再開時は到達済みのものを登録しておく
        pct_list = self.t_args.checkpoint_percents or [25, 50, 75, 100]
        saved_milestones: set[int] = {
            int(pct)
            for pct in pct_list
            if current_step >= max(1, int(round(total_steps * (float(pct) / 100.0))))
        }

        from mortm.train.epoch import EpochObserver
        epoch_loss = EpochObserver(2500)
        active_params = int(init_info.get("total_params", 0))

        epoch = start_epoch
        outer_idx = start_outer
        try:
            for epoch in range(start_epoch, max(1, int(self.t_args.num_epochs))):
                rng = np.random.RandomState(self.seed + epoch)
                if self.t_args.shuffle:
                    all_indices = rng.permutation(num_train_windows).tolist()
                else:
                    all_indices = list(range(num_train_windows))

                outer_start = start_outer if epoch == start_epoch else 0
                pbar = tqdm(
                    range(outer_start, total_outer_batches),
                    initial=outer_start,
                    total=total_outer_batches,
                    desc=f"Epoch {epoch+1}/{self.t_args.num_epochs} [#{self.ablation_subdir} {self.version}]",
                )

                for outer_idx in pbar:
                    win_slice = all_indices[outer_idx * big_bs : (outer_idx + 1) * big_bs]
                    if not win_slice:
                        continue

                    if self.colab_split_dir:
                        self.client.start_outer_batch(outer_idx=outer_idx, epoch=epoch, window_indices=win_slice)
                    else:
                        assert self.mac_train_ds is not None
                        samples = [self.mac_train_ds[idx] for idx in win_slice]
                        batch_tensor = torch.nn.utils.rnn.pad_sequence(
                            samples, batch_first=True, padding_value=PAD_TOKEN_ID
                        )
                        self.client.upload_token_batch(batch_tensor)
                        self.client.start_outer_batch(outer_idx=outer_idx, epoch=epoch, window_indices=None)

                    cursor = 0
                    while True:
                        poll = self.client.poll_outer_batch(cursor)
                        if poll.get("error"):
                            raise RuntimeError(f"Colab GPU Worker Error on outer_batch {outer_idx}: {poll['error']}")

                        for m in poll.get("results", []):
                            epoch_loss.add(float(m["loss"]))
                            pbar.set_postfix({
                                "step": f"{m['step']}/{total_steps}",
                                "loss": f"{epoch_loss.get():.4f}",
                                "val": f"{last_val_loss:.4f}",
                                "lr": f"{m['lr_muon']:.2e}",
                                "tok": f"{m['all_tokens']/1e6:.1f}M",
                            })

                        cursor = int(poll.get("next_cursor", cursor))
                        if poll.get("done"):
                            break
                        time.sleep(0.5)

                    current_tokens = int(poll.get("all_tokens", 0))
                    current_step = int(poll.get("step", 0))

                    # マイルストーン判定
                    milestone_tag = ""
                    for pct in pct_list:
                        target_step = max(1, int(round(total_steps * (float(pct) / 100.0))))
                        if current_step >= target_step and int(pct) not in saved_milestones:
                            saved_milestones.add(int(pct))
                            milestone_tag = f"pct{int(pct)}"

                    is_last_outer = (outer_idx + 1) == total_outer_batches
                    should_validate = (
                        (current_tokens - last_val_tokens) >= val_interval_tokens
                        or is_last_outer
                        or bool(milestone_tag)
                    )

                    if should_validate:
                        # 検証バッチを GPU で回している裏で Google Drive (1-8 サブディレクトリ) への保存を並列開始する
                        ckpt_info = self.client.save_checkpoint(
                            outer_idx=outer_idx,
                            epoch=epoch,
                            total_outer=total_outer_batches,
                            val_loss=last_val_loss,
                            version=self.version,
                            ablation_id=self.ablation_subdir,
                            wandb_id=w_id,
                            prepare_download=bool(self.download_ckpt_to_mac and (milestone_tag or is_last_outer)),
                            milestone=milestone_tag,
                            sync_drive=is_last_outer,
                        )

                        if not self.colab_split_dir and self.mac_val_ds is not None:
                            val_samples = [self.mac_val_ds[i] for i in range(min(len(self.mac_val_ds), big_bs))]
                            val_tensor = torch.nn.utils.rnn.pad_sequence(
                                val_samples, batch_first=True, padding_value=PAD_TOKEN_ID
                            )
                            self.client.upload_token_batch(val_tensor)
                        val_res = self.client.validate(max_batches=self.val_max_batches)
                        last_val_loss = float(val_res.get("val_loss", 0.0))
                        val_ppl = float(val_res.get("val_ppl", 1.0))
                        last_val_tokens = current_tokens
                        train_avg_loss = float(epoch_loss.get()) if len(epoch_loss.loss) > 0 else 0.0
                        seq_steps = (
                            current_tokens // max(1, int(self.m_args.vq_groups))
                            if getattr(self.m_args, "model_mode", "flat") == "grouped"
                            else current_tokens
                        )
                        print(
                            f"\n📊 [Validation @ Step {current_step} | {current_tokens/1e6:.2f}M tokens] "
                            f"train_loss={train_avg_loss:.4f} | val_loss={last_val_loss:.4f} | val_ppl={val_ppl:.2f}",
                            flush=True,
                        )
                        if use_wandb:
                            wandb.log(
                                {
                                    "axis/val_loss": last_val_loss,
                                    "axis/tokens": current_tokens,
                                    "axis/seq_steps": seq_steps,
                                    "axis/flops": 6 * active_params * seq_steps,
                                    "trainer/global_step": current_step,
                                    "train/loss": train_avg_loss,
                                    "val/loss": last_val_loss,
                                    "val/perplexity": val_ppl,
                                },
                                step=current_step,
                            )

                        if ckpt_info.get("download_bytes", 0) > 0:
                            try:
                                dl_data = self.client.download_prepared_checkpoint(int(ckpt_info["download_bytes"]))
                                mac_pth = os.path.join(self.save_dir, f"MORTM5.{self.version}_step{current_step}.pth")
                                torch.save(dl_data["model_state_dict"], mac_pth)
                                print(f"\n💾 Downloaded model weights to Mac: {mac_pth}", flush=True)
                            except Exception as dl_err:
                                print(f"\n⚠️ [Mac Download Warning] Failed to download checkpoint to Mac (already saved on Google Drive): {dl_err}", flush=True)
        except KeyboardInterrupt:
            print(f"\n🛑 学習停止を検知しました。Google Drive ({drive_ckpt_dir}) へ最新チェックポイント (Step {current_step}) を保存しています...", flush=True)
            try:
                self.client.save_checkpoint(
                    outer_idx=outer_idx,
                    epoch=epoch,
                    total_outer=total_outer_batches,
                    val_loss=last_val_loss,
                    version=self.version,
                    ablation_id=self.ablation_subdir,
                    wandb_id=w_id,
                    prepare_download=False,
                    milestone=f"stop_step{current_step}",
                    sync_drive=True,
                )
                print(f"✅ Google Drive ({drive_ckpt_dir}) への保存が完了しました。", flush=True)
            except Exception as save_err:
                print(f"⚠️ 停止時のチェックポイント保存に失敗しました: {save_err}", flush=True)
        finally:
            if use_wandb:
                wandb.finish()

        return {
            "status": "completed",
            "version": self.version,
            "ablation_subdir": self.ablation_subdir,
            "final_step": current_step,
            "final_val_loss": last_val_loss,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mac <-> Google Colab Hybrid Trainer for MORTM5 8-Way Ablation Models"
    )
    parser.add_argument("--url", type=str, required=True, help="Colab Cloudflare Tunnel URL (例: https://xxxx.trycloudflare.com)")
    parser.add_argument(
        "--ablation",
        type=str,
        default="m4_grouped_shared",
        choices=list(MORTM5_ABLATION_PRESETS.keys()) + ["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "1", "2", "3", "4", "5", "6", "7", "8"],
        help="学習するアブレーションモデル (#1〜#8 または 1〜8)",
    )
    parser.add_argument("--model-config", type=str, default=None, help="任意のモデル JSON パス (指定時は --ablation より優先)")
    parser.add_argument("--train-config", type=str, default="configs/train/mortm5/ablation_muon_cos.json", help="学習設定 JSON パスまたはプリセット名 ('muon_cos')")
    parser.add_argument(
        "--colab-split-dir",
        type=str,
        default=None,
        help="Google Drive 上のトークンディレクトリ (例: /content/drive/MyDrive/MORTM5_tokens/m4_g4l4)",
    )
    parser.add_argument(
        "--mac-split-dir",
        type=str,
        default=None,
        help="Mac ローカルのトークンディレクトリ (Colab 上にデータがない場合のストリーミング用)",
    )
    parser.add_argument("--save-dir", type=str, default="out/mortm5_ablation", help="Mac 側のモデル保存先ディレクトリ")
    parser.add_argument("--version", type=str, default=None, help="チェックポイント・W&B のバージョン識別子")
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Google Drive 上の再開チェックポイントパス")
    parser.add_argument("--download-ckpt", action="store_true", help="マイルストーン及び完了時に Mac 側へも重み (.pth) をダウンロードする")
    parser.add_argument("--project", type=str, default="MORTM5-Ablation", help="Weights & Biases プロジェクト名")
    parser.add_argument("--wandb-id", type=str, default=None, help="再開する W&B Run ID")
    parser.add_argument("--no-wandb", action="store_true", help="W&B ロギングを無効化する")

    # ハイパーパラメータ上書きオプション
    parser.add_argument("--lr", type=float, default=None, help="Muon / AdamW 基準ピーク学習率の上書き")
    parser.add_argument("--batch-size", type=int, default=None, help="ミニバッチサイズの上書き")
    parser.add_argument("--accum-steps", type=int, default=None, help="勾配累積ステップ数の上書き")
    parser.add_argument("--big-batch-size", type=int, default=None, help="Outer Batch ウィンドウ数の上書き")
    parser.add_argument("--epochs", type=int, default=None, help="エポック数の上書き")
    parser.add_argument("--val-max-batches", type=int, default=0, help="検証時の最大ミニバッチ数 (0=全件)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    m_source = cli.model_config if cli.model_config else cli.ablation
    m_args = MORTM5Args(m_source)
    t_args = TrainArgs(cli.train_config if os.path.isfile(cli.train_config) else "muon_cos")

    if cli.lr is not None:
        t_args.lr_param = float(cli.lr)
    if cli.batch_size is not None:
        t_args.batch_size = int(cli.batch_size)
    if cli.accum_steps is not None:
        t_args.accumulation_steps = int(cli.accum_steps)
    if cli.big_batch_size is not None:
        t_args.big_batch_size = int(cli.big_batch_size)
    if cli.epochs is not None:
        t_args.num_epochs = int(cli.epochs)

    # 既定の Google Drive パス自動補完 (--colab-split-dir も --mac-split-dir も未指定の場合)
    colab_split_dir = cli.colab_split_dir
    if not colab_split_dir and not cli.mac_split_dir:
        tok_key = m_args.tokenizer_key or ("m4_g4l4" if m_args.vq_groups == 4 else "m3_g8l2")
        colab_split_dir = f"/content/drive/MyDrive/MORTM5_tokens/{tok_key}"

    version = cli.version or f"{m_args.name}_muon_cos"
    client = RemoteColabMORTM5Client(worker_url=cli.url)
    hybrid_trainer = RemoteMORTM5HybridTrainer(
        client=client,
        m_args=m_args,
        t_args=t_args,
        version=version,
        save_dir=cli.save_dir,
        ablation_id=cli.ablation,
        colab_split_dir=colab_split_dir,
        mac_split_dir=cli.mac_split_dir,
        stage_to_local=False,
        resume_ckpt_path=cli.resume_ckpt,
        download_ckpt_to_mac=cli.download_ckpt,
        val_max_batches=cli.val_max_batches,
        seed=cli.seed,
    )
    hybrid_trainer.run(
        project_name=None if cli.no_wandb else cli.project,
        wandb_run_id=cli.wandb_id,
    )


if __name__ == "__main__":
    main()
