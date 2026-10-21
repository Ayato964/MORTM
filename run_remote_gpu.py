"""
Mac (Local Dataset & Checkpoint/W&B Manager) <-> Google Colab (Remote CUDA GPU Worker)
ハイブリッド分散学習クライアント (GPU待ち時間0秒・非同期バックグラウンド同期版)
"""

import argparse
import datetime
import gc
import io
import json
import os
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch

from mortm.models.modules.config import MORTM5Args
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.train.config import TrainArgs
from mortm.train.train import (
    EpochObserver,
    _build_preloading_dataset,
    _collect_grouped_paths,
    get_data_loader,
)
from run_midi_vision import MidiVisionTrainSet


WORKER_HOT_PATCH = """
import os, shutil, threading
from starlette.responses import JSONResponse, Response

STATE["job_gen"] = int(STATE.get("job_gen", 0)) + 1

class MidiVision(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNetEncoder(args)
        self.vq_groups = getattr(args, "vq_groups", 4)
        self.codebook_dim = args.encoder_wout // self.vq_groups
        self.codebook_size = getattr(args, "codebook_size", 1024)
        self.pre_vq_norm = nn.LayerNorm(self.codebook_dim, elementwise_affine=False)
        self.rvq = GroupedResidualVQ(
            dim=args.encoder_wout,
            groups=self.vq_groups,
            codebook_dim=self.codebook_dim,
            num_quantizers=args.encoder_layer,
            codebook_size=self.codebook_size,
            sync_codebook=False,
            threshold_ema_dead_code=0.05,
            rotation_trick=False,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.999,
        )
        self.decoder = ResNetDecoder(args)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        orig_shape = z.shape
        z_norm = self.pre_vq_norm(z.reshape(*orig_shape[:-1], self.vq_groups, self.codebook_dim)).reshape(orig_shape)
        x_q, tokens, commit_loss = self.rvq(z_norm)
        tokens = tokens.movedim(0, -2).flatten(-2)
        commit_loss = commit_loss.mean()
        out = self.decoder(x_q)
        return out, tokens, commit_loss


class MidiVisionLoss(nn.Module):
    def __init__(self, pos_weight=10.0, dice_weight=1.0, onset_weight=1.0, vel_weight=5.0, commit_weight=0.25):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.onset_weight = onset_weight
        self.vel_weight = vel_weight
        self.commit_weight = commit_weight
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))
        self.last_metrics = {}

    def forward(self, decoded: torch.Tensor, original: torch.Tensor, commit_loss: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight_tensor.to(decoded.device)
        pred_active_logits = decoded[:, 1::2].float()
        target_active = original[:, 1::2].float()
        bce_active = F.binary_cross_entropy_with_logits(pred_active_logits, target_active, pos_weight=pos_weight)
        prob_active = torch.sigmoid(pred_active_logits)
        inter_active = (prob_active * target_active).sum(dim=(-2, -1))
        card_active = prob_active.sum(dim=(-2, -1)) + target_active.sum(dim=(-2, -1))
        dice_active = (1.0 - (2.0 * inter_active + 1e-5) / (card_active + 1e-5)).mean()
        loss_active = bce_active + self.dice_weight * dice_active

        pred_vel_logits = decoded[:, 0::2].float()
        pred_vel = torch.sigmoid(pred_vel_logits)
        target_vel = original[:, 0::2].float()
        target_onset = (target_vel >= 0.5).float()
        onset_sum = target_onset.sum()

        vel_sample_weight = torch.where(target_onset > 0.5, pos_weight, 1.0)
        bce_onset_vel = F.binary_cross_entropy_with_logits(pred_vel_logits, target_vel, weight=vel_sample_weight)
        onset_tp_prob = (pred_vel * 2.0).clamp(max=1.0)
        onset_prob = torch.where(target_onset > 0.5, onset_tp_prob, pred_vel)
        inter_onset = (onset_prob * target_onset).sum(dim=(-2, -1))
        card_onset = onset_prob.sum(dim=(-2, -1)) + target_onset.sum(dim=(-2, -1))
        dice_onset = (1.0 - (2.0 * inter_onset + 1e-5) / (card_onset + 1e-5)).mean()

        if onset_sum > 0:
            hinge_onset = (F.relu(0.52 - pred_vel) * target_onset).sum() / (onset_sum + 1e-5)
            vel_diff = F.smooth_l1_loss(pred_vel, target_vel, reduction="none") * target_onset
            loss_vel = vel_diff.sum() / (onset_sum + 1e-5)
        else:
            hinge_onset = pred_vel.sum() * 0.0
            loss_vel = pred_vel.sum() * 0.0

        prob_active_det = prob_active.detach()
        pred_attack_edge = F.pad(F.relu(prob_active_det[..., 1:] - prob_active_det[..., :-1]), (1, 0), value=0.0)
        allowed_onset_env = torch.maximum(pred_attack_edge, target_onset)
        ghost_onset = F.relu(pred_vel - allowed_onset_env)
        loss_sync_suppress = ghost_onset.sum() / (pred_vel.detach().sum() + onset_sum + 1e-5)
        inter_sync = (onset_tp_prob[..., 1:] * pred_attack_edge[..., 1:]).sum(dim=(-2, -1))
        card_sync = onset_tp_prob[..., 1:].sum(dim=(-2, -1)) + pred_attack_edge[..., 1:].sum(dim=(-2, -1))
        loss_sync_edge = (1.0 - (2.0 * inter_sync + 1e-5) / (card_sync + 1e-5)).mean()
        loss_sync = loss_sync_suppress + 0.5 * loss_sync_edge
        loss_onset = bce_onset_vel + self.dice_weight * dice_onset + hinge_onset + loss_sync

        commit_scalar = commit_loss.mean() if isinstance(commit_loss, torch.Tensor) else commit_loss
        loss_commit = commit_scalar * self.commit_weight

        with torch.no_grad():
            pred_active_bin = (prob_active >= 0.5).float()
            tp = (pred_active_bin * target_active).sum().item()
            fp = (pred_active_bin * (1.0 - target_active)).sum().item()
            fn = ((1.0 - pred_active_bin) * target_active).sum().item()
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall + 1e-8)) if (precision + recall) > 0 else 0.0

            pred_onset_bin = (pred_vel >= 0.5).float()
            otp = (pred_onset_bin * target_onset).sum().item()
            ofp = (pred_onset_bin * (1.0 - target_onset)).sum().item()
            ofn = ((1.0 - pred_onset_bin) * target_onset).sum().item()
            onset_recall = otp / (otp + ofn + 1e-8) if (otp + ofn) > 0 else 0.0
            onset_precision = otp / (otp + ofp + 1e-8) if (otp + ofp) > 0 else 0.0
            onset_f1 = (2.0 * onset_precision * onset_recall / (onset_precision + onset_recall + 1e-8)) if (onset_precision + onset_recall) > 0 else 0.0
            vel_mae = (((pred_vel.clamp(0.5, 1.0) - target_vel).abs() * 2.0 * target_onset).sum().item() / (onset_sum.item() + 1e-5)) if onset_sum > 0 else 0.0

        total_loss = loss_active + self.onset_weight * loss_onset + self.vel_weight * loss_vel + loss_commit
        self.last_metrics = {
            "f1": f1, "recall": recall, "precision": precision,
            "onset_f1": onset_f1, "onset_recall": onset_recall, "onset_precision": onset_precision,
            "vel_mae": vel_mae, "bce": bce_active.item(), "dice": dice_active.item(),
            "onset_loss": loss_onset.item(), "vel_loss": loss_vel.item(),
            "commit": commit_scalar.item() if isinstance(commit_scalar, torch.Tensor) else float(commit_scalar),
            "total": total_loss.item(),
        }
        return total_loss

async def _clear_upload(request):
    STATE["upload_parts"] = []
    return JSONResponse({"status": "cleared"})

async def _append_upload(request):
    body = await request.body()
    STATE.setdefault("upload_parts", []).append(body)
    return JSONResponse({"parts": len(STATE["upload_parts"]), "bytes": len(body)})

def _run_big_batch_worker(raw_comp, batch_size, my_gen):
    lock = STATE.setdefault("gpu_lock", threading.Lock())
    with lock:
        if STATE.get("job_gen", 0) != my_gen:
            return
        try:
            header_len = int.from_bytes(raw_comp[:4], byteorder="big")
            header = json.loads(raw[4 : 4 + header_len].decode("utf-8")) if False else json.loads(raw_comp[4 : 4 + header_len].decode("utf-8"))
            decomp = zlib.decompress(raw_comp[4 + header_len :])
            arr_u8 = np.frombuffer(decomp, dtype=np.uint8).copy().reshape(header["shape"])
            del raw_comp, decomp

            model = STATE["model"]
            optimizer = STATE["optimizer"]
            criterion = STATE["criterion"]
            model.train()

            n_total = len(arr_u8)
            for start in range(0, n_total, batch_size):
                if STATE.get("job_gen", 0) != my_gen:
                    return
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
                    print(f"[Step {m['step']}] Loss:{m['total']:.4f} | NoteF1:{m['f1']*100:.1f}% | OnsetF1:{m['onset_f1']*100:.1f}% | VelMAE:{m['vel_mae']:.4f}", flush=True)
            STATE["job_done"] = True
        except Exception as e:
            STATE["job_error"] = str(e)
            STATE["job_done"] = True

async def _start_big_batch(request):
    batch_size = int(request.query_params.get("batch_size", "1024"))
    raw_comp = b"".join(STATE.get("upload_parts", []))
    STATE["upload_parts"] = []
    STATE["job_results"] = []
    STATE["job_done"] = False
    STATE["job_error"] = None
    my_gen = int(STATE.get("job_gen", 0)) + 1
    STATE["job_gen"] = my_gen
    t = threading.Thread(target=_run_big_batch_worker, args=(raw_comp, batch_size, my_gen), daemon=True)
    t.start()
    return JSONResponse({"status": "started", "compressed_bytes": len(raw_comp), "job_gen": my_gen})

async def _set_lr(request):
    new_lr = float(request.query_params.get("lr", "1e-4"))
    STATE["lr"] = new_lr
    if STATE.get("optimizer") is not None:
        for pg in STATE["optimizer"].param_groups:
            pg["lr"] = new_lr
    return JSONResponse({"status": "lr_updated", "lr": new_lr})

async def _poll_big_batch(request):
    cursor = int(request.query_params.get("cursor", "0"))
    res = STATE.get("job_results", [])
    new_items = res[cursor:]
    return JSONResponse({
        "results": new_items,
        "next_cursor": cursor + len(new_items),
        "done": bool(STATE.get("job_done", False)),
        "error": STATE.get("job_error"),
    })

async def _save_colab_ckpt(request):
    # Colab ローカル (/content/) およびマウント済みなら Google Drive へ即座にチェックポイント保存
    import datetime
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

    # 1. Colab SSD (/content/MORTM5.{version}.checkpoint.pt) へフル保存 (毎バッチ0.2秒で必ず実行)
    colab_path = f"/content/MORTM5.{version}.checkpoint.pt"
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
        "loss_version": "v3_m2_exact",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    torch.save(full_ckpt, colab_path + ".tmp")
    os.replace(colab_path + ".tmp", colab_path)
    STATE["version"] = version

    # 2. Google Drive がマウントされていれば、非同期バックグラウンドスレッドで Google Drive へ即座に保存 (GPU待ち時間 0.00秒・トンネル通信 0バイト)
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        def _bg_save_to_drive(src_path, model_sd, ver, o_idx, s_now):
            try:
                drive_dir = "/content/drive/MyDrive/MORTM5_checkpoints"
                os.makedirs(drive_dir, exist_ok=True)
                # (a) モデル別フルチェックポイント (model + AdamW optimizer) — 複数セッション並列でも絶対に衝突しない
                dst_ckpt = os.path.join(drive_dir, f"MORTM5.{ver}.checkpoint.pt")
                shutil.copy2(src_path, dst_ckpt + ".tmp")
                os.replace(dst_ckpt + ".tmp", dst_ckpt)
                # (b) 推論・評価用の単体モデル重み (.pth)
                dst_pth = os.path.join(drive_dir, f"MORTM5.{ver}_latest.pth")
                torch.save(model_sd, dst_pth + ".tmp")
                os.replace(dst_pth + ".tmp", dst_pth)
                # (c) 20 Outer Batch ごと、または比較基準点 (215, 220, 256) でマイルストーン .pth も保存
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
        return JSONResponse(STATE["ckpt_meta"])

    # 3. Google Drive 未マウント時のみ Mac バックグラウンド回収用バッファを作成
    if refresh_buf or "ckpt_bytes" not in STATE:
        buf = io.BytesIO()
        torch.save({
            "model_state_dict": cpu_model_sd,
            "outer_batch_idx": outer_idx,
            "step": step_now,
            "all_tokens": tokens,
        }, buf)
        comp = zlib.compress(buf.getvalue(), level=1)
        STATE["ckpt_bytes"] = comp
        STATE["ckpt_meta"] = {
            "drive_saved": False,
            "total_bytes": len(comp),
            "step": step_now,
            "outer": outer_idx,
            "tokens": tokens,
            "version": version,
        }
    return JSONResponse(STATE["ckpt_meta"])

async def _get_state_raw_chunk(request):
    start = int(request.query_params.get("start", "0"))
    end = int(request.query_params.get("end", "2097152"))
    b = STATE.get("ckpt_bytes", b"")[start:end]
    return Response(content=b, media_type="application/octet-stream")

async def _restore_live_state(request):
    version = request.query_params.get("version", "")
    drive_ok = os.path.exists("/content/drive/MyDrive")
    colab_ver_path = f"/content/MORTM5.{version}.checkpoint.pt" if version else ""

    # 起動時に Colab SSD の当該モデルチェックポイントが存在し、かつ Google Drive がマウントされていれば即座に Drive へ同期
    if drive_ok and colab_ver_path and os.path.exists(colab_ver_path):
        try:
            c_tmp = torch.load(colab_ver_path, map_location="cpu", weights_only=False)
            if c_tmp.get("loss_version") == "v3_m2_exact":
                drive_dir = "/content/drive/MyDrive/MORTM5_checkpoints"
                os.makedirs(drive_dir, exist_ok=True)
                dst_ckpt = os.path.join(drive_dir, f"MORTM5.{version}.checkpoint.pt")
                shutil.copy2(colab_ver_path, dst_ckpt + ".tmp")
                os.replace(dst_ckpt + ".tmp", dst_ckpt)
        except Exception:
            pass

    saved = STATE.pop("_saved_live", None)
    if saved is not None and (not version or saved.get("version") in (None, "", version)) and hasattr(saved.get("model"), "rvq"):
        STATE["model"] = saved["model"]
        STATE["optimizer"] = saved["optimizer"]
        if STATE["optimizer"] is not None:
            for pg in STATE["optimizer"].param_groups:
                pg["lr"] = STATE.get("lr", 4e-4)
        STATE["criterion"] = saved["criterion"]
        STATE["step"] = saved["step"]
        meta = STATE.get("ckpt_meta", {})
        return JSONResponse({
            "status": "restored_live",
            "version": version,
            "step": STATE["step"],
            "outer": int(meta.get("outer", 0)),
            "tokens": int(meta.get("tokens", 0)),
            "lr": STATE.get("lr", 4e-4),
            "drive_mounted": drive_ok,
        })

    # Colab SSD または Google Drive から version 一致のチェックポイントを検索して即時復元
    best_c = None
    cand_list = []
    if version:
        cand_list.append((f"/content/MORTM5.{version}.checkpoint.pt", "colab_ssd"))
        cand_list.append((f"/content/drive/MyDrive/MORTM5_checkpoints/MORTM5.{version}.checkpoint.pt", "google_drive"))
    for cand_path, src_name in cand_list:
        if os.path.exists(cand_path):
            try:
                c = torch.load(cand_path, map_location="cpu", weights_only=False)
                if version and c.get("version") not in (None, "", version):
                    continue
                if c.get("loss_version") != "v3_m2_exact":
                    continue
                c["_source"] = src_name
                if best_c is None or int(c.get("optimizer_step", 0)) > int(best_c.get("optimizer_step", 0)):
                    best_c = c
            except Exception:
                pass

    if best_c is not None and STATE.get("model") is not None and int(best_c.get("optimizer_step", 0)) >= STATE.get("step", 0):
        STATE["model"].load_state_dict(best_c["model_state_dict"], strict=True)
        STATE["model"] = STATE["model"].to(DEVICE).train()
        if best_c.get("optimizer_state_dict") is not None and STATE.get("optimizer") is not None:
            STATE["optimizer"].load_state_dict(best_c["optimizer_state_dict"])
            for pg in STATE["optimizer"].param_groups:
                pg["lr"] = STATE.get("lr", 4e-4)
        STATE["step"] = int(best_c.get("optimizer_step", STATE["step"]))
        STATE["version"] = version
        return JSONResponse({
            "status": "restored_from_colab_ssd",
            "source": best_c.get("_source", "colab_ssd"),
            "version": version,
            "step": STATE["step"],
            "outer": int(best_c.get("outer_batch_idx", 0)),
            "tokens": int(best_c.get("all_tokens", 0)),
            "wandb_run_id": best_c.get("wandb_run_id"),
            "drive_mounted": drive_ok,
        })
    return JSONResponse({"status": "no_saved_live", "version": version, "step": STATE.get("step", 0), "drive_mounted": drive_ok})

if STATE.get("model") is not None and STATE.get("step", 0) > 0 and hasattr(STATE.get("model"), "rvq"):
    STATE["_saved_live"] = {
        "model": STATE["model"],
        "optimizer": STATE["optimizer"],
        "criterion": STATE["criterion"],
        "step": STATE["step"],
        "version": STATE.get("version", ""),
    }

app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") not in {
    "/clear_upload", "/append_upload", "/start_big_batch", "/poll_big_batch",
    "/save_colab_ckpt", "/get_state_raw_chunk", "/restore_live_state", "/set_lr"
}]
app.add_route("/clear_upload", _clear_upload, methods=["POST"])
app.add_route("/append_upload", _append_upload, methods=["POST"])
app.add_route("/start_big_batch", _start_big_batch, methods=["POST"])
app.add_route("/poll_big_batch", _poll_big_batch, methods=["GET"])
app.add_route("/save_colab_ckpt", _save_colab_ckpt, methods=["POST"])
app.add_route("/get_state_raw_chunk", _get_state_raw_chunk, methods=["GET"])
app.add_route("/restore_live_state", _restore_live_state, methods=["POST"])
app.add_route("/set_lr", _set_lr, methods=["POST"])
"""


class _WorkerHotPatch:
    def __reduce__(self):
        return (eval, (f"exec({WORKER_HOT_PATCH!r}, __import__('colab_gpu_worker').__dict__)",))


def _compress_u8_batch(arr_u8: np.ndarray) -> bytes:
    arr_u8 = np.ascontiguousarray(arr_u8, dtype=np.uint8)
    header = json.dumps({"shape": list(arr_u8.shape)}).encode("utf-8")
    header_len = len(header).to_bytes(4, byteorder="big")
    comp = zlib.compress(arr_u8.tobytes(), level=1)
    return header_len + header + comp


def _upload_and_start_big_batch(session: requests.Session, remote_url: str, payload: bytes, batch_size: int):
    r = session.post(f"{remote_url}/clear_upload", timeout=30)
    r.raise_for_status()

    chunk_size = 12 * 1024 * 1024  # 12 MB per upload chunk (upstream is fast!)
    for start in range(0, len(payload), chunk_size):
        part = payload[start : start + chunk_size]
        for attempt in range(5):
            try:
                rp = session.post(
                    f"{remote_url}/append_upload",
                    data=part,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=60,
                )
                rp.raise_for_status()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.0)

    rs = session.post(f"{remote_url}/start_big_batch?batch_size={batch_size}", timeout=30)
    rs.raise_for_status()


def main():
    parser = argparse.ArgumentParser(description="MORTM5 Remote Colab GPU Hybrid Trainer")
    parser.add_argument("--remote_url", type=str, required=True, help="Colab cloudflared tunnel URL")
    parser.add_argument("--model_config", type=str, default="configs/models/mortm5/m2_compact_g4l2_d128_k2048.json")
    parser.add_argument("--train_config", type=str, default="configs/train/mortm5/default.json")
    parser.add_argument("--data_dir", type=str, default="data/gmd/train.json")
    parser.add_argument("--eval_json", type=str, default="data/gmd/eval.json")
    parser.add_argument("--save_dir", type=str, default="out/models/mortm5/m2_compact_g4l2_d128_k2048")
    parser.add_argument("--version", type=str, default="M2_Compact_G4L2_D128_K2048")
    parser.add_argument("--override_lr", type=float, default=None, help="Optional learning rate override")
    parser.add_argument("--lr_anneal_outer", type=int, default=160, help="Outer batch index to switch LR to lr_anneal_val (0 to disable)")
    parser.add_argument("--lr_anneal_val", type=float, default=1e-4, help="Annealed learning rate value")
    parser.add_argument("--max_outer", type=int, default=220, help="Stop cleanly after reaching this outer batch index (0 for full epoch)")
    args = parser.parse_args()

    remote_url = args.remote_url.rstrip("/")
    session = requests.Session()

    print(f"[RemoteGPU] Connecting to {remote_url} ...", flush=True)
    r = session.get(f"{remote_url}/health", timeout=30)
    r.raise_for_status()
    health = r.json()
    remote_current_step = int(health.get("step", 0))
    print(
        f"[RemoteGPU] Connected! Remote GPU: {health.get('gpu_name')} | "
        f"VRAM: {health.get('vram_gb')} GB | Remote Step: {remote_current_step}",
        flush=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    m_args = MORTM5Args(args.model_config)
    t_args = TrainArgs(args.train_config)
    if args.override_lr is not None:
        t_args.lr_param = args.override_lr
    progress = _DefaultLearningProgress()

    model_name = f"{m_args.name}.{args.version}"
    checkpoint_path = os.path.join(args.save_dir, f"{model_name}.checkpoint.pt")

    wandb_id = None
    start_epoch = 0
    resume_outer_batch_idx = 0
    optimizer_step = 0
    verification_loss = 0.0
    all_tokens_val = 0
    ckpt = None

    if os.path.exists(checkpoint_path):
        print(f"\033[32m[Checkpoint]\033[0m ローカルチェックポイントを検出: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        wandb_id = ckpt.get("wandb_run_id")
        start_epoch = ckpt.get("epoch", 0)
        resume_outer_batch_idx = ckpt.get("outer_batch_idx", 0)
        optimizer_step = int(ckpt.get("optimizer_step", 0))
        verification_loss = float(ckpt.get("verification_loss", 0.0))
        all_tokens_val = int(ckpt.get("all_tokens", 0))
        if "rng_states" in ckpt and ckpt["rng_states"] is not None:
            rng = ckpt["rng_states"]
            if rng.get("torch_rng") is not None:
                t_rng = rng["torch_rng"]
                if isinstance(t_rng, torch.Tensor):
                    t_rng = t_rng.cpu().byte()
                torch.set_rng_state(t_rng)
            if rng.get("numpy_rng") is not None:
                np.random.set_state(rng["numpy_rng"])
            if rng.get("python_rng") is not None:
                random.setstate(rng["python_rng"])

    with open(args.model_config, "r") as f:
        model_cfg_dict = json.load(f)

    # まず 5KB の軽量ペイロードでホットパッチ＆モデル枠のみ初期化し、
    # Colab 側の (1) VRAM上のライブ状態 / (2) Colab SSD / (3) Google Drive のいずれかに最新チェックポイントがあれば 0.2秒で即時復元する
    init_buf = io.BytesIO()
    torch.save(
        {
            "patch": _WorkerHotPatch(),
            "model_config": model_cfg_dict,
            "lr": float(t_args.lr_param),
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "optimizer_step": optimizer_step,
        },
        init_buf,
    )
    init_bytes = zlib.compress(init_buf.getvalue(), level=1)
    print(f"[RemoteGPU] Probing Colab GPU worker ({len(init_bytes)/1e3:.1f} KB fast-init)...", flush=True)
    r = session.post(
        f"{remote_url}/init",
        data=init_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=60,
    )
    r.raise_for_status()
    rr = session.post(f"{remote_url}/restore_live_state?version={args.version}", timeout=60)
    rr.raise_for_status()
    rinfo = rr.json()
    restored_step = int(rinfo.get("step", 0))

    if rinfo.get("status") in ("restored_live", "restored_from_colab_ssd") and restored_step >= optimizer_step and restored_step > 0:
        resume_outer_batch_idx = max(resume_outer_batch_idx, int(rinfo.get("outer", resume_outer_batch_idx)))
        optimizer_step = max(optimizer_step, restored_step)
        all_tokens_val = max(all_tokens_val, int(rinfo.get("tokens", all_tokens_val)))
        if rinfo.get("wandb_run_id"):
            wandb_id = rinfo["wandb_run_id"]
        print(f"[RemoteGPU] Zero-Upload Instant Restore succeeded! {rinfo}", flush=True)
    elif ckpt is not None:
        # Colab 側にチェックポイントが存在しない場合のみ Mac からアップロード
        full_buf = io.BytesIO()
        torch.save(
            {
                "model_config": model_cfg_dict,
                "lr": float(t_args.lr_param),
                "model_state_dict": ckpt["model_state_dict"],
                "optimizer_state_dict": ckpt["optimizer_state_dict"],
                "optimizer_step": optimizer_step,
            },
            full_buf,
        )
        full_bytes = zlib.compress(full_buf.getvalue(), level=1)
        print(f"[RemoteGPU] Uploading Mac checkpoint to fresh Colab worker ({len(full_bytes)/1e6:.2f} MB)...", flush=True)
        r = session.post(
            f"{remote_url}/init",
            data=full_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=180,
        )
        r.raise_for_status()
        print(f"[RemoteGPU] Remote model initialized from Mac checkpoint: {r.json()}", flush=True)

    if args.override_lr is None and args.lr_anneal_outer > 0 and resume_outer_batch_idx >= args.lr_anneal_outer:
        t_args.lr_param = args.lr_anneal_val
        session.post(f"{remote_url}/set_lr?lr={args.lr_anneal_val}", timeout=30)

    trainer = MidiVisionTrainSet(
        args=m_args,
        t_args=t_args,
        progress=progress,
        enable_shift=True,
        use_wandb=True,
        project_name="MORTM5_MidiVision",
        run_name=args.version,
        wandb_id=wandb_id,
        initial_step=optimizer_step,
    )
    trainer.optimizer_steps = optimizer_step
    trainer.global_step = optimizer_step
    trainer.all_tokens.fill_(all_tokens_val)

    grouped_paths, dataset_names = _collect_grouped_paths(args.data_dir)
    mortm_dataset = _build_preloading_dataset(grouped_paths, progress, dataset_names)
    eval_grouped_paths, eval_dataset_names = _collect_grouped_paths(args.eval_json)
    val_dataset = _build_preloading_dataset(eval_grouped_paths, progress, eval_dataset_names)
    train_loader, val_loader = get_data_loader(t_args, (mortm_dataset, val_dataset), shuffle=True)

    tokens_per_chunk = m_args.vq_groups * m_args.encoder_layer
    batch_size = t_args.batch_size

    print(
        f"\033[32m[Hybrid Train Start]\033[0m Model: {args.version}, Epoch: {start_epoch + 1}/{t_args.num_epochs}, "
        f"Resume Outer Batch: {resume_outer_batch_idx}/{len(train_loader)}, "
        f"Step: {optimizer_step}, LR: {t_args.lr_param:.2e}, BatchSize: {batch_size} "
        f"(Anneal@{args.lr_anneal_outer}->{args.lr_anneal_val:.1e}, MaxOuter:{args.max_outer})",
        flush=True,
    )

    last_saved_opt_sd = ckpt["optimizer_state_dict"] if ckpt is not None else None
    last_saved_model_sd = ckpt["model_state_dict"] if ckpt is not None else None

    # トンネル帯域競合ゼロのピギーバック回収ステート:
    # Big Batch アップロード中は一切の下り通信を行わず、GPU が次の Big Batch を計算している最中の
    # ポーリング間隔（1.5秒）を使って 256KB ずつ順番に Mac へ回収する。
    ckpt_sync = {
        "active": False,
        "meta": None,
        "epoch": 0,
        "total_outer": 0,
        "w_id": None,
        "offset": 0,
        "parts": [],
    }

    def _pull_one_ckpt_slice():
        if not ckpt_sync["active"]:
            return False
        meta = ckpt_sync["meta"]
        total_bytes = int(meta["total_bytes"])
        st = ckpt_sync["offset"]
        if st >= total_bytes:
            return False
        en = min(total_bytes, st + 262144)  # 256 KB (~1.2s, ポーリング待機時間と完全一致)
        try:
            rc = session.get(f"{remote_url}/get_state_raw_chunk?start={st}&end={en}", timeout=30)
            rc.raise_for_status()
            ckpt_sync["parts"].append(rc.content)
            ckpt_sync["offset"] = en
        except Exception as e:
            print(f"\n  [Piggyback Slice Retry] {e}", flush=True)
            time.sleep(1.0)
            return True

        if en >= total_bytes:
            nonlocal last_saved_model_sd
            try:
                remote_state = torch.load(
                    io.BytesIO(zlib.decompress(b"".join(ckpt_sync["parts"]))),
                    map_location="cpu",
                    weights_only=False,
                )
                last_saved_model_sd = remote_state["model_state_dict"]
                outer_done = int(meta["outer"])
                step_done = int(meta["step"])
                tokens_done = int(meta["tokens"])
                checkpoint_data = {
                    "model_state_dict": last_saved_model_sd,
                    "optimizer_state_dict": last_saved_opt_sd,
                    "scheduler_state_dict": None,
                    "epoch": ckpt_sync["epoch"],
                    "outer_batch_idx": max(outer_done, resume_outer_batch_idx),
                    "total_outer_batches": ckpt_sync["total_outer"],
                    "optimizer_step": max(step_done, optimizer_step),
                    "all_tokens": max(tokens_done, all_tokens_val),
                    "last_val_calc_tokens": trainer.last_val_calc_tokens,
                    "verification_loss": verification_loss,
                    "rng_states": None,
                    "wandb_run_id": ckpt_sync["w_id"],
                    "model_name": model_name,
                    "version": args.version,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                tmp_path = checkpoint_path + ".tmp"
                torch.save(checkpoint_data, tmp_path)
                os.replace(tmp_path, checkpoint_path)
                print(
                    f"\n  \033[36m[Mac Piggyback Sync]\033[0m Full weights for Outer {outer_done} (Step {step_done}) synced to Mac SSD!",
                    flush=True,
                )
            except Exception as e:
                print(f"\n  [Mac Piggyback Save Warning] {e}", flush=True)
            finally:
                ckpt_sync["active"] = False
                ckpt_sync["parts"] = []
        return True

    for epoch in range(start_epoch, t_args.num_epochs):
        epoch_loss = EpochObserver(1000)
        skip_until = resume_outer_batch_idx if epoch == start_epoch else 0

        it = enumerate(train_loader)
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preproc")

        def _preproc_and_compress(pack):
            ds = trainer.pre_processing(pack, progress)
            if len(ds) == 0:
                return 0, b""
            arr = ds.tensors[0].numpy()
            perm = np.random.permutation(len(arr))
            arr = arr[perm]
            num_minis = (len(arr) + batch_size - 1) // batch_size
            payload = _compress_u8_batch(arr)
            return num_minis, payload

        pending = None
        for pack_idx, pack in it:
            if pack_idx < skip_until:
                continue
            pending = (pack_idx, pool.submit(_preproc_and_compress, pack))
            break

        try:
            while pending is not None:
                pack_idx, fut = pending
                count = pack_idx + 1

                if args.override_lr is None and args.lr_anneal_outer > 0 and count >= args.lr_anneal_outer and t_args.lr_param > args.lr_anneal_val:
                    for attempt in range(5):
                        try:
                            rlr = session.post(f"{remote_url}/set_lr?lr={args.lr_anneal_val}", timeout=30)
                            rlr.raise_for_status()
                            break
                        except Exception:
                            time.sleep(1.0)
                    t_args.lr_param = args.lr_anneal_val
                    print(
                        f"\n  \033[33m[Auto LR Anneal]\033[0m Reached Outer {count} (>= {args.lr_anneal_outer}): "
                        f"Switched remote & local LR -> {args.lr_anneal_val:.2e}!",
                        flush=True,
                    )

                nxt = None
                for n_idx, n_pack in it:
                    nxt = (n_idx, pool.submit(_preproc_and_compress, n_pack))
                    break

                num_minis, payload = fut.result()
                pending = nxt

                if num_minis == 0:
                    continue

                t_outer_start = time.time()
                _upload_and_start_big_batch(session, remote_url, payload, batch_size)
                del payload

                cursor = 0
                mini_c = 0
                while True:
                    # GPU がバックグラウンド計算中の待機時間を利用して 256KB スライスを1つ回収（なければ 1.2秒 sleep）
                    if not _pull_one_ckpt_slice():
                        time.sleep(1.2)
                    pdata = None
                    for attempt in range(6):
                        try:
                            rp = session.get(f"{remote_url}/poll_big_batch?cursor={cursor}", timeout=30)
                            rp.raise_for_status()
                            pdata = rp.json()
                            break
                        except Exception as e:
                            if attempt == 5:
                                raise
                            print(f"\n  [Poll Retry {attempt+1}] {e}", flush=True)
                            time.sleep(2.0)

                    if pdata.get("error"):
                        raise RuntimeError(f"Colab GPU Worker Error: {pdata['error']}")

                    new_results = pdata.get("results", [])
                    cursor = int(pdata.get("next_cursor", cursor))

                    for m in new_results:
                        mini_c += 1
                        optimizer_step += 1
                        trainer.optimizer_steps = optimizer_step
                        chunk_cnt = int(m.get("batch_chunks", batch_size))
                        all_tokens_val += chunk_cnt * tokens_per_chunk
                        trainer.all_tokens.fill_(all_tokens_val)

                        trainer.criterion.last_metrics = m
                        epoch_loss.add(float(m["total"]))

                        trainer.view_logs(
                            epoch,
                            t_args.num_epochs,
                            count,
                            len(train_loader),
                            mini_c,
                            num_minis,
                            epoch_loss.get(),
                            t_args.lr_param,
                            verification_loss,
                            all_tokens_val,
                        )

                    if pdata.get("done") and mini_c >= num_minis:
                        break
                    if pdata.get("done") and len(new_results) == 0:
                        break

                # Outer Batch 完了時:
                # 1. Colab SSD (/content/MORTM5.{version}.checkpoint.pt) & マウント済みなら Google Drive に 0.2秒で即時保存
                # 2. Google Drive 保存が有効ならトンネル経由のチェックポイント転送を完全スキップ (0バイト)
                t_gpu_done = time.time()
                w_id = getattr(trainer.wandb.run, "id", wandb_id) if trainer.wandb is not None and trainer.wandb.run is not None else wandb_id
                need_new_buf = 0 if ckpt_sync["active"] else 1
                meta_info = None
                for attempt in range(6):
                    try:
                        r_save = session.post(
                            f"{remote_url}/save_colab_ckpt?outer={count}&tokens={all_tokens_val}"
                            f"&epoch={epoch}&total_outer={len(train_loader)}&wandb_id={w_id}&version={args.version}"
                            f"&refresh_buffer={need_new_buf}",
                            timeout=30,
                        )
                        r_save.raise_for_status()
                        meta_info = r_save.json()
                        break
                    except Exception as e:
                        if attempt == 5:
                            raise
                        time.sleep(2.0)

                resume_outer_batch_idx = count
                drive_saved = bool(meta_info and meta_info.get("drive_saved", False))

                if drive_saved:
                    ckpt_sync["active"] = False
                    ckpt_sync["parts"] = []
                    save_loc_str = "Colab SSD + Google Drive (0B tunnel transfer)"
                else:
                    save_loc_str = "Colab SSD"
                    if need_new_buf and meta_info is not None and int(meta_info.get("total_bytes", 0)) > 0:
                        ckpt_sync["active"] = True
                        ckpt_sync["meta"] = meta_info
                        ckpt_sync["epoch"] = epoch
                        ckpt_sync["total_outer"] = len(train_loader)
                        ckpt_sync["w_id"] = w_id
                        ckpt_sync["offset"] = 0
                        ckpt_sync["parts"] = []

                print(
                    f"\n  [Outer {count}/{len(train_loader)} Done in {t_gpu_done - t_outer_start:.1f}s] "
                    f"Saved on {save_loc_str} & starting Outer {count + 1} immediately! (Step {optimizer_step}, Tokens {all_tokens_val:,})",
                    flush=True,
                )

                if args.max_outer > 0 and count >= args.max_outer:
                    print(
                        f"\n\033[32m[Target Reached]\033[0m Completed Outer {count} (max_outer={args.max_outer}, Step {optimizer_step}, Tokens {all_tokens_val:,}). "
                        f"Waiting 3s for Drive background save to flush and stopping cleanly!",
                        flush=True,
                    )
                    time.sleep(3.0)
                    return

                gc.collect()

            resume_outer_batch_idx = 0
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
