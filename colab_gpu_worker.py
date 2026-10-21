"""
Google Colab Remote GPU Worker for MORTM5 (MidiVision)
"""
import io
import json
import zlib
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request, Response
from vector_quantize_pytorch import GroupedResidualVQ


# ============================================================
# 1. MORTM5 (MidiVision) Model & Loss Definitions (Exact Match)
# ============================================================
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int | Tuple[int, int] = 1, expand_ratio: int = 4, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        hidden_dim = in_channels * expand_ratio
        self.pwconv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x))))) + self.shortcut(x)


class ResNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int | Tuple[int, int] = 1, expand_ratio: int = 4, kernel_size: int = 7):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest") if scale_factor != 1 else nn.Identity()
        self.block = ResNetBlock(in_channels, out_channels, stride=1, expand_ratio=expand_ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.upsample(x))


class ResNetEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.track_size = args.track_size
        self.input_channel = self.track_size * 2
        self.first_channel = args.first_channel
        self.max_channel = args.max_channel
        self.mid_channel = max(self.first_channel, self.max_channel // 2)
        self.bottle_channel = getattr(args, "bottle_channel", 128)
        self.resnet = nn.Sequential(
            ResNetBlock(self.input_channel, self.first_channel, stride=1),
            ResNetBlock(self.first_channel, self.first_channel, stride=1),
            ResNetBlock(self.first_channel, self.mid_channel, stride=2),
            ResNetBlock(self.mid_channel, self.mid_channel, stride=1),
            ResNetBlock(self.mid_channel, self.max_channel, stride=(2, 1)),
            ResNetBlock(self.max_channel, self.max_channel, stride=1),
            ResNetBlock(self.max_channel, self.max_channel, stride=2),
            ResNetBlock(self.max_channel, self.max_channel, stride=1),
            ResNetBlock(self.max_channel, self.bottle_channel, stride=2),
            ResNetBlock(self.bottle_channel, self.bottle_channel, stride=1),
        )
        self.init_h, self.init_w = 8, 3
        self.w_out = nn.Linear(self.bottle_channel * self.init_h * self.init_w, args.encoder_wout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return self.w_out(x.reshape(x.shape[0], -1))


class ResNetDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.track_size = args.track_size
        self.output_channel = self.track_size * 2
        self.max_channel = args.max_channel
        self.first_channel = args.first_channel
        self.mid_channel = max(self.first_channel, self.max_channel // 2)
        self.bottle_channel = getattr(args, "bottle_channel", 128)
        self.roll_h, self.roll_w = args.roll_h, args.roll_w
        self.init_h, self.init_w = 8, 3
        self.w_in = nn.Linear(args.encoder_wout, self.bottle_channel * self.init_h * self.init_w)
        self.decoder = nn.Sequential(
            ResNetUpBlock(self.bottle_channel, self.max_channel, scale_factor=2),
            ResNetBlock(self.max_channel, self.max_channel, stride=1),
            ResNetUpBlock(self.max_channel, self.max_channel, scale_factor=2),
            ResNetBlock(self.max_channel, self.max_channel, stride=1),
            ResNetUpBlock(self.max_channel, self.mid_channel, scale_factor=(2, 1)),
            ResNetBlock(self.mid_channel, self.mid_channel, stride=1),
            ResNetUpBlock(self.mid_channel, self.first_channel, scale_factor=2),
            ResNetBlock(self.first_channel, self.first_channel, stride=1),
            ResNetBlock(self.first_channel, self.first_channel, stride=1),
            nn.Conv2d(self.first_channel, self.output_channel, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w_in(x).reshape(x.shape[0], self.bottle_channel, self.init_h, self.init_w)
        return self.decoder(x)


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


# ============================================================
# 2. FastAPI Remote GPU Worker
# ============================================================
app = FastAPI()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
STATE = {"model": None, "optimizer": None, "criterion": None, "step": 0, "lr": 4e-4}


@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(DEVICE)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0
    return {"status": "ok", "device": str(DEVICE), "gpu_name": gpu_name, "vram_gb": vram_gb, "step": STATE["step"]}


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
    import threading
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
    import datetime
    import os
    import shutil
    import threading
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


@app.get("/get_state_raw_chunk")
async def get_state_raw_chunk(request: Request):
    start = int(request.query_params.get("start", "0"))
    end = int(request.query_params.get("end", "2097152"))
    b = STATE.get("ckpt_bytes", b"")[start:end]
    return Response(content=b, media_type="application/octet-stream")


