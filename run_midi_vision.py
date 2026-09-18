import argparse
import datetime
import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader

from mortm.models.modules.config import MORTM5Args
from mortm.models.mortm5 import MidiVision
from mortm.train.config import AbstractTrainSet, TrainArgs
from mortm.train.train import train_custom, find_files
from mortm.utils.convert_pianoroll import midi_to_chunks
from mortm.utils.messager import _DefaultMessenger, Messenger
from mortm.models.modules.progress import _DefaultLearningProgress, LearningProgress


class MidiVisionLoss(nn.Module):
    """MORTM5 (MidiVision) ピアノロール再構成損失
    
    1. Note-Active 損失 (音符の有無・復元率極大化):
       - 空間の 98% 以上が無音であるため、通常の Loss では音符が欠落する問題を防止
       - 重み付き BCE (pos_weight=30.0) + Dice Loss (Soft F1 最大化)
    2. Velocity 損失 (音量):
       - 発音箇所 (target_vel > 0) のみを対象とした Masked Smooth L1 Loss
    3. ResidualVQ Commitment 損失:
       - コードブック学習用正則化
    4. 復元率評価メトリクス:
       - Note Recall (再現率), Note Precision (適合率), Note F1 スコア, Velocity MAE
    """
    def __init__(
        self,
        pos_weight: float = 30.0,
        dice_weight: float = 1.0,
        vel_weight: float = 5.0,
        commit_weight: float = 0.25,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.vel_weight = vel_weight
        self.commit_weight = commit_weight
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))
        self.last_metrics = {}

    def forward(
        self,
        decoded: torch.Tensor,
        original: torch.Tensor,
        commit_loss: torch.Tensor,
    ) -> torch.Tensor:
        # decoded: [B, 8, 128, 24] (logits)
        # original: [B, 8, 128, 24] ([0.0, 1.0])
        # commit_loss: スカラーテンソル

        # 1. Note-Active (チャンネル 1, 3, 5, 7) - float32 で計算してアンダーフローを防止
        pred_active_logits = decoded[:, 1::2].float()
        target_active = original[:, 1::2].float()

        pos_weight = self.pos_weight_tensor.to(decoded.device)
        bce = F.binary_cross_entropy_with_logits(
            pred_active_logits, target_active, pos_weight=pos_weight
        )

        prob = torch.sigmoid(pred_active_logits)
        intersection = (prob * target_active).sum(dim=(-2, -1))
        cardinality = prob.sum(dim=(-2, -1)) + target_active.sum(dim=(-2, -1))
        dice = 1.0 - (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
        dice_loss = dice.mean()

        loss_active = bce + self.dice_weight * dice_loss

        # 2. Velocity (チャンネル 0, 2, 4, 6)
        pred_vel_logits = decoded[:, 0::2].float()
        pred_vel = torch.sigmoid(pred_vel_logits)
        target_vel = original[:, 0::2].float()

        vel_mask = (target_vel > 0.0).float()
        vel_diff = F.smooth_l1_loss(pred_vel, target_vel, reduction="none") * vel_mask
        mask_sum = vel_mask.sum()
        if mask_sum > 0:
            loss_vel = vel_diff.sum() / (mask_sum + 1e-5)
        else:
            loss_vel = vel_diff.sum() * 0.0

        # 3. RVQ Commitment
        commit_scalar = commit_loss.mean() if isinstance(commit_loss, torch.Tensor) else commit_loss
        loss_commit = commit_scalar * self.commit_weight

        # 4. Note F1 メトリクス計算
        with torch.no_grad():
            pred_bin = (prob > 0.5).float()
            tp = (pred_bin * target_active).sum().item()
            fp = (pred_bin * (1.0 - target_active)).sum().item()
            fn = ((1.0 - pred_bin) * target_active).sum().item()
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall + 1e-8)) if (precision + recall) > 0 else 0.0
            vel_mae = ((pred_vel - target_vel).abs() * vel_mask).sum().item() / (mask_sum.item() + 1e-5) if mask_sum > 0 else 0.0

        total_loss = loss_active + self.vel_weight * loss_vel + loss_commit

        self.last_metrics = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "vel_mae": vel_mae,
            "bce": bce.item(),
            "dice": dice_loss.item(),
            "vel_loss": loss_vel.item(),
            "commit": commit_scalar.item() if isinstance(commit_scalar, torch.Tensor) else float(commit_scalar),
            # 瞬時の総損失。学習ループが渡してくる loss は EpochObserver(1000) による
            # 直近1000stepの移動平均であり、隣に並ぶ BCE/Dice/Commit(瞬時値)とは意味が違う。
            # 両方を表示して「平均が遅れて追従しているだけ」を発散と誤読しないようにする。
            "total": total_loss.item(),
        }
        return total_loss


def pitch_shift_chunks(chunks: np.ndarray, shift: int) -> np.ndarray:
    """ドラム(Ch 0, 1)を除外し、楽器トラック(Ch 2~7)のみピッチ軸方向にシフトする。
    
    chunks: [N, 8, 128, 24] (Pitch=128, Time=24)
    shift: -6 ~ +5 (半音)
    はみ出し部分はゼロクリア（ループによる異音混入を防止）
    """
    if shift == 0:
        return chunks

    shifted = chunks.copy()
    # チャンネル 2:8 が楽器パート (Ch 0, 1 はドラムのため不変)
    if shift > 0:
        shifted[:, 2:, shift:, :] = chunks[:, 2:, :-shift, :]
        shifted[:, 2:, :shift, :] = 0.0
    else:
        s = abs(shift)
        shifted[:, 2:, :-s, :] = chunks[:, 2:, s:, :]
        shifted[:, 2:, -s:, :] = 0.0
    return shifted


class MidiVisionTrainSet(AbstractTrainSet):
    """MORTM5 (MidiVision) 専用 Trainer (AbstractTrainSet 準拠)"""

    def __init__(
        self,
        args: MORTM5Args,
        t_args: TrainArgs,
        progress: LearningProgress,
        load_directory: Optional[str] = None,
        enable_shift: bool = True,
        use_wandb: bool = True,
        project_name: str = "MORTM5_MidiVision",
        run_name: Optional[str] = None,
    ):
        self.args = args
        self.t_args = t_args
        self.progress = progress
        self.enable_shift = enable_shift

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # 1. モデル構築
        model = MidiVision(args)
        if load_directory is not None and os.path.exists(load_directory):
            sd = torch.load(load_directory, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            if self.local_rank == 0:
                print(f"[MidiVision] Loaded weights from {load_directory}")

        model = model.to(device)

        if dist.is_initialized():
            self.model = DDP(model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
        else:
            self.model = model

        # 2. オプティマイザ & 損失関数
        lr = t_args.lr_param if t_args.lr_param is not None else 3e-4
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=1e-2,
        )
        criterion = MidiVisionLoss().to(device)

        super().__init__(
            criterion=criterion,
            optimizer=optimizer,
            t_args=t_args,
            m_args=args,
        )
        self.device = device
        self.all_tokens = torch.tensor(0, device=device, dtype=torch.long)

        # 3. Weights & Biases (wandb) 初期化 (Rank 0 のみ)
        self.wandb = None
        if self.local_rank == 0 and use_wandb:
            try:
                import wandb
                wandb.init(
                    project=project_name,
                    name=run_name,
                    config={
                        "model": vars(args),
                        "train": vars(t_args),
                    },
                    reinit=True,
                )
                self.wandb = wandb
                print(f"[MidiVision] wandb 連携完了: project='{project_name}', run='{run_name}'")
            except Exception as e:
                print(f"[MidiVision] wandb 無効化 ({e})。TensorBoard のみで記録を継続します。")
                self.wandb = None
        self.global_step = 0

    def get_synced_tokens(self) -> int:
        if dist.is_initialized() and dist.get_world_size() > 1:
            sync_tokens = self.all_tokens.detach().clone().to(self.device)
            dist.all_reduce(sync_tokens, op=dist.ReduceOp.SUM)
            return int(sync_tokens.item())
        return int(self.all_tokens.item())

    def pre_processing(self, pack, progress):
        """Outer Loader から渡された MIDI パス群をオンザフライでパース・チャンク化・転調"""
        if isinstance(pack, (list, tuple)) and len(pack) > 0 and isinstance(pack[0], (list, tuple)):
            midi_paths = pack[0]
        elif isinstance(pack, (list, tuple)):
            midi_paths = pack
        else:
            midi_paths = [pack]

        collected_chunks = []
        for p in midi_paths:
            p_str = str(p)
            if not (p_str.lower().endswith('.mid') or p_str.lower().endswith('.midi')):
                continue
            try:
                # [chunks, 8, 24, 128]
                chunks, _ = midi_to_chunks(p_str)
                if chunks.shape[0] == 0:
                    continue

                # 軸転置: [N, 8, 24, 128] -> [N, 8, 128, 24] (Pitch 128, Time 24)
                chunks = np.transpose(chunks, (0, 1, 3, 2))

                # float32 -> uint8 に可逆圧縮してから積む。
                # convert_pianoroll は velocity を velocity/127.0 で格納しているため、
                # 127倍すれば元の MIDI velocity(整数 0-127)に厳密に戻る。
                # active は {0,1} なので 127倍して {0,127} に揃え、GPU 側で一律 /127 する。
                # 効果: 1 Outer バッチのメモリが 27.1GB -> 6.8GB。
                # (実測で RAM が 62.2GB/62.5GB まで張り付き、空きが 0.3GB まで枯渇していた)
                chunks = np.rint(chunks * 127.0).astype(np.uint8)

                # ドラム除外ランダム転調（uint8 のまま実施するのでコピー量も 1/4）
                if self.enable_shift:
                    shift = random.randint(-6, 5)
                    chunks = pitch_shift_chunks(chunks, shift)

                collected_chunks.append(chunks)
            except Exception:
                # 破損ファイル等はスキップ
                continue

        if len(collected_chunks) == 0:
            # DDP のデッドロックを防ぐため、空の場合はゼロチャンクを1つ生成
            all_chunks = np.zeros((1, 8, self.args.roll_h, self.args.roll_w), dtype=np.uint8)
        else:
            all_chunks = np.concatenate(collected_chunks, axis=0)

        # float 変換は GPU 側(epoch_fc)で行う。CPU 側は uint8 のまま保持してメモリを節約する。
        tensor_data = torch.from_numpy(all_chunks)
        return TensorDataset(tensor_data)

    def epoch_fc(self, model, pack, progress):
        """Inner Loader の 1 バッチ順伝播"""
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        # pre_processing は uint8 で保持しているので GPU 上で float に戻す。
        # velocity/active とも 127倍で格納されているため、一律 /127 で元の [0,1] に厳密復元される。
        x = pack[0].to(device, non_blocking=True)  # [B, 8, 128, 24] uint8
        if x.dtype == torch.uint8:
            x = x.float().div_(127.0)
        else:
            x = x.float()
        if model.training:
            self.all_tokens += x.shape[0] * self.args.encoder_layer
        reconstructed, tokens, commit_loss = model(x)
        return reconstructed, x, commit_loss

    def optional_logging(self, val_loss, step):
        if self.local_rank == 0 and self.wandb is not None:
            self.wandb.log({
                "val/loss": val_loss,
                "step": step,
            })

    def view_logs(self, *args, **kwargs):
        """復元率 (Note F1) と各損失項のフォーマット出力
        self_turing からの呼び出し:
          (epoch, sum_epoch, count, all_pac, mini_seq_count, mini_seq_pac, loss, lr, verif_loss, tokens)
        または dry_run / 直接呼び出し:
          (count, loss, lr, is_val=False)
        """
        metrics = self.criterion.last_metrics
        f1 = metrics.get("f1", 0.0) * 100.0
        rec = metrics.get("recall", 0.0) * 100.0
        prec = metrics.get("precision", 0.0) * 100.0
        vel_mae = metrics.get("vel_mae", 0.0)
        commit = metrics.get("commit", 0.0)
        bce = metrics.get("bce", 0.0)
        dice = metrics.get("dice", 0.0)
        vel = metrics.get("vel_loss", 0.0)
        inst = metrics.get("total", 0.0)

        if len(args) >= 10:
            epoch, sum_epoch, seq_count, all_pac, mini_c, mini_pac, loss, lr, verif_loss, tokens = args[:10]
            lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)
            self.global_step += 1

            # 1. コンソール表示は毎ステップリアルタイムで滑らかに更新
            if self.local_rank == 0:
                msg = (
                    f"\rEp {epoch+1}/{sum_epoch} "
                    f"Outer {seq_count}/{all_pac} Inner {mini_c}/{mini_pac} | "
                    f"Loss: {loss:.4f}(avg1k) {inst:.4f}(now) "
                    f"(BCE:{bce:.3f} Dice:{dice:.3f} Vel:{vel:.4f} Commit:{commit:.3f}) | "
                    f"Note F1: {f1:.1f}% (R:{rec:.1f}% P:{prec:.1f}%) | "
                    f"Vel MAE: {vel_mae:.4f} | Val: {verif_loss:.4f} | LR: {lr_str}"
                )
                print(msg, end="", flush=True)
                if mini_c == mini_pac:
                    print()

                # 2. wandb の書き込みは 500 ステップごと、およびビッグバッチ完了時のみに限定
                should_log_wandb = (self.global_step % 500 == 0) or (mini_c == mini_pac)
                if should_log_wandb and self.wandb is not None:
                    self.wandb.log({
                        "train/loss": loss,
                        "train/loss_now": inst,
                        "train/vel_loss": vel,
                        "train/bce": bce,
                        "train/dice": dice,
                        "train/commit": commit,
                        "train/note_f1": f1,
                        "train/recall": rec,
                        "train/precision": prec,
                        "train/vel_mae": vel_mae,
                        "train/lr": lr if isinstance(lr, (int, float)) else 0.0,
                        "train/tokens": tokens,
                        "train/epoch": epoch + 1,
                        "outer_batch": seq_count,
                        "step": self.global_step,
                    })
        else:
            count = args[0] if len(args) > 0 else 0
            loss = args[1] if len(args) > 1 else 0.0
            lr = args[2] if len(args) > 2 else 0.0
            is_val = kwargs.get("is_val", False)
            prefix = "[VAL]" if is_val else f"[Step {count}]"
            msg = (
                f"{prefix} Loss: {loss:.4f} (BCE:{bce:.3f} Dice:{dice:.3f} "
                f"Vel:{vel:.4f} Commit:{commit:.3f}) | "
                f"Note F1: {f1:.1f}% (R:{rec:.1f}% P:{prec:.1f}%) | "
                f"Vel MAE: {vel_mae:.4f} | LR: {lr:.2e}"
            )
            if self.local_rank == 0:
                print(msg)


def main():
    parser = argparse.ArgumentParser(description="MORTM5 (MidiVision) DDP Training")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/mortm5/default.json",
        help="Model config path",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/mortm5/default.json",
        help="Train config path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/generate",
        help="Root directory containing MIDI files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/models/mortm5/",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="MORTM5_MidiVision",
        help="Version / run label",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Pretrained weights path to resume",
    )
    parser.add_argument(
        "--no_shift",
        action="store_true",
        help="Disable pitch shifting augmentation",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Enable wandb logging (default: True, auto-disabled if not logged in)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_false",
        dest="use_wandb",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="MORTM5_MidiVision",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default=None,
        help="Optional eval dataset json path",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run single mini-batch self-test and exit",
    )

    args = parser.parse_args()

    # CUDA_LAUNCH_BLOCKING の残存を除去して最大スループットを確保
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # DDP 環境下では trainer 構築前にプロセスグループを初期化 (DDP化およびコードブック同期のため)
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        backend = os.environ.get("MORTM_DDP_BACKEND", "nccl")
        device_id = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        dist.init_process_group(backend=backend, device_id=device_id)

    # Dry run モード: 単体動作検証
    if args.dry_run:
        print("=== Running Dry-Run Self-Test ===")
        m_args = MORTM5Args(args.model_config)
        t_args = TrainArgs(args.train_config)
        progress = _DefaultLearningProgress()

        trainer = MidiVisionTrainSet(
            args=m_args,
            t_args=t_args,
            progress=progress,
            load_directory=args.load_model,
            enable_shift=not args.no_shift,
            use_wandb=False,
        )

        directory, filenames = find_files(args.data_dir, '.mid')
        print(f"Found {len(filenames)} MIDI files in {args.data_dir}")
        if len(filenames) == 0:
            print("[Warning] No MIDI files found, using synthetic batch.")
            dummy_x = torch.zeros((4, 8, m_args.roll_h, m_args.roll_w))
            dummy_x[:, 1, 60, :12] = 1.0  # Dummy note
            dummy_x[:, 0, 60, :12] = 0.8  # Dummy velocity
            recon, tokens, commit = trainer.model(dummy_x)
            loss = trainer.criterion(recon, dummy_x, commit)
            loss.backward()
            trainer.view_logs(1, loss.item(), 3e-4)
            print("Dry run passed successfully with synthetic batch!")
            return

        test_pack = [_paths_from_dir_files(directory[:4], filenames[:4])]
        ds = trainer.pre_processing(test_pack, progress)
        print(f"Preprocessed chunks: {len(ds)}, item shape: {ds[0][0].shape}")
        loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False)
        batch = next(iter(loader))

        recon, orig, commit = trainer.epoch_fc(trainer.model, batch, progress)
        loss = trainer.criterion(recon, orig, commit)
        loss.backward()
        trainer.view_logs(1, loss.item(), 3e-4)
        print("=== Dry-Run Self-Test Finished Successfully! ===")
        return

    # 通常の DDP / 学習実行
    os.makedirs(args.save_dir, exist_ok=True)
    m_args = MORTM5Args(args.model_config)
    t_args = TrainArgs(args.train_config)
    progress = _DefaultLearningProgress()

    trainer = MidiVisionTrainSet(
        args=m_args,
        t_args=t_args,
        progress=progress,
        load_directory=args.load_model,
        enable_shift=not args.no_shift,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.version,
    )

    # データソースの準備: .json またはディレクトリからマニフェストを作成
    if args.data_dir.endswith(".json"):
        manifest_path = args.data_dir
    else:
        manifest_path = os.path.join(args.save_dir, "dataset_manifest.json")
        if trainer.local_rank == 0:
            directory, filenames = find_files(args.data_dir, '.mid')
            all_midi_paths = [os.path.join(d, f) for d, f in zip(directory, filenames)]
            with open(manifest_path, "w") as f:
                json.dump(all_midi_paths, f)
            print(f"Found {len(all_midi_paths)} MIDI files for training in {args.data_dir}")

        if dist.is_initialized():
            dist.barrier()

    train_custom(
        trainer=trainer,
        t_args=t_args,
        root_directory=manifest_path,
        save_directory=args.save_dir,
        version=args.version,
        eval_list_json=args.eval_json,
    )

    if trainer.local_rank == 0 and trainer.wandb is not None:
        trainer.wandb.finish()


def _paths_from_dir_files(directory, filenames):
    return [os.path.join(d, f) for d, f in zip(directory, filenames)]


if __name__ == "__main__":
    main()
