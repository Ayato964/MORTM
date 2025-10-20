from torch import nn, Tensor
import torch

import torch.nn.functional as F


class MusicEntropyLoss(nn.Module):
    """
    VAEの損失を計算するカスタム損失関数。
    - 復元誤差:
      - 音高 (偶数ch): バイナリクロスエントロピー
      - ベロシティ (奇数ch): 平均二乗誤差
    - 正則化項:
      - KLダイバージェンス
    """
    def __init__(self, beta: float = 1.0, pitch_weight: float = 1.0, velocity_weight: float = 1.0):
        """
        Args:
            beta (float): KLダイバージェンスの重み (β-VAE)
            pitch_weight (float): 音高(BCE)損失の重み
            velocity_weight (float): ベロシティ(MSE)損失の重み
        """
        super().__init__()
        self.beta = beta
        self.pitch_weight = pitch_weight
        self.velocity_weight = velocity_weight

    def forward(self, decoded: torch.Tensor, original: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:

        # 1. チャンネルを音高(pitch)とベロシティ(velocity)に分離
        original_pitch = original[:, 0::2, :, :]
        original_velocity = original[:, 1::2, :, :]

        decoded_pitch_logits = decoded[:, 0::2, :, :]
        decoded_velocity_logits = decoded[:, 1::2, :, :]

        loss_pitch_bce = F.binary_cross_entropy_with_logits(
            decoded_pitch_logits, original_pitch, reduction='sum'
        )

        # ベロシティの損失: 平均二乗誤差
        loss_velocity_mse = F.mse_loss(
            torch.sigmoid(decoded_velocity_logits), original_velocity, reduction='sum'
        )

        # KLダイバージェンス
        loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 3. 重みを付けて全ての損失を合計する
        total_loss = (self.pitch_weight * loss_pitch_bce +
                      self.velocity_weight * loss_velocity_mse +
                      self.beta * loss_kl)

        return total_loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)

        cross_loss = self.cross_entropy(inputs, targets)
        cross_loss = cross_loss * mask

        return cross_loss.sum() / (mask.sum() + 1e-12)


class RLDFLoss(nn.Module):
    def __init__(self, clip_epsilon: float = 0.2, vf_coeff: float = 0.5):
        super(RLDFLoss, self).__init__()
        self.clip_epsilon = clip_epsilon
        self.vf_coeff = vf_coeff

    def forward(self, new_log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, new_values: Tensor, old_values: Tensor, returns: Tensor):

        # --- デバッグ出力：入力値の確認 ---
        """
        print("\n--- 損失計算 デバッグ開始 ---")
        print(f"advantages (raw) | mean: {advantages.mean():.4f}, std: {advantages.std():.4f}, max: {advantages.max():.4f}, min: {advantages.min():.4f}")
        print(f"old_values       | mean: {old_values.mean():.4f}, std: {old_values.std():.4f}, max: {old_values.max():.4f}, min: {old_values.min():.4f}")
        """

        # アドバンテージを正規化（学習を安定させるためのテクニック）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #print(f"advantages (norm)  | mean: {advantages.mean():.4f}, std: {advantages.std():.4f}, max: {advantages.max():.4f}, min: {advantages.min():.4f}")

        # --- 方策損失 (Policy Loss) の計算 ---
        # 新旧の方策の確率比を計算
        ratio = torch.exp(new_log_probs - old_log_probs)
        #print(f"ratio            | mean: {ratio.mean():.4f}, std: {ratio.std():.4f}, max: {ratio.max():.4f}, min: {ratio.min():.4f}")

        # クリッピングされていない目的関数
        surr1 = ratio * advantages
        # クリッピングされた目的関数
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # --- 価値損失 (Value Loss) の計算 ---
        # PPOの論文に則ったValueのクリッピング
        values_clipped = old_values + torch.clamp(
            new_values - old_values, -self.clip_epsilon, self.clip_epsilon
        )

        # print(f"Values (new)     | mean: {new_values.mean():.4f}")
        # print(f"Returns          | mean: {returns.mean():.4f}, std: {returns.std():.4f}, max: {returns.max():.4f}, min: {returns.min():.4f}")

        vf_loss_unclipped = F.mse_loss(new_values, returns)
        vf_loss_clipped = F.mse_loss(values_clipped, returns)

        value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)

        # --- デバッグ出力：最終損失の確認 ---
        """
        print("---")
        print(f"Policy Loss: {policy_loss.item():.4f}")
        print(f"Value Loss (unclipped): {vf_loss_unclipped.item():.4f}")
        print(f"Value Loss (clipped):   {vf_loss_clipped.item():.4f}")
        print(f"Value Loss (final):     {value_loss.item():.4f}")
        print("---------------------------\n")
        """

        # --- 合計損失 ---
        total_loss = policy_loss + self.vf_coeff * value_loss

        return total_loss