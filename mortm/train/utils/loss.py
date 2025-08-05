from torch import nn, Tensor
import torch

import torch.nn.functional as F

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