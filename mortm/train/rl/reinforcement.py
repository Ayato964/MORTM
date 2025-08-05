import json

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from mortm.train.train import AbstractTrainSet, TrainArgs
from mortm.train.utils.loss import RLDFLoss
from mortm.train.noam import noam_lr
from mortm.models.modules.config import MORTMArgs
from mortm.models.mortm import MORTM
from mortm.models.bertm import BERTM, ActorCritic
from mortm.train.datasets import MORTM_SEQDataset, PPODataset
from mortm.de_convert import ct_token_to_midi
from mortm.train.tokenizer import TO_MUSIC, Tokenizer, get_token_converter

def pad_variable_sequences(sequences: list, padding_value: int = 0, device='cpu', dtype=torch.long):
    """
    可変長のシーケンスのリストをパディングし、Tensorとアテンションマスクを返す。

    Args:
        sequences (list[list[int]] or list[np.ndarray]): 可変長のシーケンスのリスト。
        padding_value (int): パディングに使用するトークンID。通常は0。
        device: 作成するテンソルのデバイス。

    Returns:
        padded_tensor (torch.Tensor): パディングされたシーケンスのテンソル。
        attention_mask (torch.Tensor): 対応するアテンションマスク。
    """
    # 1. バッチ内の最大長を取得
    max_len = max(len(seq) for seq in sequences)

    # 2. パディング値で埋めたテンソルを作成
    #    形状: (バッチサイズ, 最大長)
    padded_tensor = torch.full((len(sequences), max_len), padding_value, dtype=dtype, device=device)

    # 3. アテンションマスク用のテンソルを作成 (0で初期化)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)

    # 4. 各シーケンスをテンソルにコピーし、マスクを更新
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_tensor[i, :length] = torch.tensor(seq, dtype=dtype)
        attention_mask[i, :length] = True # 本当のトークンの位置をTrueにする

    return padded_tensor, attention_mask


def compute_advantages_and_returns(rewards, values, gamma=0.99, lambda_=0.95):
    last_gae_lam = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(rewards.size(1))):
        next_values = values[:, t + 1] if t < rewards.size(1) - 1 else 0.0
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        last_gae_lam = delta + gamma * lambda_ * last_gae_lam
        advantages[:, t] = last_gae_lam
    returns = advantages + values
    return advantages, returns


class RLTrainerArgs(TrainArgs):
    def __init__(self, json_directory: str):
        super().__init__(json_directory)
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.rl_gamma = data["rl_gamma"] if data.get('rl_gamma') else 0.2
            self.ppo_epoch = data["ppo_epoch"] if data.get('ppo_epoch') else 4
            self.ppo_batch = data["ppo_batch"] if data.get('ppo_batch') else 8


class RLDF(AbstractTrainSet):

    def __init__(self, t_args: RLTrainerArgs, args: MORTMArgs, b_args: MORTMArgs, progress, load_mortm_directory, load_bertm_directory, use_lora=False):
        self.args = args
        self.t_args = t_args
        self.args.use_lora = use_lora
        ############ LOAD MODEL ###########################
        self.base_model = MORTM(self.args, progress)
        self.base_model.to(progress.get_device())
        self.base_model.load_state_dict(torch.load(load_mortm_directory), strict=True)
        for param in self.base_model.parameters():
            param.requires_grad = False

        ########### LOAD REWARD MODEL ####################
        self.bertm = BERTM(args=b_args, progress=progress)
        self.actor_critic = ActorCritic(args=self.args, progress=progress)
        self.bertm.load_state_dict(torch.load(load_bertm_directory), strict=True)
        self.actor_critic.load_state_dict(torch.load(load_mortm_directory), strict=False)
        self.actor_critic.to(progress.get_device())
        self.bertm.to(progress.get_device())
        for param in self.bertm.parameters():
            param.requires_grad = False

        adam = AdamW(self.actor_critic.parameters(), lr=5e-1)
        sc = LambdaLR(optimizer= adam, lr_lambda=noam_lr(d_model=args.d_model, warmup_steps=t_args.warmup_steps))
        sc.step()
        super().__init__(criterion=RLDFLoss(clip_epsilon=0.2, vf_coeff=0.5),
                         optimizer=adam,
                         scheduler=sc)
        self.model = self.actor_critic

    def epoch_fc(self, model, pack, progress):
        with torch.no_grad():
            src = pack
            self.model.eval()
            processed_src_list = []
            for i in range(len(src)):
                src_single = src[i]
                indices = (src_single == 8).nonzero(as_tuple=True)[0]

                if indices is not None and len(indices) > 4:
                    processed_src_list.append(src_single[:indices[4]])
                else:
                    processed_src_list.append(src_single)

            # 3. [変更なし] 長さが異なるシーケンスのリストをパディングし、新しいバッチを作成
            src_padded = pad_sequence(processed_src_list, batch_first=True, padding_value=0).to(progress.get_device())

            out, value, probs = self.model.eval_seq(src_padded)

            sequences, padding_mask = pad_variable_sequences(out, device=progress.get_device(), dtype=torch.long)
            value, _ = pad_variable_sequences(value, device=progress.get_device(), dtype=torch.float32)
            probs, _ = pad_variable_sequences(probs, device=progress.get_device(), dtype=torch.float32)


        return (sequences, padding_mask, value, probs)

    def pre_processing(self, pack, progress):
        dt: DataLoader = pack
        mini_dataset = MORTM_SEQDataset(progress, self.args.position_length, self.args.min_length)
        for d in dt:
            np_load_data = np.load(d, allow_pickle=True)
            mini_dataset.add_data(np_load_data)

        return mini_dataset

    def backward(self, accumulation_steps, is_step, progress, lr_param, *args):
        sequence, padding_mask, value, probs = args
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reward_base = self.bertm(x=sequence, padding_mask=padding_mask)
        reward_score = 1 - F.sigmoid(reward_base)
        print(f"平均スコア：{reward_score.mean(), reward_score.max(), reward_score.std()}")

        with torch.no_grad():

            ref_log_probs = self.base_model.get_log_probs(sequence, padding_mask=padding_mask)
            kl_penalty = probs - ref_log_probs
            final_rewards = reward_score - self.t_args.rl_gamma * kl_penalty
            print(f"最終スコア：{final_rewards.mean(), final_rewards.max(), final_rewards.std()}")

            advantages, returns = compute_advantages_and_returns(final_rewards, value, gamma=0.99, lambda_=0.95)


        # --- 2. PPO最適化ループ ---
        self.actor_critic.train()
        ppo_dataset = PPODataset(sequence, probs, value, advantages, returns)
        ppo_dataloader = DataLoader(ppo_dataset, batch_size=self.t_args.ppo_batch, shuffle=True)

        total_loss_for_log = 0
        for _ in range(self.t_args.ppo_epoch):
            for mini_batch in ppo_dataloader:
                # ミニバッチのデータを取得
                mb_sequences, mb_old_log_probs, mb_values, mb_advantages, mb_returns = mini_batch

                """
                self.actor_critic.eval() # 評価モードで勾配計算をオフに
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        recalculated_log_probs, _ = self.actor_critic.evaluate_actions(mb_sequences, (mb_sequences != 0))

                # 2つのlog_probsの差を計算
                alignment_error = (recalculated_log_probs - mb_old_log_probs).abs().sum()
                print(f"\n[ALIGNMENT TEST] Sum of absolute difference: {alignment_error.item():.8f}\n")
                self.actor_critic.train() # 必ず訓練モードに戻す
                """
                mb_padding = (mb_sequences != 0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    new_log_probs, new_values = self.actor_critic.evaluate_actions(mb_sequences, mb_padding)
                    new_values: Tensor

                #print(f"new_value:{new_values.mean(), new_values.std(), new_values.max()}")
                #print(f"old_value:{mb_values.mean(), mb_values.std(), mb_values.max()}")
                #print(f"new_probs            | mean: {new_log_probs.mean():.4f}, std: {new_log_probs.std():.4f}, max: {new_log_probs.max():.4f}, min: {new_log_probs.min():.4f}")
                #print(f"old_probs            | mean: {mb_old_log_probs.mean():.4f}, std: {mb_old_log_probs.std():.4f}, max: {mb_old_log_probs.max():.4f}, min: {mb_old_log_probs.min():.4f}")

                # 損失を計算
                loss = self.criterion(
                    new_log_probs=new_log_probs,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    new_values=new_values,
                    old_values=mb_values,
                    returns=mb_returns
                )

                # 勾配計算と更新
                loss.backward()
                progress.step_optimizer(self.optimizer, self.model, accumulation_steps)
                total_loss_for_log += loss.item()
            if lr_param is None:
                self.scheduler.step()

        # ログ記録用の平均損失を返す
        avg_loss = total_loss_for_log / (self.t_args.ppo_epoch * len(ppo_dataloader))

        return torch.tensor(avg_loss)

    def get_eval_loss(self, *eval):
        sequence, padding_mask, value, probs = eval
        reward_base = self.bertm(x=sequence, padding_mask=padding_mask)
        reward_score: Tensor = 1 - reward_base

        return reward_score.sum() / len(reward_score)

