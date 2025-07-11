import copy

import torch
from torch import Tensor
from  torch.nn import *
from torch.optim import Adam

from mortm.models.bertm import BERTM, Critic
from mortm.models.modules.config import MORTMArgs

from abc import abstractmethod

class AbstractReinforcementLearning:
    def __init__(self, actor, critic, args: MORTMArgs, progress,
        actor_lr: float = 1e-5,
        critic_lr: float = 5e-5,
        eps_clip: float = 0.2,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        kl_coeff: float = 0.2):

        self.actor = actor
        self.critic = critic
        self.args = args
        self.progress = progress

        self.ref_model = copy.deepcopy(self.actor).to(self.progress.get_device())
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        # オプティマイザ
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # PPOハイパーパラメータ
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.kl_coeff = kl_coeff

    def run_train(self, epoch):
        for _ in range(epoch):
            pass

    def compute_advantages_and_returns(self, rewards, values, masks):
        """Generalized Advantage Estimation (GAE) を計算"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(rewards.size(1))):
            # マスクされていないタイムステップでのみ計算
            mask = masks[:, t]

            # 価値の差分 (TD誤差)
            delta = rewards[:, t] + self.gamma * values[:, t + 1] * mask - values[:, t]

            # GAE
            last_advantage = delta + self.gamma * self.lambda_gae * last_advantage * mask
            advantages[:, t] = last_advantage

        returns = values[:, :-1] + advantages
        return advantages, returns

    @abstractmethod
    def generate(self, actor) -> [Tensor, Tensor]:
        pass



