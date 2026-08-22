"""CARL: Critique-Augmented Reinforcement Learning for MORTM.

コンセプト: **MORTM に MORTM を評価させて成長させる**。
別アーキテクチャの報酬モデル(旧 RLDF の BERTM 等)を用意せず、同一の MORTM 系列モデルを
2つの役割で使う自己批評(self-critique)型の強化学習:

  - **Actor(方策)**: 学習対象の MORTM。プロンプト(タスク意味ブロック列)から系列を生成する。
  - **Critic(評価器)**: MORTM(通常は凍結した参照重み、または分析方向を持つチェックポイント)。
    Actor の生成物を **分析方向 p(META|音楽) で読み** 、指定 META との整合や生成物の尤度から
    スカラー報酬を与える。Any-Order 事前学習により同一チェックポイントが生成と分析の
    両方向を供給できるため、外部報酬モデルなしに自己評価が閉じる。

学習ループ側(`mortm.train.train._train` / `self_turing`)から見た本クラスの位置づけは
Pre-train(`MORTMTrainSet`)や SFT と同一である。すなわち `AbstractTrainSet` の契約
(`pre_processing` / `epoch_fc` / `backward` / `view_logs`)を満たす限り、既存の
分散学習・チェックポイント・ロギング基盤をそのまま再利用できる。

本モジュールは骨組み(インタフェース確定)であり、各メソッドの中身は未実装。
"""

from abc import abstractmethod
from typing import Optional

import torch
from torch import nn, Tensor

from ..config import AbstractTrainSet, TrainArgs


class CARLArgs(TrainArgs):
    """CARL 固有のハイパーパラメータ。

    TrainArgs(batch_size / accumulation_steps / scheduler / lr_param ...)に、
    ロールアウトと報酬整形・方策更新の設定を追加する。
    """

    def __init__(self, json_directory: str):
        super().__init__(json_directory)
        raise NotImplementedError("CARLArgs is not implemented.")


class CARL(AbstractTrainSet):
    """MORTM の自己批評強化学習トレーナ。

    役割分担:
      - `self.model`      : Actor(学習対象・DDPラップ)。`AbstractTrainSet` の契約上必須。
      - `self.critic`     : Critic(評価器としての MORTM。通常は凍結)。
      - `self.ref_model`  : 参照方策(KL 正則化用。方策が初期分布から離れすぎるのを防ぐ)。

    学習の 1 ステップ(想定):
      1. `pre_processing` でプロンプト(タスク意味ブロック列)のミニデータセットを作る。
      2. `rollout` で Actor から系列をサンプリングする。
      3. `critique` で Critic に生成物を評価させ、スカラー報酬を得る。
      4. `compute_reward` で報酬を整形する(KL ペナルティ・長さ正規化など)。
      5. `epoch_fc` が方策勾配の材料を返し、`backward` が更新する。
    """

    def __init__(self, t_args: "CARLArgs", args, progress, **kwargs):
        raise NotImplementedError("CARL.__init__ is not implemented.")

    # ------------------------------------------------------------------
    # AbstractTrainSet の契約(学習ループから呼ばれる)
    # ------------------------------------------------------------------
    def pre_processing(self, pack, progress):
        """npz のパック → プロンプト(ロールアウト元)のミニデータセットを構築して返す。"""
        raise NotImplementedError("CARL.pre_processing is not implemented.")

    def epoch_fc(self, model, pack, progress):
        """1 ミニバッチ分の前向き計算。`backward`/`get_eval_loss` に渡す材料を返す。

        CARL では通常の (logits, target) ではなく、方策勾配に必要な量
        (対数尤度・アドバンテージ・参照方策との KL など)を返す想定。
        """
        raise NotImplementedError("CARL.epoch_fc is not implemented.")

    def backward(self, accumulation_steps, is_step, progress, lr_param, *args):
        """方策勾配で Actor を更新する(既定の教師あり CE 更新を置き換える)。"""
        raise NotImplementedError("CARL.backward is not implemented.")

    def view_logs(self, *args, **kwargs):
        """進捗表示(報酬平均・KL・生成長などを併記する想定)。"""
        raise NotImplementedError("CARL.view_logs is not implemented.")

    def optional_logging(self, *args, **kwargs):
        """検証時の追加ロギング(報酬の分布、Critic の一致率など)。"""
        raise NotImplementedError("CARL.optional_logging is not implemented.")

    def get_eval_loss(self, *eval):
        """検証用スカラー(報酬の符号反転など、低いほど良い量)を返す。"""
        raise NotImplementedError("CARL.get_eval_loss is not implemented.")

    # ------------------------------------------------------------------
    # CARL 固有(自己批評ループ)
    # ------------------------------------------------------------------
    @abstractmethod
    def rollout(self, prompts: Tensor, progress) -> dict:
        """Actor からサンプリングして生成系列とその対数尤度を得る。"""
        raise NotImplementedError("CARL.rollout is not implemented.")

    @abstractmethod
    def critique(self, rollouts: dict, progress) -> Tensor:
        """**MORTM が MORTM を評価する**中核。

        Critic に生成物を分析方向で読ませ、指定 META との整合・尤度などから
        系列ごとのスカラー評価を返す。
        """
        raise NotImplementedError("CARL.critique is not implemented.")

    @abstractmethod
    def compute_reward(self, rollouts: dict, critique: Tensor) -> Tensor:
        """Critic の評価を報酬へ整形する(KL ペナルティ・正規化・クリッピング等)。"""
        raise NotImplementedError("CARL.compute_reward is not implemented.")
