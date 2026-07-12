"""再現性ユーティリティ (研究設計書 v1.3 §9.6)。

全実験の乱数種・決定論フラグを単一箇所で固定する。設計書 §4.3:
  「データローダ・torch・numpy の全シードを固定し、
   torch.use_deterministic_algorithms(True) が不可能な演算はログに残す」

使い方:
    from mortm.utils.repro import set_seed, seed_worker
    info = set_seed(1337)              # 学習/評価スクリプト冒頭で1回
    DataLoader(..., worker_init_fn=seed_worker, generator=info.generator)

set_seed は決定論設定の結果(どのフラグが立ったか/非決定論的に留まる要因)を
ReproInfo として返し、呼び出し側が wandb 等へ記録できるようにする。
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ReproInfo:
    """set_seed が確定させた再現性状態。wandb config へそのまま流し込める。"""
    seed: int
    deterministic_algorithms: bool          # torch.use_deterministic_algorithms が有効か
    cudnn_deterministic: bool
    cublas_workspace_configured: bool
    torch_version: str
    cuda_available: bool
    notes: list[str] = field(default_factory=list)  # 非決定論に留まる要因のログ
    generator: object = None                # DataLoader 用 torch.Generator (seed 済み)


def seed_worker(worker_id: int) -> None:
    """DataLoader の worker_init_fn。各 worker の numpy/random 種を torch 初期種から導出する。"""
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int = 1337, *, deterministic: bool = True) -> ReproInfo:
    """python/numpy/torch(+CUDA) の全乱数種を固定し、決定論フラグを設定する。

    deterministic=True のとき torch.use_deterministic_algorithms(True) を試み、
    CUBLAS のワークスペースも設定する。決定論化できない場合は例外を握りつぶさず
    notes に記録して警告付きで続行する(設計書 §4.3: 不可能な演算はログに残す)。
    """
    import torch

    notes: list[str] = []

    # --- CUBLAS 決定論のための環境変数は torch import/CUDA init 前が理想 ---
    # ここで設定しても既に CUDA 初期化済みなら効かない可能性がある旨を記録。
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prev_cublas not in (":4096:8", ":16:8"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            notes.append(
                "CUBLAS_WORKSPACE_CONFIG を CUDA 初期化後に設定した。"
                "完全な決定論には set_seed をプロセス冒頭(CUDA 使用前)で呼ぶこと。"
            )
    cublas_configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PYTHONHASHSEED は既に起動済みプロセスには遡及しない。次回起動用に設定+記録。
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        notes.append(
            "PYTHONHASHSEED はプロセス起動時にのみ有効。"
            "完全再現には起動前に PYTHONHASHSEED を設定すること。"
        )

    cudnn_det = False
    det_algos = False
    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cudnn_det = True
        except Exception as e:  # noqa: BLE001 - 環境差を握らず記録
            notes.append(f"cudnn 決定論設定に失敗: {e!r}")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            det_algos = True
        except Exception as e:  # noqa: BLE001
            notes.append(f"use_deterministic_algorithms 設定に失敗(非決定論のまま): {e!r}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return ReproInfo(
        seed=seed,
        deterministic_algorithms=det_algos,
        cudnn_deterministic=cudnn_det,
        cublas_workspace_configured=cublas_configured,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        notes=notes,
        generator=generator,
    )
