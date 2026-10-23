from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer


def _is_embedding_or_head_param(name: str) -> bool:
    lowered = name.lower()
    non_muon_tokens = (
        "embedding",
        "embed_tokens",
        "group_embeddings",
        "wout",
        "lm_head",
        "output_head",
        "wte",
        "wpe",
    )
    return any(tok in lowered for tok in non_muon_tokens)


class HybridMuonOptimizer(Optimizer):
    """
    PyTorch 標準搭載の `torch.optim.Muon` (2D 隠れ層重み用) と `torch.optim.AdamW`
    (Embedding / LM 出力ヘッド / RMSNorm / 1D パラメータ用) を統合するラッパー。

    `torch.optim.Muon` は仕様上 2D パラメータのみを受け付け (1D パラメータを渡すと
    `ValueError: Muon only supports 2D parameters` が送出される)、また公式ドキュメントの通り
    Embedding・Bias・Norm には AdamW を併用する必要がある。

    本クラスは `self.param_groups = self.muon.param_groups + self.adamw.param_groups` を共有するため、
    既存の `AbstractTrainSet` の `LambdaLR` スケジューラ、`progress.step_optimizer`、
    および `_save_checkpoint` / `_load_checkpoint` が一切の変更なしでそのまま動作する。
    """

    def __init__(
        self,
        muon_optimizer: torch.optim.Muon,
        adamw_optimizer: torch.optim.AdamW,
    ):
        self.muon = muon_optimizer
        self.adamw = adamw_optimizer
        self.defaults: Dict[str, Any] = {}
        self.state: Dict[Any, Any] = {}
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "muon" in state_dict and "adamw" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
            self.adamw.load_state_dict(state_dict["adamw"])
            self.param_groups = self.muon.param_groups + self.adamw.param_groups
        else:
            super().load_state_dict(state_dict)


def build_muon_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adjust_lr_fn: Optional[str] = "match_rms_adamw",
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    verbose: bool = True,
) -> HybridMuonOptimizer:
    """
    PyTorch 標準の `torch.optim.Muon` と `torch.optim.AdamW` を用いてモデルのオプティマイザを構築する。
    - 隠れ層の 2D 重み行列 (`p.ndim == 2`, Embedding / 出力層を除く) -> `torch.optim.Muon`
    - Embedding / 出力層の 2D 重み (`p.ndim >= 2`) -> `torch.optim.AdamW` (weight_decay あり)
    - RMSNorm / Bias 等の 1D パラメータ (`p.ndim < 2`) -> `torch.optim.AdamW` (weight_decay = 0.0)
    """
    raw_model = model.module if hasattr(model, "module") else model

    muon_params: list[Tensor] = []
    adamw_decay_params: list[Tensor] = []
    adamw_nodecay_params: list[Tensor] = []

    seen_ids: set[int] = set()
    for name, p in raw_model.named_parameters():
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))

        if p.ndim == 2 and not _is_embedding_or_head_param(name):
            muon_params.append(p)
        elif p.ndim >= 2:
            adamw_decay_params.append(p)
        else:
            adamw_nodecay_params.append(p)

    muon_opt = torch.optim.Muon(
        muon_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )

    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append(
            {"params": adamw_decay_params, "lr": lr, "weight_decay": weight_decay}
        )
    if adamw_nodecay_params:
        adamw_groups.append(
            {"params": adamw_nodecay_params, "lr": lr, "weight_decay": 0.0}
        )

    adamw_opt = torch.optim.AdamW(
        adamw_groups,
        lr=lr,
        betas=adamw_betas,
        eps=adamw_eps,
        weight_decay=weight_decay,
    )

    if verbose:
        n_muon = sum(p.numel() for p in muon_params)
        n_adam_wd = sum(p.numel() for p in adamw_decay_params)
        n_adam_nowd = sum(p.numel() for p in adamw_nodecay_params)
        print(
            f"[torch.optim.Muon + AdamW] lr={lr:.2e} wd={weight_decay} momentum={momentum} adjust_lr_fn={adjust_lr_fn} | "
            f"Muon(2D hidden): {n_muon:,} params ({len(muon_params)} tensors) | "
            f"AdamW(Emb/Head): {n_adam_wd:,} params | AdamW(Norm/Bias, wd=0): {n_adam_nowd:,} params"
        )

    return HybridMuonOptimizer(muon_opt, adamw_opt)
