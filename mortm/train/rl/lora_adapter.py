"""Lightweight PyTorch-native LoRA adapter without weight-merging side effects.

Unlike `loralib.Linear`, which mutates `self.weight` in-place on `eval()` / `train()`
and caused checkpoint double-merge bugs in legacy CARL v1, `LoRALinearAdapter`
computes the low-rank residual dynamically while keeping the frozen base `nn.Linear`
strictly immutable.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinearAdapter(nn.Module):
    """Drop-in `nn.Linear` wrapper adding a trainable rank-`r` residual branch."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_linear).__name__}.")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad_(False)

        dev = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((self.rank, self.in_features), device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.rank), device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout: nn.Module = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()

    @property
    def weight(self) -> Tensor:
        return self.base_linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_linear.bias

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        x_drop = self.lora_dropout(x).to(dtype=self.lora_A.dtype)
        delta = F.linear(F.linear(x_drop, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta.to(dtype=base_out.dtype)


def inject_attention_lora(
    decoder_layers: Iterable[nn.Module],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Tuple[int, int]:
    """Wrap `qkv_weight` and `W_o` inside every decoder layer's `self_attention.qkv_block`.

    Returns:
        ``(num_wrapped_modules, total_lora_params)``
    """
    wrapped = 0
    lora_params = 0
    for layer in decoder_layers:
        attn = getattr(layer, "self_attention", None)
        qkv_block = getattr(attn, "qkv_block", None)
        if qkv_block is None:
            continue
        for attr in ("qkv_weight", "W_o"):
            mod = getattr(qkv_block, attr, None)
            if isinstance(mod, LoRALinearAdapter):
                wrapped += 1
                lora_params += mod.lora_A.numel() + mod.lora_B.numel()
            elif isinstance(mod, nn.Linear):
                adapter = LoRALinearAdapter(mod, rank=rank, alpha=alpha, dropout=dropout)
                setattr(qkv_block, attr, adapter)
                wrapped += 1
                lora_params += adapter.lora_A.numel() + adapter.lora_B.numel()
    if wrapped == 0:
        raise RuntimeError("inject_attention_lora found no compatible self_attention.qkv_block Linear layers.")
    return wrapped, lora_params
