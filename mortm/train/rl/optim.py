"""Mixed-precision FP32 Master-Weight AdamW wrapper for low-LR RL updates.

In MORTM / MORTM5, `QKVLinear` (`qkv_weight`, `W_o`) stores weights directly in
`torch.bfloat16` (7-bit mantissa, half-ULP ~ `1.22e-4` at `|w| ~ 0.031`).
Standard `torch.optim.AdamW` on `bfloat16` parameters rounds each update `lr ~ 2e-6`
directly back to the identical `bfloat16` bit pattern (`bf16(w + dw) == w`), freezing
all 50M+ attention weights completely.

`MasterWeightAdamW` maintains detached `torch.float32` master copies for any
`bfloat16` / `float16` trainable parameters so micro-updates accumulate losslessly
in FP32 and cross `bfloat16` quantization boundaries accurately while keeping the
live module weights in their native `bfloat16` dtype, and skips non-finite (`NaN`/`Inf`)
gradient steps before they can corrupt AdamW moments or model weights.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import Tensor, nn


class MasterWeightAdamW:
    """AdamW optimizer maintaining FP32 master weights for `bf16`/`fp16` parameters."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self._native_fp32: List[nn.Parameter] = []
        self._low_prec_pairs: List[Tuple[nn.Parameter, nn.Parameter]] = []
        optim_params: List[nn.Parameter] = []

        for param in params:
            if not param.requires_grad:
                continue
            if param.dtype in (torch.bfloat16, torch.float16):
                master = nn.Parameter(param.detach().float().clone())
                self._low_prec_pairs.append((param, master))
                optim_params.append(master)
            else:
                self._native_fp32.append(param)
                optim_params.append(param)

        if not optim_params:
            raise ValueError("MasterWeightAdamW received 0 trainable parameters.")

        self._all_optim_params = optim_params
        self._opt = torch.optim.AdamW(
            optim_params,
            lr=float(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @property
    def num_low_precision_tensors(self) -> int:
        return len(self._low_prec_pairs)

    def zero_grad(self) -> None:
        self._opt.zero_grad(set_to_none=True)
        for low_p, _ in self._low_prec_pairs:
            low_p.grad = None
        for p in self._native_fp32:
            p.grad = None

    def _has_all_finite_grads(self) -> bool:
        for p in self._all_optim_params:
            if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
                return False
        return True

    def step_and_zero_grad(self, grad_clip: float = 0.0) -> bool:
        """Transfer low-precision gradients to FP32 masters, verify finiteness, clip, step, and write back.

        Returns:
            ``True`` if the optimizer step was applied; ``False`` if skipped due to `NaN`/`Inf` gradients.
        """
        for low_p, master_p in self._low_prec_pairs:
            if low_p.grad is not None:
                master_p.grad = low_p.grad.detach().float()
                low_p.grad = None
            else:
                master_p.grad = None

        if not self._has_all_finite_grads():
            self.zero_grad()
            return False

        if grad_clip > 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(self._all_optim_params, float(grad_clip))
            if not bool(torch.isfinite(total_norm).all().item()):
                self.zero_grad()
                return False

        self._opt.step()
        self.zero_grad()

        if self._low_prec_pairs:
            with torch.no_grad():
                for low_p, master_p in self._low_prec_pairs:
                    low_p.copy_(master_p.to(dtype=low_p.dtype))
        return True

    @torch.no_grad()
    def sync_from_model(self) -> None:
        """Re-synchronize FP32 master weights after external `load_state_dict` / rollback."""
        for low_p, master_p in self._low_prec_pairs:
            master_p.copy_(low_p.detach().float())
