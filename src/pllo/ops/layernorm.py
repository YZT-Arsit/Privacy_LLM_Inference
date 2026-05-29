"""LayerNorm helpers for plain and trusted-stage execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def layer_norm_plain(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply standard LayerNorm over the last dimension."""
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


def trusted_layer_norm(
    x_plain: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply LayerNorm in the simulated trusted side.

    Stage 2 uses trusted LayerNorm as an engineering simplification. This is
    not the final security design; it isolates end-to-end Transformer
    correctness before a fully obfuscated LayerNorm protocol is implemented.
    """
    return layer_norm_plain(x_plain, weight, bias, eps=eps)
