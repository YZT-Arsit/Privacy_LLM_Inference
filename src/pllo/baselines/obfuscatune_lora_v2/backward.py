"""Manual gradients for transformed-coordinate LoRA linear operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora_layers import effective_weight


@dataclass(frozen=True)
class LoRAGradients:
    dx: torch.Tensor
    da: torch.Tensor
    db: torch.Tensor


def transformed_lora_backward(
    x_stored: torch.Tensor,
    grad_output_stored: torch.Tensor,
    weight_star: torch.Tensor,
    a_star: torch.Tensor,
    b_star: torch.Tensor,
    scale: float,
) -> LoRAGradients:
    """Return gradients in the exact coordinates of the stored operands."""
    s = float(scale)
    merged = effective_weight(weight_star, a_star, b_star, s)
    dx = grad_output_stored @ merged.transpose(0, 1)
    da = s * (x_stored.transpose(-2, -1) @ grad_output_stored) @ b_star.transpose(0, 1)
    db = s * a_star.transpose(0, 1) @ (x_stored.transpose(-2, -1) @ grad_output_stored)
    return LoRAGradients(dx=dx, da=da, db=db)
