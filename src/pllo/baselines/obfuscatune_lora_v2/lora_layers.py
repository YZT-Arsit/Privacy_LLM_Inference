"""Minimal transformed-coordinate LoRA forward operators."""

from __future__ import annotations

import torch


def lora_scale(alpha: float, rank: int) -> float:
    if rank <= 0:
        raise ValueError("rank must be positive")
    return float(alpha) / int(rank)


def effective_weight(
    weight_star: torch.Tensor,
    a_star: torch.Tensor,
    b_star: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if weight_star.ndim != 2 or a_star.ndim != 2 or b_star.ndim != 2:
        raise ValueError("weight and factors must be matrices")
    if weight_star.shape != (a_star.shape[0], b_star.shape[1]) or a_star.shape[1] != b_star.shape[0]:
        raise ValueError("incompatible transformed weight/factor shapes")
    return weight_star + float(scale) * (a_star @ b_star)


def transformed_lora_forward(
    x_stored: torch.Tensor,
    weight_star: torch.Tensor,
    a_star: torch.Tensor,
    b_star: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return x_stored @ effective_weight(weight_star, a_star, b_star, scale)


def plaintext_lora_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return x @ (weight + float(scale) * (a @ b))
