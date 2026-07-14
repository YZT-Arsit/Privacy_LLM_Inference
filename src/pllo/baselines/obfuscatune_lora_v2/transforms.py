"""Orthogonal coordinate maps for the ObfuscaTune-style LoRA v2 contract.

Weights use mathematical layout ``(d_in, d_out)``.  This module is isolated
from the protected G2 implementation and contains no runtime or optimizer code.
"""

from __future__ import annotations

import torch


def orthogonal_matrix(dim: int, seed: int, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(dim, dim, dtype=torch.float64, generator=generator)
    q, upper = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(upper))
    signs[signs == 0] = 1
    return (q * signs.unsqueeze(0)).to(dtype=dtype, device=device)


def _check_matrix(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
    if value.ndim != 2 or tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")


def transform_input(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    _check_matrix("rotation", rotation, (x.shape[-1], x.shape[-1]))
    return x @ rotation


def restore_input_gradient(dx_star: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    _check_matrix("rotation", rotation, (dx_star.shape[-1], dx_star.shape[-1]))
    return dx_star @ rotation.transpose(0, 1)


def transform_output_gradient(dy: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    _check_matrix("rotation", rotation, (dy.shape[-1], dy.shape[-1]))
    return dy @ rotation


def restore_output(y_star: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    _check_matrix("rotation", rotation, (y_star.shape[-1], y_star.shape[-1]))
    return y_star @ rotation.transpose(0, 1)


def transform_base(weight: torch.Tensor, rotation: torch.Tensor, direction: str) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("weight must have mathematical layout (d_in, d_out)")
    if direction == "input":
        _check_matrix("rotation", rotation, (weight.shape[0], weight.shape[0]))
        return rotation.transpose(0, 1) @ weight
    if direction == "output":
        _check_matrix("rotation", rotation, (weight.shape[1], weight.shape[1]))
        return weight @ rotation
    raise ValueError("direction must be 'input' or 'output'")


def transform_factors(
    a: torch.Tensor, b: torch.Tensor, rotation: torch.Tensor, direction: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A(d_in,r) and B(r,d_out)")
    if direction == "input":
        _check_matrix("rotation", rotation, (a.shape[0], a.shape[0]))
        return rotation.transpose(0, 1) @ a, b.clone()
    if direction == "output":
        _check_matrix("rotation", rotation, (b.shape[1], b.shape[1]))
        return a.clone(), b @ rotation
    raise ValueError("direction must be 'input' or 'output'")


def restore_factors(
    a_star: torch.Tensor, b_star: torch.Tensor, rotation: torch.Tensor, direction: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if direction == "input":
        return rotation @ a_star, b_star.clone()
    if direction == "output":
        return a_star.clone(), b_star @ rotation.transpose(0, 1)
    raise ValueError("direction must be 'input' or 'output'")
