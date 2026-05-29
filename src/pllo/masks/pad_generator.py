"""Fresh one-time pad generation."""

from __future__ import annotations

import torch


def generate_pad(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = torch.device("cpu"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Generate a fresh random pad tensor with the same shape as an input."""
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"all shape dimensions must be positive, got {shape}")
    if scale < 0:
        raise ValueError(f"scale must be non-negative, got {scale}")

    return torch.randn(shape, dtype=dtype, device=torch.device(device)) * scale
