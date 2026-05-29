"""Invertible mask generation."""

from __future__ import annotations

import torch


def generate_invertible_matrix(
    dim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a numerically stable invertible matrix and its inverse.

    The implementation uses QR decomposition to create an orthogonal matrix.
    Orthogonal masks have condition number 1 in exact arithmetic, which keeps
    the first-stage correctness tests focused on the obfuscation equations
    rather than numerical pathology.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    device = torch.device(device)
    random_matrix = torch.randn(dim, dim, dtype=dtype, device=device)
    q, r = torch.linalg.qr(random_matrix)

    # Make the sign convention deterministic for a fixed random draw.
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)

    return q, q.transpose(-2, -1)
