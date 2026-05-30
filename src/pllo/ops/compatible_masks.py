"""Stage 5.2 — Operator-compatible mask families.

Four mask families used inside nonlinear islands, each commuting with a
specific class of nonlinear operator:

* ``generate_dense_invertible`` — general invertible right multiply
  (Linear / Attention / KV cache boundaries).
* ``generate_orthogonal`` — QR-orthogonal mask used at the RMSNorm core.
  ``rms(X N) = rms(X)`` and ``normalize(X N) = normalize(X) N``.
* ``generate_mean_preserving_orthogonal`` — orthogonal mask with ``N 1 = 1``
  used at the LayerNorm core. Preserves both mean and centered norm.
* ``generate_permutation`` — pure index permutation. Element-wise
  activations commute exactly with right-permutation.

All four return material that the trusted side can use to fold transitions
into adjacent Linear weights offline; the GPU side never sees the masks
themselves, only the transformed weights / permutation-applied tensors.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix


# ---------------------------------------------------------------------------
# Dense invertible mask (re-export under the compatible-masks namespace)
# ---------------------------------------------------------------------------


def generate_dense_invertible(
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a dense invertible right-mask ``N`` and return ``(N, N_inv)``."""
    return generate_invertible_matrix(hidden_size, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Orthogonal mask (RMSNorm-compatible)
# ---------------------------------------------------------------------------


def generate_orthogonal(
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample an orthogonal ``hidden_size × hidden_size`` matrix via QR."""
    device_obj = torch.device(device)
    g = torch.randn(hidden_size, hidden_size, dtype=torch.float64, device=device_obj)
    q, r = torch.linalg.qr(g)
    # Sign-fix R's diagonal so Q is Haar-distributed up to numerics.
    sign = torch.sign(torch.diag(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    return q.to(dtype=dtype)


def orthogonal_error(N: torch.Tensor) -> float:
    """Return ``max |N^T N - I|`` for diagnostics."""
    eye = torch.eye(N.shape[0], dtype=N.dtype, device=N.device)
    return float((N.T @ N - eye).abs().max().item())


# ---------------------------------------------------------------------------
# Mean-preserving orthogonal mask (LayerNorm-compatible)
# ---------------------------------------------------------------------------


def center_matrix(
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the centering matrix ``C = I - 1/d * 1 1^T``."""
    device_obj = torch.device(device)
    one = torch.ones(hidden_size, dtype=dtype, device=device_obj)
    eye = torch.eye(hidden_size, dtype=dtype, device=device_obj)
    return eye - (one.unsqueeze(0) * one.unsqueeze(1)) / hidden_size


def generate_mean_preserving_orthogonal(
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample an orthogonal mask satisfying ``N @ 1 = 1`` (and ``N^T C N = C``).

    Construction: build an orthonormal basis ``B = [u, U_perp]`` where
    ``u = 1/sqrt(d) * 1`` is the all-ones direction; sample a random
    ``(d-1) × (d-1)`` orthogonal matrix ``R``; return
    ``N = B diag(1, R) B^T``. By construction ``N u = u`` so ``N @ 1 = 1``,
    and ``N`` is orthogonal because ``B`` and ``diag(1, R)`` are.
    """
    device_obj = torch.device(device)
    d = hidden_size
    if d < 2:
        raise ValueError(f"hidden_size must be >= 2 for mean-preserving orthogonal, got {d}")

    # Build orthonormal basis whose first column is the all-ones direction.
    g = torch.randn(d, d, dtype=torch.float64, device=device_obj)
    g[:, 0] = 1.0  # plant the all-ones direction as the first vector
    B, _ = torch.linalg.qr(g)
    # The first column of B is now ±u with u = 1/sqrt(d) * 1. Flip sign so B[:, 0] = u.
    sign0 = float(torch.sign(B[:, 0].sum()).item())
    if sign0 == 0.0:
        sign0 = 1.0
    B = B * (1.0 if sign0 > 0 else -1.0)

    # Sample a random (d-1) × (d-1) orthogonal R on the centered subspace.
    R = generate_orthogonal(d - 1, torch.float64, device_obj).to(torch.float64)
    eye_block = torch.zeros(d, d, dtype=torch.float64, device=device_obj)
    eye_block[0, 0] = 1.0
    eye_block[1:, 1:] = R

    N = B @ eye_block @ B.T
    return N.to(dtype=dtype)


def mean_preservation_error(N: torch.Tensor) -> float:
    """Return ``max |N @ 1 - 1|``."""
    ones = torch.ones(N.shape[0], dtype=N.dtype, device=N.device)
    return float((N @ ones - ones).abs().max().item())


def centered_orthogonality_error(N: torch.Tensor) -> float:
    """Return ``max |N^T C N - C|`` where ``C = I - 11^T/d``."""
    C = center_matrix(N.shape[0], N.dtype, N.device)
    return float((N.T @ C @ N - C).abs().max().item())


# ---------------------------------------------------------------------------
# Permutation mask (activation-compatible)
# ---------------------------------------------------------------------------


def generate_permutation(
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Sample a random permutation and return both index and dense matrix forms.

    Returns a dict with:

    * ``perm`` — ``[hidden_size]`` long tensor; column ``j`` of the masked
      tensor reads from column ``perm[j]`` of the plain tensor
      (so ``X_perm[..., j] = X[..., perm[j]]``).
    * ``inv_perm`` — inverse permutation indices.
    * ``matrix`` — dense permutation matrix ``P`` such that ``X P = X[:, perm]``
      in row-vector convention. ``P[i, j] = 1`` iff ``perm[j] == i``.
    """
    device_obj = torch.device(device)
    perm = torch.randperm(hidden_size, device=device_obj)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(hidden_size, device=device_obj)

    P = torch.zeros(hidden_size, hidden_size, dtype=dtype, device=device_obj)
    P[perm, torch.arange(hidden_size, device=device_obj)] = 1.0
    return {"perm": perm, "inv_perm": inv_perm, "matrix": P}


def apply_permutation_columns(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply the permutation to the last dimension of ``x`` (i.e. ``x P``)."""
    return x.index_select(dim=-1, index=perm)


# ---------------------------------------------------------------------------
# Fingerprinting — used by reports to dedup mask draws without leaking content
# ---------------------------------------------------------------------------


def matrix_fingerprint(tensor: torch.Tensor) -> str:
    """SHA-256 over the tensor's float32 byte representation."""
    buf = tensor.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


__all__ = [
    "apply_permutation_columns",
    "center_matrix",
    "centered_orthogonality_error",
    "generate_dense_invertible",
    "generate_mean_preserving_orthogonal",
    "generate_orthogonal",
    "generate_permutation",
    "matrix_fingerprint",
    "mean_preservation_error",
    "orthogonal_error",
]
