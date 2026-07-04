"""Random-matrix generation for ObfuscaTune obfuscation (arXiv:2407.02960).

The paper obfuscates the high-parameter attention/MLP linear layers by
multiplying operands with random matrices. Numerical accuracy is governed by
the matrix condition number kappa (Sec. 4, Table 2, App. B):

  * ``orthogonal_matrix``           -> kappa = 1 (Protected(ours), error-free
    inverse via transpose).
  * ``gaussian_invertible_matrix``  -> naive Gaussian matrix with an
    uncontrolled (typically large) kappa (Protected(random)).
  * ``matrix_with_condition_number``-> a matrix with a *prescribed* kappa built
    from R = QA @ S @ QB^T (App. B), used for the condition-number sweep.

All generators are deterministic in ``seed`` and return ``(R, R_inv)`` where
``R_inv`` is formed analytically (transpose for orthogonal, SVD factors for the
prescribed-kappa case, ``torch.linalg.inv`` for the naive case). Raw matrices
are meant for local numerical study only; they are never serialised by the
baseline protocol.
"""

from __future__ import annotations

import torch


def _generator(seed: int) -> torch.Generator:
    # Always seed a CPU generator: torch RNG streams differ per device, and we
    # want bit-identical matrices regardless of the target ``device``.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _place(t: torch.Tensor, dtype: torch.dtype, device: str) -> torch.Tensor:
    return t.to(dtype=dtype, device=device)


def orthogonal_matrix(
    dim: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal ``Q`` (kappa = 1) and its exact inverse ``Q^T``.

    Built via QR of a standard-normal matrix; ``Q`` is orthogonal so
    ``Q^{-1} = Q^T`` is an error-free operation (paper Sec. 4).
    """
    g = _generator(seed)
    a = torch.randn(dim, dim, dtype=torch.float64, generator=g)
    q, r = torch.linalg.qr(a)
    # Fix the sign ambiguity of QR so the result is deterministic across
    # LAPACK versions: force positive diagonal of R.
    sign = torch.sign(torch.diagonal(r))
    sign[sign == 0] = 1.0
    q = q * sign.unsqueeze(0)
    return _place(q, dtype, device), _place(q.transpose(0, 1).contiguous(), dtype, device)


def gaussian_invertible_matrix(
    dim: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    jitter: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive Gaussian matrix (Protected(random)) with an uncontrolled kappa.

    A small ridge ``jitter * I`` is added to keep the draw comfortably
    invertible; the inverse is computed numerically (``torch.linalg.inv``),
    which is exactly the error-prone path the paper contrasts against the
    orthogonal choice.
    """
    g = _generator(seed)
    a = torch.randn(dim, dim, dtype=torch.float64, generator=g)
    a = a + jitter * torch.eye(dim, dtype=torch.float64)
    a_inv = torch.linalg.inv(a)
    return _place(a, dtype, device), _place(a_inv, dtype, device)


def matrix_with_condition_number(
    dim: int,
    cond: float,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matrix with a *prescribed* condition number ``cond`` (paper App. B).

    ``R = QA @ S @ QB^T`` with orthogonal ``QA``, ``QB`` and singular values
    spanning ``[sigma_max/cond, sigma_max]`` (sigma_max fixed to 1). The largest
    and smallest singular values are pinned to the endpoints so ``cond(R)``
    equals the target; the inverse ``R_inv = QB @ S^{-1} @ QA^T`` is formed from
    the same factors, avoiding a numerical inversion.

    ``cond <= 1`` degenerates to :func:`orthogonal_matrix`.
    """
    if cond <= 1.0:
        return orthogonal_matrix(dim, seed, dtype, device)
    g = _generator(seed)

    def _orth(gen: torch.Generator) -> torch.Tensor:
        a = torch.randn(dim, dim, dtype=torch.float64, generator=gen)
        q, r = torch.linalg.qr(a)
        sign = torch.sign(torch.diagonal(r))
        sign[sign == 0] = 1.0
        return q * sign.unsqueeze(0)

    qa = _orth(g)
    qb = _orth(g)
    smax = 1.0
    smin = smax / float(cond)
    if dim > 2:
        mid = torch.rand(dim - 2, dtype=torch.float64, generator=g) * (smax - smin) + smin
        sv = torch.cat([
            torch.tensor([smax], dtype=torch.float64),
            mid,
            torch.tensor([smin], dtype=torch.float64),
        ])
    elif dim == 2:
        sv = torch.tensor([smax, smin], dtype=torch.float64)
    else:  # dim == 1: condition number is undefined; return scalar 1
        sv = torch.tensor([smax], dtype=torch.float64)
    s = torch.diag(sv)
    s_inv = torch.diag(1.0 / sv)
    r_mat = qa @ s @ qb.transpose(0, 1)
    r_inv = qb @ s_inv @ qa.transpose(0, 1)
    return _place(r_mat, dtype, device), _place(r_inv, dtype, device)


def condition_number(matrix: torch.Tensor) -> float:
    """L2 condition number sigma_max / sigma_min (float, computed in float64)."""
    m = matrix.detach().to(torch.float64)
    return float(torch.linalg.cond(m).item())


__all__ = [
    "orthogonal_matrix",
    "gaussian_invertible_matrix",
    "matrix_with_condition_number",
    "condition_number",
]
