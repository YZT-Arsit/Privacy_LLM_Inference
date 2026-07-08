"""Variant D — exact two-sided non-orthogonal key-matrix transform.

Goal (experiment-only): borrow ONLY the "invertible non-orthogonal two-sided key
matrices" idea (as used by e.g. AloePri's ``P_hat Q_hat = I`` pair) and test
whether it lowers anchor-based *weight-alignment* attacks (Gram / VMA /
ArrowMatch) while staying **exact-lossless**. We add NO Gaussian noise and NO
RMSNorm expectation correction — this is deliberately NOT an AloePri replication,
only its exact-invertible key-matrix core.

Linear-layer identity (exact for a linear chain):

    Y = X W + b
    X_tilde = X P_in                 (P_in non-orthogonal, invertible)
    W_tilde = Q_in W P_out           (Q_in = P_in^{-1},  P_in Q_in = I)
    b_tilde = b P_out
    =>  X_tilde W_tilde + b_tilde = X P_in Q_in W P_out + b P_out
                                  = (X W + b) P_out = Y P_out = Y_tilde

So the stable state of the next layer is ``Y P_out``; chaining requires the next
layer's ``P_in`` to equal this layer's ``P_out``. The transform is exact for the
*linear* chain at any lambda (up to floating-point conditioning).

SCOPE / honesty (see docs/rmsnorm_exact_norm_impossibility.md):
  * A non-orthogonal P does NOT commute with elementwise nonlinearities
    (GELU/SiLU) nor with RMSNorm's per-token normalisation. So this transform is
    exact ONLY across linear layers; an end-to-end Transformer block would need a
    TEE crossing at each nonlinearity (ObfuscaTune-class), OR an approximation
    (which we refuse — no noise, no expectation correction). We therefore test
    and claim exactness for the linear chain only.
  * ``exact_lossless_claim`` is asserted by tests, not declared here.
  * ``formal_security_claim = False``; the anchor-attack numbers are empirical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.baselines.obfuscatune.random_matrices import orthogonal_matrix


@dataclass
class KeyMatDiagnostics:
    """Conditioning diagnostics for a key matrix P (all computed in float64)."""

    lam: float
    cond: float
    min_singular: float
    max_singular: float
    inverse_max_abs_error: float          # ||P Q - I||_max


def sample_nonorthogonal_keymat(
    d: int,
    lam: float,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, KeyMatDiagnostics]:
    """Noise-free non-orthogonal key matrix ``P = U + lam V`` and ``Q = P^{-1}``.

    ``U ~ orthogonal(d)``; ``V ~ Gaussian(0, 1/d)`` (entrywise std ``1/sqrt(d)``).
    ``lam = 0`` returns an exactly-orthogonal key (the ObfuscaTune case). Larger
    ``lam`` moves P away from orthogonality (raises the condition number and makes
    the transform non-norm-preserving). Returns ``(P, Q, diagnostics)`` with P, Q
    in ``dtype`` and diagnostics measured in float64.
    """
    u, _ = orthogonal_matrix(d, seed=seed, dtype=torch.float64, device=device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 1)
    v = torch.randn(d, d, dtype=torch.float64, generator=g).to(device) / (d ** 0.5)
    p64 = u + float(lam) * v
    q64 = torch.linalg.inv(p64)
    sv = torch.linalg.svdvals(p64)
    diag = KeyMatDiagnostics(
        lam=float(lam),
        cond=float((sv.max() / sv.min().clamp_min(1e-300)).item()),
        min_singular=float(sv.min().item()),
        max_singular=float(sv.max().item()),
        inverse_max_abs_error=float((p64 @ q64 - torch.eye(d, dtype=torch.float64, device=device)).abs().max().item()),
    )
    return p64.to(dtype), q64.to(dtype), diag


def fold_weight(w: torch.Tensor, q_in: torch.Tensor, p_out: torch.Tensor) -> torch.Tensor:
    """``W_tilde = Q_in W P_out`` (the offloaded folded weight, shape in==out d)."""
    return q_in @ w @ p_out


def fold_bias(b: torch.Tensor, p_out: torch.Tensor) -> torch.Tensor:
    """``b_tilde = b P_out``."""
    return b @ p_out


def two_sided_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    p_in: torch.Tensor,
    q_in: torch.Tensor,
    p_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run one obfuscated linear layer and return the pieces for auditing.

    Returns ``x_tilde = X P_in`` (offloaded stable state), ``w_tilde``/``b_tilde``
    (offloaded folded weights), ``y_tilde`` (= X_tilde W_tilde + b_tilde) and the
    plaintext-then-masked reference ``y_ref = (X W + b) P_out`` for the exactness
    check.
    """
    x_tilde = x @ p_in
    w_tilde = fold_weight(w, q_in, p_out)
    y_plain = x @ w
    if b is not None:
        y_plain = y_plain + b
    b_tilde = fold_bias(b, p_out) if b is not None else None
    y_tilde = x_tilde @ w_tilde
    if b_tilde is not None:
        y_tilde = y_tilde + b_tilde
    return {
        "x_tilde": x_tilde,
        "w_tilde": w_tilde,
        "b_tilde": b_tilde,
        "y_tilde": y_tilde,
        "y_ref": y_plain @ p_out,          # exact target of the obfuscated layer
        "y_plain": y_plain,
    }


def two_sided_keymat_report_fields(
    diag: KeyMatDiagnostics | None = None,
    *,
    max_abs_error: float | None = None,
    relative_l2_error: float | None = None,
    exact_lossless_tests_passed: bool = False,
) -> dict[str, Any]:
    """JSON-safe audit fields for Variant D (honest-scope flags)."""
    fields: dict[str, Any] = {
        "variant": "two_sided_nonorthogonal_exact",
        "aliases": ["exact_keymat_two_sided", "variant_D"],
        "uses_two_sided_keymat": True,
        "uses_nonorthogonal_transform": True,
        "uses_noise": False,                       # NO Gaussian noise (not AloePri)
        "uses_rmsnorm_expectation_correction": False,
        "stable_state_form": "X_tilde = X P_in (P_in non-orthogonal)",
        "folded_weight_form": "W_tilde = Q_in W P_out",
        "exact_scope": "linear_chain_only",        # not through RMSNorm/nonlinearity
        "passes_elementwise_nonlinearity_exactly": False,
        "passes_rmsnorm_exactly": False,
        # exactness is a *tested* property, never self-declared:
        "exact_lossless_claim": bool(exact_lossless_tests_passed),
        "formal_security_claim": False,
        "anchor_attack_audit": True,
        "attack_surface": "anchor_weight_alignment",
        "requires_plaintext_weight_anchor": True,
        "is_aloepri_replication": False,
        "production_qwen7b_integration": False,
    }
    if diag is not None:
        fields.update({
            "lambda": diag.lam,
            "cond_P": diag.cond,
            "min_singular": diag.min_singular,
            "max_singular": diag.max_singular,
            "keymat_inverse_max_abs_error": diag.inverse_max_abs_error,
        })
    if max_abs_error is not None:
        fields["max_abs_error"] = float(max_abs_error)
    if relative_l2_error is not None:
        fields["relative_l2_error"] = float(relative_l2_error)
    return fields


__all__ = [
    "KeyMatDiagnostics",
    "sample_nonorthogonal_keymat",
    "fold_weight",
    "fold_bias",
    "two_sided_linear",
    "two_sided_keymat_report_fields",
]
