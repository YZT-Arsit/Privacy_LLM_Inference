"""Stage 5.2 — Operator-compatible nonlinear islands.

Reference cores, affine-folding helpers, and full-island forwards for:

* RMSNorm core under an *orthogonal* right mask.
* LayerNorm core under a *mean-preserving orthogonal* right mask.
* Element-wise activations (GELU / ReLU / SiLU) under a *permutation* mask.
* SwiGLU under a *paired* permutation shared by the up- and gate-branch.
* Full GELU / ReLU / SiLU MLP island: ``activation(X W1 + b1) W2 + b2``.
* Full SwiGLU MLP island: ``((X W_up + b_up) * SiLU(X W_gate + b_gate)) W_down + b_down``.

The island wrappers fold every mask + permutation transition into adjacent
Linear weights offline, so no extra online matmul is introduced. Pad
compensation is applied only at the Linear boundary — pads are never
pushed through an activation.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    bundle_metadata,
)


# ---------------------------------------------------------------------------
# Norm cores (no affine)
# ---------------------------------------------------------------------------


def layernorm_core(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``LayerNorm`` without affine parameters: ``(x - mean) / sqrt(var + eps)``."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    # Use biased variance to match ``F.layer_norm`` semantics.
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(var + eps)


def rmsnorm_core(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm without affine: ``x / sqrt(mean(x^2) + eps)``."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Affine folding helpers
# ---------------------------------------------------------------------------


def fold_layernorm_affine_into_linear(
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold LayerNorm gamma / beta into the following Linear.

    Row-vector convention:

        Input after norm = LNCore(X) * gamma + beta
        Following linear = Z W + b
        Folded:
            W_folded = diag(gamma) @ W
            b_folded = beta @ W + b
    """
    W = linear_weight
    if norm_weight is not None:
        W_folded = norm_weight.unsqueeze(-1) * W   # diag(gamma) @ W via broadcast
    else:
        W_folded = W.clone()

    b_folded = linear_bias.clone() if linear_bias is not None else None
    if norm_bias is not None:
        beta_term = norm_bias @ W
        b_folded = beta_term if b_folded is None else (b_folded + beta_term)
    return W_folded, b_folded


def fold_rmsnorm_affine_into_linear(
    norm_weight: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold RMSNorm gamma into the following Linear; RMSNorm has no bias."""
    if norm_weight is not None:
        W_folded = norm_weight.unsqueeze(-1) * linear_weight
    else:
        W_folded = linear_weight.clone()
    b_folded = linear_bias.clone() if linear_bias is not None else None
    return W_folded, b_folded


# ---------------------------------------------------------------------------
# Norm + Linear islands
# ---------------------------------------------------------------------------


def run_rmsnorm_orthogonal_island(
    x: torch.Tensor,
    n_in_orthogonal: torch.Tensor,
    norm_weight: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    n_out: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """RMSNorm core → affine fold into a following Linear.

    Plain:  ``Y = (rmsnorm_core(X) * gamma) @ W + b``
    Masked: ``X_tilde = X @ N``  (N orthogonal)
            ``core_tilde = rmsnorm_core(X_tilde) = rmsnorm_core(X) @ N``
            ``W_folded = diag(gamma) @ W``
            ``W_tilde = N^T @ W_folded @ N_out``  (offline)
            ``b_tilde = b @ N_out``               (offline)
            ``Y_tilde = core_tilde @ W_tilde + b_tilde``
    Invariant: ``Y_tilde ≈ Y @ N_out``.
    """
    N = n_in_orthogonal
    # Plain reference.
    core = rmsnorm_core(x, eps=eps)
    if norm_weight is not None:
        scaled = core * norm_weight
    else:
        scaled = core
    Y = scaled @ linear_weight
    if linear_bias is not None:
        Y = Y + linear_bias

    # Masked.
    W_folded, b_folded = fold_rmsnorm_affine_into_linear(
        norm_weight, linear_weight, linear_bias
    )
    x_tilde = x @ N
    core_tilde = rmsnorm_core(x_tilde, eps=eps)
    W_tilde = N.T @ W_folded @ n_out
    b_tilde = b_folded @ n_out if b_folded is not None else None
    Y_tilde = core_tilde @ W_tilde
    if b_tilde is not None:
        Y_tilde = Y_tilde + b_tilde

    return {
        "y_plain": Y,
        "y_tilde": Y_tilde,
        "expected_y_tilde": Y @ n_out,
    }


def run_layernorm_mean_preserving_island(
    x: torch.Tensor,
    n_in_mp_orthogonal: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    n_out: torch.Tensor,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """LayerNorm core → affine fold into a following Linear under a
    mean-preserving orthogonal mask."""
    N = n_in_mp_orthogonal
    core = layernorm_core(x, eps=eps)
    scaled = core if norm_weight is None else core * norm_weight
    if norm_bias is not None:
        scaled = scaled + norm_bias
    Y = scaled @ linear_weight
    if linear_bias is not None:
        Y = Y + linear_bias

    W_folded, b_folded = fold_layernorm_affine_into_linear(
        norm_weight, norm_bias, linear_weight, linear_bias
    )
    x_tilde = x @ N
    core_tilde = layernorm_core(x_tilde, eps=eps)
    W_tilde = N.T @ W_folded @ n_out
    b_tilde = b_folded @ n_out if b_folded is not None else None
    Y_tilde = core_tilde @ W_tilde
    if b_tilde is not None:
        Y_tilde = Y_tilde + b_tilde

    return {
        "y_plain": Y,
        "y_tilde": Y_tilde,
        "expected_y_tilde": Y @ n_out,
    }


# ---------------------------------------------------------------------------
# Activation references
# ---------------------------------------------------------------------------


def gelu_reference(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def relu_reference(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def silu_reference(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def swiglu_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SwiGLU: ``a * silu(b)``."""
    return a * F.silu(b)


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": gelu_reference,
    "relu": relu_reference,
    "silu": silu_reference,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; supported: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]


# ---------------------------------------------------------------------------
# Activation + SwiGLU permutation islands
# ---------------------------------------------------------------------------


def run_activation_permutation_island(
    z: torch.Tensor,
    permutation: torch.Tensor,
    activation_type: str,
) -> dict[str, torch.Tensor]:
    """Verify ``activation(Z P) = activation(Z) P`` for an element-wise activation."""
    f = get_activation(activation_type)
    z_perm = z.index_select(dim=-1, index=permutation)
    lhs = f(z_perm)
    rhs = f(z).index_select(dim=-1, index=permutation)
    return {"lhs": lhs, "rhs": rhs}


def run_swiglu_paired_permutation_island(
    a: torch.Tensor,
    b: torch.Tensor,
    permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Verify ``swiglu(A P, B P) = swiglu(A, B) P`` with shared P."""
    a_perm = a.index_select(dim=-1, index=permutation)
    b_perm = b.index_select(dim=-1, index=permutation)
    lhs = swiglu_reference(a_perm, b_perm)
    rhs = swiglu_reference(a, b).index_select(dim=-1, index=permutation)
    return {"lhs": lhs, "rhs": rhs}


# ---------------------------------------------------------------------------
# Full MLP islands
# ---------------------------------------------------------------------------


def _permute_w_cols(W: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """``W[:, perm]`` — equivalent to ``W @ P`` in row-vector convention."""
    return W.index_select(dim=-1, index=perm)


def _permute_w_rows(W: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """``W[perm, :]`` — equivalent to ``P^T @ W`` (inverse-permute rows)."""
    return W.index_select(dim=0, index=perm)


def run_gelu_mlp_island(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    permutation: torch.Tensor,
    n_out: torch.Tensor,
    activation_type: str,
    pad_in: torch.Tensor | None = None,
    mitigation_bundle: str | None = None,
) -> dict[str, torch.Tensor]:
    """Two-layer MLP with operator-compatible masks at the activation.

    Plain:  ``Y = activation(X W1 + b1) W2 + b2``.

    Masked (``activation ∈ {gelu, relu, silu}``):
        ``X_tilde   = X N_in``                          (no pad)
                    = (X - T_in) N_in                   (with input pad)
        ``W1_tilde  = N_in^{-1} W1[:, perm]``           (offline)
        ``b1_tilde  = b1[perm]``                        (offline)
        ``C1        = T_in W1[:, perm]``                (pad compensation)
        ``Z_tilde   = X_tilde W1_tilde + b1_tilde + C1 = Z[:, perm]``
        ``A_tilde   = activation(Z_tilde)               = A[:, perm]``
        ``W2_tilde  = (W2 N_out)[perm, :]``             (offline)
        ``b2_tilde  = b2 N_out``                        (offline)
        ``Y_tilde   = A_tilde W2_tilde + b2_tilde       = Y N_out``

    No pad is pushed through the activation. ``online_extra_matmul_count``
    is ``0`` — the masked path has the same matmul count as the plain path
    plus the standard mask + pad compensation that already exists for
    every Stage-1 obfuscated linear.
    """
    f = get_activation(activation_type)
    perm = permutation

    # Plain reference.
    Z = x @ w1
    if b1 is not None:
        Z = Z + b1
    A = f(Z)
    Y = A @ w2
    if b2 is not None:
        Y = Y + b2

    # Masked path.
    W1_perm = _permute_w_cols(w1, perm)        # W1 @ P
    if pad_in is None:
        x_tilde = x @ n_in
        c1 = None
    else:
        x_tilde = (x - pad_in) @ n_in
        c1 = pad_in @ W1_perm
    W1_tilde = n_in_inv @ W1_perm              # N_in^{-1} W1 P
    b1_tilde = b1.index_select(dim=-1, index=perm) if b1 is not None else None
    Z_tilde = x_tilde @ W1_tilde
    if b1_tilde is not None:
        Z_tilde = Z_tilde + b1_tilde
    if c1 is not None:
        Z_tilde = Z_tilde + c1

    A_tilde = f(Z_tilde)

    W2_tilde = _permute_w_rows(w2, perm) @ n_out   # P^{-1} W2 N_out
    Y_tilde = A_tilde @ W2_tilde
    if b2 is not None:
        Y_tilde = Y_tilde + b2 @ n_out

    bundle = mitigation_bundle or DEFAULT_MITIGATION_BUNDLE
    bundle_meta = bundle_metadata(
        bundle, use_pad=pad_in is not None, online_extra_matmul_count=0
    )
    return {
        "y_plain": Y,
        "y_tilde": Y_tilde,
        "expected_y_tilde": Y @ n_out,
        "z_tilde": Z_tilde,
        "z_plain_permuted": Z.index_select(dim=-1, index=perm),
        "a_tilde": A_tilde,
        "a_plain_permuted": A.index_select(dim=-1, index=perm),
        "used_input_pad": pad_in is not None,
        "mitigation_bundle_metadata": bundle_meta,
    }


def run_swiglu_mlp_island(
    x: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    permutation: torch.Tensor,
    n_out: torch.Tensor,
    pad_in: torch.Tensor | None = None,
    mitigation_bundle: str | None = None,
) -> dict[str, torch.Tensor]:
    """SwiGLU MLP with paired permutation on the up- and gate-branches.

    Plain:  ``A = X W_up + b_up``
            ``B = X W_gate + b_gate``
            ``G = A * silu(B)``
            ``Y = G W_down + b_down``

    Masked: shared permutation ``P`` for both branches, so
            ``A_tilde = A P``, ``B_tilde = B P``,
            ``G_tilde = A_tilde * silu(B_tilde) = G P``.
            ``W_down_tilde = (W_down N_out)[perm, :]``.
    """
    perm = permutation

    # Plain reference.
    A = x @ w_up
    if b_up is not None:
        A = A + b_up
    B = x @ w_gate
    if b_gate is not None:
        B = B + b_gate
    G = swiglu_reference(A, B)
    Y = G @ w_down
    if b_down is not None:
        Y = Y + b_down

    # Masked path.
    W_up_perm = _permute_w_cols(w_up, perm)
    W_gate_perm = _permute_w_cols(w_gate, perm)
    if pad_in is None:
        x_tilde = x @ n_in
        c_up = None
        c_gate = None
    else:
        x_tilde = (x - pad_in) @ n_in
        c_up = pad_in @ W_up_perm
        c_gate = pad_in @ W_gate_perm

    W_up_tilde = n_in_inv @ W_up_perm
    W_gate_tilde = n_in_inv @ W_gate_perm
    b_up_tilde = b_up.index_select(dim=-1, index=perm) if b_up is not None else None
    b_gate_tilde = (
        b_gate.index_select(dim=-1, index=perm) if b_gate is not None else None
    )

    A_tilde = x_tilde @ W_up_tilde
    if b_up_tilde is not None:
        A_tilde = A_tilde + b_up_tilde
    if c_up is not None:
        A_tilde = A_tilde + c_up
    B_tilde = x_tilde @ W_gate_tilde
    if b_gate_tilde is not None:
        B_tilde = B_tilde + b_gate_tilde
    if c_gate is not None:
        B_tilde = B_tilde + c_gate

    G_tilde = swiglu_reference(A_tilde, B_tilde)

    W_down_tilde = _permute_w_rows(w_down, perm) @ n_out
    Y_tilde = G_tilde @ W_down_tilde
    if b_down is not None:
        Y_tilde = Y_tilde + b_down @ n_out

    bundle = mitigation_bundle or DEFAULT_MITIGATION_BUNDLE
    bundle_meta = bundle_metadata(
        bundle, use_pad=pad_in is not None, online_extra_matmul_count=0
    )
    return {
        "y_plain": Y,
        "y_tilde": Y_tilde,
        "expected_y_tilde": Y @ n_out,
        "g_tilde": G_tilde,
        "g_plain_permuted": G.index_select(dim=-1, index=perm),
        "used_input_pad": pad_in is not None,
        "mitigation_bundle_metadata": bundle_meta,
    }


__all__ = [
    "fold_layernorm_affine_into_linear",
    "fold_rmsnorm_affine_into_linear",
    "gelu_reference",
    "get_activation",
    "layernorm_core",
    "relu_reference",
    "rmsnorm_core",
    "run_activation_permutation_island",
    "run_gelu_mlp_island",
    "run_layernorm_mean_preserving_island",
    "run_rmsnorm_orthogonal_island",
    "run_swiglu_mlp_island",
    "run_swiglu_paired_permutation_island",
    "silu_reference",
    "swiglu_reference",
]
