"""Tests for the Stage 5.2 nonlinear island ops."""

from __future__ import annotations

import pytest
import torch

from pllo.ops.compatible_masks import (
    generate_dense_invertible,
    generate_mean_preserving_orthogonal,
    generate_orthogonal,
    generate_permutation,
)
from pllo.ops.nonlinear_islands import (
    fold_layernorm_affine_into_linear,
    fold_rmsnorm_affine_into_linear,
    layernorm_core,
    rmsnorm_core,
    run_activation_permutation_island,
    run_gelu_mlp_island,
    run_layernorm_mean_preserving_island,
    run_rmsnorm_orthogonal_island,
    run_swiglu_mlp_island,
    run_swiglu_paired_permutation_island,
)


BATCH = 2
SEQ = 4
HIDDEN = 16
INTERMEDIATE = 64
TOL = 1e-4


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, HIDDEN)


# ---------------------------------------------------------------------------
# Norm islands
# ---------------------------------------------------------------------------


def test_rmsnorm_orthogonal_island_with_affine_folding() -> None:
    x = _x()
    N = generate_orthogonal(HIDDEN, torch.float32, "cpu")
    n_out, _ = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    gamma = torch.randn(HIDDEN)
    W = torch.randn(HIDDEN, HIDDEN)
    b = torch.randn(HIDDEN)
    r = run_rmsnorm_orthogonal_island(x, N, gamma, W, b, n_out)
    err = (r["y_tilde"] - r["expected_y_tilde"]).abs().max().item()
    assert err < TOL


def test_layernorm_mean_preserving_island_with_affine_folding() -> None:
    x = _x()
    N = generate_mean_preserving_orthogonal(HIDDEN, torch.float32, "cpu")
    n_out, _ = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    gamma = torch.randn(HIDDEN)
    beta = torch.randn(HIDDEN)
    W = torch.randn(HIDDEN, HIDDEN)
    b = torch.randn(HIDDEN)
    r = run_layernorm_mean_preserving_island(x, N, gamma, beta, W, b, n_out)
    err = (r["y_tilde"] - r["expected_y_tilde"]).abs().max().item()
    assert err < TOL


def test_affine_folding_helpers_are_correct() -> None:
    gamma = torch.randn(HIDDEN)
    beta = torch.randn(HIDDEN)
    W = torch.randn(HIDDEN, HIDDEN)
    b = torch.randn(HIDDEN)
    W_f, b_f = fold_layernorm_affine_into_linear(gamma, beta, W, b)
    # Independent reference: ((gamma * x + beta) W + b) for some x.
    x = torch.randn(2, HIDDEN)
    lhs = (gamma * x + beta) @ W + b
    rhs = x @ W_f + b_f
    assert torch.allclose(lhs, rhs, atol=1e-5)
    # RMSNorm folding (no beta).
    W_f2, b_f2 = fold_rmsnorm_affine_into_linear(gamma, W, b)
    lhs = (gamma * x) @ W + b
    rhs = x @ W_f2 + b_f2
    assert torch.allclose(lhs, rhs, atol=1e-5)


# ---------------------------------------------------------------------------
# Activation permutation islands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
def test_activation_permutation_island(activation: str) -> None:
    torch.manual_seed(0)
    z = torch.randn(BATCH, SEQ, HIDDEN)
    perm = generate_permutation(HIDDEN, torch.float32, "cpu")["perm"]
    r = run_activation_permutation_island(z, perm, activation)
    err = (r["lhs"] - r["rhs"]).abs().max().item()
    assert err == 0.0, (
        f"activation {activation} must commute exactly with permutation; got {err}"
    )


def test_swiglu_paired_permutation_island() -> None:
    torch.manual_seed(0)
    a = torch.randn(BATCH, SEQ, HIDDEN)
    b = torch.randn(BATCH, SEQ, HIDDEN)
    perm = generate_permutation(HIDDEN, torch.float32, "cpu")["perm"]
    r = run_swiglu_paired_permutation_island(a, b, perm)
    err = (r["lhs"] - r["rhs"]).abs().max().item()
    assert err == 0.0


# ---------------------------------------------------------------------------
# Full MLP islands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
def test_gelu_mlp_island_no_pad(activation: str) -> None:
    torch.manual_seed(1)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    n_in, n_in_inv = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    n_out, _ = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    perm = generate_permutation(INTERMEDIATE, torch.float32, "cpu")["perm"]
    scale = INTERMEDIATE ** -0.5
    W1 = torch.randn(HIDDEN, INTERMEDIATE) * scale
    b1 = torch.randn(INTERMEDIATE) * scale
    W2 = torch.randn(INTERMEDIATE, HIDDEN) * scale
    b2 = torch.randn(HIDDEN) * scale
    r = run_gelu_mlp_island(x, W1, b1, W2, b2, n_in, n_in_inv, perm, n_out, activation)
    err = (r["y_tilde"] - r["expected_y_tilde"]).abs().max().item()
    assert err < TOL


def test_swiglu_mlp_island_no_pad() -> None:
    torch.manual_seed(1)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    n_in, n_in_inv = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    n_out, _ = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    perm = generate_permutation(INTERMEDIATE, torch.float32, "cpu")["perm"]
    scale = INTERMEDIATE ** -0.5
    W_up = torch.randn(HIDDEN, INTERMEDIATE) * scale
    b_up = torch.randn(INTERMEDIATE) * scale
    W_gate = torch.randn(HIDDEN, INTERMEDIATE) * scale
    b_gate = torch.randn(INTERMEDIATE) * scale
    W_down = torch.randn(INTERMEDIATE, HIDDEN) * scale
    b_down = torch.randn(HIDDEN) * scale
    r = run_swiglu_mlp_island(
        x, W_up, b_up, W_gate, b_gate, W_down, b_down,
        n_in, n_in_inv, perm, n_out,
    )
    err = (r["y_tilde"] - r["expected_y_tilde"]).abs().max().item()
    assert err < TOL


def test_mlp_island_with_input_pad_compensation() -> None:
    """Pad at the first Linear boundary must be compensated by ``T W P``."""
    torch.manual_seed(2)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    pad = torch.randn(BATCH, SEQ, HIDDEN)
    n_in, n_in_inv = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    n_out, _ = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    perm = generate_permutation(INTERMEDIATE, torch.float32, "cpu")["perm"]
    scale = INTERMEDIATE ** -0.5
    W1 = torch.randn(HIDDEN, INTERMEDIATE) * scale
    b1 = torch.randn(INTERMEDIATE) * scale
    W2 = torch.randn(INTERMEDIATE, HIDDEN) * scale
    b2 = torch.randn(HIDDEN) * scale
    r = run_gelu_mlp_island(
        x, W1, b1, W2, b2,
        n_in, n_in_inv, perm, n_out, "gelu", pad_in=pad,
    )
    err = (r["y_tilde"] - r["expected_y_tilde"]).abs().max().item()
    assert err < TOL
    # ``Z_tilde`` must equal ``Z[:, perm]`` once pad compensation lands —
    # i.e. the value at the activation input must be pad-free.
    assert torch.allclose(r["z_tilde"], r["z_plain_permuted"], atol=TOL)


def test_pad_is_not_pushed_through_activation() -> None:
    """``activation((Z - T) P) ≠ activation(Z P)`` for any nontrivial T."""
    torch.manual_seed(3)
    Z = torch.randn(BATCH, SEQ, INTERMEDIATE)
    T = torch.randn(BATCH, SEQ, INTERMEDIATE)
    perm = generate_permutation(INTERMEDIATE, torch.float32, "cpu")["perm"]
    Z_perm = Z.index_select(dim=-1, index=perm)
    T_perm = T.index_select(dim=-1, index=perm)
    from pllo.ops.nonlinear_islands import gelu_reference
    lhs = gelu_reference(Z_perm - T_perm)
    rhs = gelu_reference(Z_perm)
    # The post-pad activation is *not* the same as the no-pad activation —
    # which is exactly why the Stage 5.2 protocol forbids pushing pad through
    # the activation. The pad must be compensated at the Linear boundary.
    assert (lhs - rhs).abs().max().item() > 1e-3


# ---------------------------------------------------------------------------
# Reference cores (sanity checks)
# ---------------------------------------------------------------------------


def test_rmsnorm_core_under_orthogonal_mask_equals_core_then_mask() -> None:
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    N = generate_orthogonal(HIDDEN, torch.float32, "cpu")
    lhs = rmsnorm_core(x @ N)
    rhs = rmsnorm_core(x) @ N
    assert (lhs - rhs).abs().max().item() < TOL


def test_layernorm_core_under_mean_preserving_orthogonal_mask() -> None:
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    N = generate_mean_preserving_orthogonal(HIDDEN, torch.float32, "cpu")
    lhs = layernorm_core(x @ N)
    rhs = layernorm_core(x) @ N
    assert (lhs - rhs).abs().max().item() < TOL
