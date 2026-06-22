"""Tests for Amulet-style lifted nonlinear islands (CPU, float64)."""

from __future__ import annotations

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.ops.amulet_lifted_islands import (
    apply_feature_permutation,
    block_lift,
    inverse_permutation,
    make_relu_squeeze,
    make_selector_lift_params,
    run_layernorm_gadget_island,
    run_relu_lifted_mlp_island,
    run_selector_lifted_mlp_island,
    run_swiglu_selector_lifted_mlp_island,
)
from pllo.ops.amulet_lifted_islands import (
    make_layernorm_shift_gadget,
)
from pllo.ops.nonlinear_islands import (
    gelu_reference,
    layernorm_core,
    relu_reference,
    silu_reference,
)

ATOL = 1e-8
RTOL = 1e-8

# Small deterministic dimensions.
M, D, H, OUT, K = 6, 8, 10, 8, 4
DTYPE = torch.float64


def _seed(s: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(s)
    return g


def _randn(*shape: int, g: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=DTYPE)


def _masks(dim: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(s)
    return generate_invertible_matrix(dim, dtype=DTYPE)


# 1.
def test_block_lift_shapes() -> None:
    g = _seed(0)
    U = _randn(M, H, g=g)
    R = torch.rand(H, K, generator=g, dtype=DTYPE) + 0.5
    lifted = block_lift(U, R)
    assert lifted.shape == (M, H * K)
    # Spot-check the indexing identity.
    for j in range(H):
        for ell in range(K):
            torch.testing.assert_close(
                lifted[:, j * K + ell], U[:, j] * R[j, ell],
                atol=ATOL, rtol=RTOL,
            )


# 2.
def test_relu_lifted_squeeze_identity() -> None:
    g = _seed(1)
    U = _randn(M, H, g=g)
    R = torch.rand(H, K, generator=g, dtype=DTYPE) * 1.75 + 0.25
    S = make_relu_squeeze(R)
    out = relu_reference(block_lift(U, R)) @ S
    torch.testing.assert_close(out, relu_reference(U), atol=ATOL, rtol=RTOL)


# 3.
def test_gelu_selector_squeeze_identity() -> None:
    g = _seed(2)
    U = _randn(M, H, g=g)
    R, _valid, S = make_selector_lift_params(H, K, DTYPE, "cpu", g)
    out = gelu_reference(block_lift(U, R)) @ S
    torch.testing.assert_close(out, gelu_reference(U), atol=ATOL, rtol=RTOL)


# 4.
def test_silu_selector_squeeze_identity() -> None:
    g = _seed(3)
    U = _randn(M, H, g=g)
    R, _valid, S = make_selector_lift_params(H, K, DTYPE, "cpu", g)
    out = silu_reference(block_lift(U, R)) @ S
    torch.testing.assert_close(out, silu_reference(U), atol=ATOL, rtol=RTOL)


def _mlp_weights(s: int) -> tuple[torch.Tensor, ...]:
    g = _seed(s)
    x = _randn(M, D, g=g)
    w1 = _randn(D, H, g=g)
    b1 = _randn(H, g=g)
    w2 = _randn(H, OUT, g=g)
    b2 = _randn(OUT, g=g)
    return x, w1, b1, w2, b2


# 5.
def test_relu_lifted_mlp_correctness_no_pad() -> None:
    x, w1, b1, w2, b2 = _mlp_weights(10)
    n_in, n_in_inv = _masks(D, 100)
    n_out, _ = _masks(OUT, 101)
    res = run_relu_lifted_mlp_island(
        x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=K, seed=42,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    torch.testing.assert_close(
        res["squeeze_check"], res["expected_squeeze"], atol=ATOL, rtol=RTOL,
    )
    assert res["metadata"]["lift_dim"] == H * K
    assert res["metadata"]["used_pad"] is False


# 6.
def test_relu_lifted_mlp_correctness_with_pad() -> None:
    x, w1, b1, w2, b2 = _mlp_weights(11)
    n_in, n_in_inv = _masks(D, 102)
    n_out, _ = _masks(OUT, 103)
    pad_in = _randn(M, D, g=_seed(999))
    res = run_relu_lifted_mlp_island(
        x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=K, pad_in=pad_in, seed=7,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    assert res["metadata"]["used_pad"] is True


# 7.
def test_gelu_selector_lifted_mlp_correctness_no_pad() -> None:
    x, w1, b1, w2, b2 = _mlp_weights(12)
    n_in, n_in_inv = _masks(D, 104)
    n_out, _ = _masks(OUT, 105)
    res = run_selector_lifted_mlp_island(
        "gelu", x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=K, seed=3,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    torch.testing.assert_close(
        res["squeeze_check"], res["expected_squeeze"], atol=ATOL, rtol=RTOL,
    )


# 8.
def test_silu_selector_lifted_mlp_correctness_with_pad() -> None:
    x, w1, b1, w2, b2 = _mlp_weights(13)
    n_in, n_in_inv = _masks(D, 106)
    n_out, _ = _masks(OUT, 107)
    pad_in = _randn(M, D, g=_seed(888))
    res = run_selector_lifted_mlp_island(
        "silu", x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=K,
        pad_in=pad_in, seed=5,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    assert res["metadata"]["used_pad"] is True


def _swiglu_weights(s: int) -> tuple[torch.Tensor, ...]:
    g = _seed(s)
    x = _randn(M, D, g=g)
    w_up = _randn(D, H, g=g)
    b_up = _randn(H, g=g)
    w_gate = _randn(D, H, g=g)
    b_gate = _randn(H, g=g)
    w_down = _randn(H, OUT, g=g)
    b_down = _randn(OUT, g=g)
    return x, w_up, b_up, w_gate, b_gate, w_down, b_down


# 9.
def test_swiglu_selector_lifted_mlp_correctness_no_pad() -> None:
    x, w_up, b_up, w_gate, b_gate, w_down, b_down = _swiglu_weights(20)
    n_in, n_in_inv = _masks(D, 108)
    n_out, _ = _masks(OUT, 109)
    res = run_swiglu_selector_lifted_mlp_island(
        x, w_up, b_up, w_gate, b_gate, w_down, b_down,
        n_in, n_in_inv, n_out, k=K, seed=9,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    torch.testing.assert_close(
        res["squeeze_check"], res["expected_squeeze"], atol=ATOL, rtol=RTOL,
    )


# 10.
def test_swiglu_selector_lifted_mlp_correctness_with_pad() -> None:
    x, w_up, b_up, w_gate, b_gate, w_down, b_down = _swiglu_weights(21)
    n_in, n_in_inv = _masks(D, 110)
    n_out, _ = _masks(OUT, 111)
    pad_in = _randn(M, D, g=_seed(777))
    res = run_swiglu_selector_lifted_mlp_island(
        x, w_up, b_up, w_gate, b_gate, w_down, b_down,
        n_in, n_in_inv, n_out, k=K, pad_in=pad_in, seed=11,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    assert res["metadata"]["used_pad"] is True


# 11.
def test_layernorm_gadget_core_identity() -> None:
    g = _seed(30)
    x = _randn(M, D, g=g)
    perm = torch.randperm(D, generator=g)
    gadget = make_layernorm_shift_gadget(D, DTYPE, "cpu", g)
    eps = 1e-5
    # Shift invariance: LNCore(xG) == LNCore(x).
    core_shift = layernorm_core(x @ gadget, eps)
    torch.testing.assert_close(
        core_shift, layernorm_core(x, eps), atol=ATOL, rtol=RTOL,
    )
    # Permutation equivariance: LNCore(xG P) == LNCore(x)[:, perm].
    p_mat = torch.eye(D, dtype=DTYPE).index_select(1, perm)
    core_shift_perm = layernorm_core(x @ gadget @ p_mat, eps)
    torch.testing.assert_close(
        core_shift_perm, apply_feature_permutation(layernorm_core(x, eps), perm),
        atol=ATOL, rtol=RTOL,
    )


# 12.
def test_layernorm_gadget_island_correctness() -> None:
    g = _seed(31)
    x = _randn(M, D, g=g)
    norm_weight = _randn(D, g=g)
    norm_bias = _randn(D, g=g)
    linear_weight = _randn(D, OUT, g=g)
    linear_bias = _randn(OUT, g=g)
    n_in, n_in_inv = _masks(D, 112)
    n_out, _ = _masks(OUT, 113)
    res = run_layernorm_gadget_island(
        x, n_in, n_in_inv, norm_weight, norm_bias,
        linear_weight, linear_bias, n_out, eps=1e-5, seed=13,
    )
    torch.testing.assert_close(
        res["core_tilde"], res["expected_core_tilde"], atol=ATOL, rtol=RTOL,
    )
    torch.testing.assert_close(
        res["y_tilde"], res["expected_y_tilde"], atol=ATOL, rtol=RTOL,
    )
    assert res["metadata"]["scale_invariant_with_eps"] is True


# 13.
def test_selector_mode_reports_leakage_warning() -> None:
    x, w1, b1, w2, b2 = _mlp_weights(40)
    n_in, n_in_inv = _masks(D, 114)
    n_out, _ = _masks(OUT, 115)
    res = run_selector_lifted_mlp_island(
        "gelu", x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=K, seed=1,
    )
    md = res["metadata"]
    assert md["selector_rows_zero_for_decoys"] is True
    assert "selector_leakage_warning" in md
    assert "correctness prototype" in md["selector_leakage_warning"].lower()
    # The ReLU homogeneous island must NOT carry the selector warning.
    res_relu = run_relu_lifted_mlp_island(
        x, w1, b1, w2, b2, n_in, n_in_inv, n_out, k=K, seed=1,
    )
    assert res_relu["metadata"]["selector_rows_zero_for_decoys"] is False
    assert "selector_leakage_warning" not in res_relu["metadata"]


def test_inverse_permutation_roundtrip() -> None:
    g = _seed(50)
    perm = torch.randperm(H, generator=g)
    inv = inverse_permutation(perm)
    U = _randn(M, H, g=g)
    back = apply_feature_permutation(apply_feature_permutation(U, perm), inv)
    torch.testing.assert_close(back, U, atol=ATOL, rtol=RTOL)
