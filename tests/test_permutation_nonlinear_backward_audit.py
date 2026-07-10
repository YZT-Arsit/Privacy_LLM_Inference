"""Tests for the permutation nonlinear-island forward/backward + boundary audit."""

from __future__ import annotations

from pllo.experiments.permutation_nonlinear_backward_audit import (
    autograd_nonlinear_backward,
    cross_gram_boundary,
    lora_compatibility,
    mlp_forward_backward,
    stabilizer_test,
    tiny_transformer_mlp_integration,
)

PASS = 1e-8
FAIL = 1e-3


def test_gelu_permutation_passes_dense_fails():
    assert stabilizer_test("gelu", "permutation")["rel_error"] < PASS
    assert stabilizer_test("gelu", "dense_orthogonal")["rel_error"] > FAIL
    assert stabilizer_test("gelu", "dense_gl")["rel_error"] > FAIL


def test_silu_permutation_passes_signed_fails():
    assert stabilizer_test("silu", "permutation")["rel_error"] < PASS
    assert stabilizer_test("silu", "signed_permutation")["rel_error"] > FAIL


def test_relu_positive_diagonal_passes_gelu_fails():
    assert stabilizer_test("relu", "positive_diagonal")["rel_error"] < PASS
    assert stabilizer_test("gelu", "positive_diagonal")["rel_error"] > FAIL
    # ReLU still fails signed permutation (negative signs present)
    assert stabilizer_test("relu", "signed_permutation")["rel_error"] > FAIL


def test_mlp_permutation_forward_correctness():
    for act in ("gelu", "silu", "relu"):
        r = mlp_forward_backward(act)
        assert r["forward_z_error"] < PASS
        assert r["forward_u_error"] < PASS
        assert r["forward_y_error"] < PASS


def test_mlp_permutation_backward_correctness():
    for act in ("gelu", "silu", "relu"):
        r = mlp_forward_backward(act)
        assert r["backward_gu_error"] < PASS
        assert r["backward_gz_error"] < PASS
        assert r["backward_gh_error"] < PASS


def test_standard_autograd_nonlinear_backward_in_permutation_domain():
    for act in ("gelu", "silu", "relu"):
        r = autograd_nonlinear_backward(act)
        assert r["pass_bool"]
        assert r["custom_primitive_used"] is False
    # autograd path and manual phi' path must agree
    a = mlp_forward_backward("silu", use_autograd_nonlinear=True)
    b = mlp_forward_backward("silu", use_autograd_nonlinear=False)
    assert a["backward_gz_error"] < PASS and b["backward_gz_error"] < PASS


def test_general_region_cross_gram_not_exact():
    r = cross_gram_boundary(trials=50)
    assert r["mean_rel_error_general"] > FAIL
    assert r["general_region_exact"] is False


def test_nonlinear_region_cross_gram_exact():
    r = cross_gram_boundary(trials=50)
    assert r["mean_rel_error_z"] < PASS
    assert r["mean_rel_error_u"] < PASS
    assert r["mean_corr_z"] > 1 - 1e-6
    assert r["nonlinear_region_exact"] is True


def test_lora_up_down_permutation_compatibility():
    for act in ("gelu", "silu"):
        r = lora_compatibility(act)
        assert r["up_pass"]
        assert r["down_pass"]
        assert r["rank_independent_of_pi"]
        # rank space and activation-permutation space are distinct dims
        assert r["activation_perm_dim"] not in r["rank_space_dims"]


def test_tiny_transformer_integration_forward_backward():
    r = tiny_transformer_mlp_integration()
    if r.get("synthetic_mlp_only"):
        return  # skipped: components unavailable
    assert r["forward_pass"]
    assert r["autograd_backward_pass"]
