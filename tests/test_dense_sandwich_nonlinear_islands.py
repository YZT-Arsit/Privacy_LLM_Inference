"""Stage 5.3e — Dense-sandwich integration tests for the Stage 5.2a islands."""

from __future__ import annotations

import pytest
import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.ops.compatible_masks import generate_permutation
from pllo.ops.nonlinear_islands import (
    run_gelu_mlp_island,
    run_swiglu_mlp_island,
)


def _build_gelu_inputs(hidden: int = 32, inter: int = 64, batch: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(batch, hidden)
    w1 = torch.randn(hidden, inter) * 0.1
    b1 = torch.randn(inter) * 0.05
    w2 = torch.randn(inter, hidden) * 0.1
    b2 = torch.randn(hidden) * 0.05
    n_in, n_in_inv = generate_invertible_matrix(hidden, torch.float32, "cpu")
    n_out, n_out_inv = generate_invertible_matrix(hidden, torch.float32, "cpu")
    perm = generate_permutation(inter, dtype=torch.float32, device="cpu")["perm"]
    return x, w1, b1, w2, b2, n_in, n_in_inv, perm, n_out, n_out_inv


def _build_swiglu_inputs(hidden: int = 32, inter: int = 64, batch: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(batch, hidden)
    w_up = torch.randn(hidden, inter) * 0.1
    w_gate = torch.randn(hidden, inter) * 0.1
    w_down = torch.randn(inter, hidden) * 0.1
    n_in, n_in_inv = generate_invertible_matrix(hidden, torch.float32, "cpu")
    n_out, n_out_inv = generate_invertible_matrix(hidden, torch.float32, "cpu")
    perm = generate_permutation(inter, dtype=torch.float32, device="cpu")["perm"]
    return x, w_up, w_gate, w_down, n_in, n_in_inv, perm, n_out, n_out_inv


# ---------------------------------------------------------------------------
# GELU MLP island
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [False, True])
def test_gelu_mlp_full_bundle_allclose(use_pad: bool) -> None:
    x, w1, b1, w2, b2, n_in, n_in_inv, perm, n_out, n_out_inv = _build_gelu_inputs()
    pad = torch.randn_like(x) if use_pad else None
    r = run_gelu_mlp_island(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        activation_type="gelu", pad_in=pad,
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    recovered = r["y_tilde"] @ n_out_inv
    assert torch.allclose(recovered, r["y_plain"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(r["y_tilde"], r["expected_y_tilde"], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("use_pad", [False, True])
def test_gelu_mlp_full_bundle_metadata(use_pad: bool) -> None:
    x, w1, b1, w2, b2, n_in, n_in_inv, perm, n_out, _ = _build_gelu_inputs()
    pad = torch.randn_like(x) if use_pad else None
    r = run_gelu_mlp_island(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        activation_type="gelu", pad_in=pad,
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    meta = r["mitigation_bundle_metadata"]
    assert meta["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"
    assert meta["dense_sandwich_enabled"] is True
    assert meta["fresh_permutation_enabled"] is True
    assert meta["boundary_pad_required"] is True
    assert meta["boundary_pad_enabled"] is use_pad
    assert meta["activation_input_form"] == "ZP"
    assert meta["activation_pad_forbidden"] is True
    assert meta["pad_placement"] == ("linear_boundary_only" if use_pad else "n/a")
    assert meta["online_extra_matmul_count"] == 0
    assert meta["default_on_candidate_under_stage_5_4"] is use_pad


def test_gelu_default_bundle_metadata_preserves_legacy_flags() -> None:
    x, w1, b1, w2, b2, n_in, n_in_inv, perm, n_out, _ = _build_gelu_inputs()
    r = run_gelu_mlp_island(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        activation_type="gelu", pad_in=None,
        # No mitigation_bundle ⇒ default.
    )
    meta = r["mitigation_bundle_metadata"]
    assert meta["mitigation_bundle"] == "fresh_perm_only"
    assert meta["dense_sandwich_enabled"] is False
    assert meta["activation_input_form"] == "ZP"
    assert meta["online_extra_matmul_count"] == 0


# ---------------------------------------------------------------------------
# SwiGLU MLP island
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [False, True])
def test_swiglu_mlp_full_bundle_allclose(use_pad: bool) -> None:
    x, w_up, w_gate, w_down, n_in, n_in_inv, perm, n_out, n_out_inv = _build_swiglu_inputs()
    pad = torch.randn_like(x) if use_pad else None
    r = run_swiglu_mlp_island(
        x=x, w_up=w_up, b_up=None, w_gate=w_gate, b_gate=None,
        w_down=w_down, b_down=None,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        pad_in=pad, mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    recovered = r["y_tilde"] @ n_out_inv
    assert torch.allclose(recovered, r["y_plain"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(r["y_tilde"], r["expected_y_tilde"], atol=1e-4, rtol=1e-4)


def test_swiglu_full_bundle_metadata_dense_sandwich_on() -> None:
    x, w_up, w_gate, w_down, n_in, n_in_inv, perm, n_out, _ = _build_swiglu_inputs()
    r = run_swiglu_mlp_island(
        x=x, w_up=w_up, b_up=None, w_gate=w_gate, b_gate=None,
        w_down=w_down, b_down=None,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        pad_in=torch.randn_like(x),
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    meta = r["mitigation_bundle_metadata"]
    assert meta["dense_sandwich_enabled"] is True
    assert meta["activation_input_form"] == "ZP"
    assert meta["activation_pad_forbidden"] is True
    assert meta["online_extra_matmul_count"] == 0
    assert meta["default_on_candidate_under_stage_5_4"] is True


def test_island_math_is_bundle_invariant() -> None:
    """Y_tilde must equal across both bundles (bundle is metadata, not math)."""
    x, w1, b1, w2, b2, n_in, n_in_inv, perm, n_out, _ = _build_gelu_inputs()
    pad = torch.randn_like(x)
    r_a = run_gelu_mlp_island(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        activation_type="gelu", pad_in=pad,
        mitigation_bundle="fresh_perm_only",
    )
    r_b = run_gelu_mlp_island(
        x=x, w1=w1, b1=b1, w2=w2, b2=b2,
        n_in=n_in, n_in_inv=n_in_inv, permutation=perm, n_out=n_out,
        activation_type="gelu", pad_in=pad,
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    assert torch.equal(r_a["y_tilde"], r_b["y_tilde"])
    assert torch.equal(r_a["y_plain"], r_b["y_plain"])
    # But metadata differs.
    assert (
        r_a["mitigation_bundle_metadata"]["dense_sandwich_enabled"]
        != r_b["mitigation_bundle_metadata"]["dense_sandwich_enabled"]
    )
