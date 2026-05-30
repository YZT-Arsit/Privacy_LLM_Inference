"""Tests for the Stage 5.1 trusted norm primitive."""

from __future__ import annotations

import pytest
import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.ops.norm import (
    TrustedNormConfig,
    layer_norm_reference,
    rms_norm_reference,
    trusted_norm_forward,
)


HIDDEN = 32
BATCH = 2
SEQ = 4
EPS = 1e-5


def _make_state(use_pad: bool, dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    device = torch.device("cpu")
    H = torch.randn(BATCH, SEQ, HIDDEN, dtype=dtype, device=device)
    flat = H.reshape(-1, HIDDEN)
    n_in, n_in_inv = generate_invertible_matrix(HIDDEN, dtype, device)
    n_out, _ = generate_invertible_matrix(HIDDEN, dtype, device)
    if use_pad:
        pad_in = generate_pad(tuple(flat.shape), dtype, device, 1.0)
        pad_out = generate_pad(tuple(flat.shape), dtype, device, 1.0)
        x_tilde = (flat - pad_in) @ n_in
    else:
        pad_in = None
        pad_out = None
        x_tilde = flat @ n_in
    return flat, x_tilde, n_in, n_in_inv, n_out, pad_in, pad_out


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [False, True])
def test_trusted_layernorm_correctness(use_pad: bool) -> None:
    flat, x_tilde, _, n_in_inv, n_out, pad_in, pad_out = _make_state(use_pad)
    weight = torch.randn(HIDDEN)
    bias = torch.randn(HIDDEN)
    result = trusted_norm_forward(
        x_tilde=x_tilde,
        n_in_inv=n_in_inv,
        norm_weight=weight,
        norm_bias=bias,
        n_out=n_out,
        norm_type="layernorm",
        eps=EPS,
        pad_in=pad_in,
        pad_out=pad_out,
    )
    expected = layer_norm_reference(flat, weight, bias, EPS)
    assert torch.allclose(result["y_plain"], expected, atol=1e-5)
    assert result["allclose"] is True
    assert result["max_abs_error"] < 1e-4


@pytest.mark.parametrize("use_pad", [False, True])
def test_trusted_rmsnorm_correctness(use_pad: bool) -> None:
    flat, x_tilde, _, n_in_inv, n_out, pad_in, pad_out = _make_state(use_pad)
    weight = torch.randn(HIDDEN)
    result = trusted_norm_forward(
        x_tilde=x_tilde,
        n_in_inv=n_in_inv,
        norm_weight=weight,
        norm_bias=None,
        n_out=n_out,
        norm_type="rmsnorm",
        eps=EPS,
        pad_in=pad_in,
        pad_out=pad_out,
    )
    expected = rms_norm_reference(flat, weight, EPS)
    assert torch.allclose(result["y_plain"], expected, atol=1e-5)
    assert result["allclose"] is True


# ---------------------------------------------------------------------------
# y_tilde shape invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("use_pad", [False, True])
def test_y_tilde_matches_remasked_y_plain(norm_type: str, use_pad: bool) -> None:
    """``y_tilde`` must equal ``Y N_out`` (no pad) or ``(Y - T_out) N_out``."""
    flat, x_tilde, _, n_in_inv, n_out, pad_in, pad_out = _make_state(use_pad)
    weight = torch.randn(HIDDEN)
    bias = torch.randn(HIDDEN) if norm_type == "layernorm" else None
    result = trusted_norm_forward(
        x_tilde=x_tilde,
        n_in_inv=n_in_inv,
        norm_weight=weight,
        norm_bias=bias,
        n_out=n_out,
        norm_type=norm_type,
        eps=EPS,
        pad_in=pad_in,
        pad_out=pad_out,
    )
    y_plain = result["y_plain"]
    if pad_out is None:
        expected_tilde = y_plain @ n_out
    else:
        expected_tilde = (y_plain - pad_out) @ n_out
    assert torch.allclose(result["y_tilde"], expected_tilde, atol=1e-5)


# ---------------------------------------------------------------------------
# Recovery: y_recovered ≈ y_plain
# ---------------------------------------------------------------------------


def test_recovered_output_matches_plaintext_norm() -> None:
    flat, x_tilde, _, n_in_inv, n_out, pad_in, pad_out = _make_state(use_pad=True)
    weight = torch.randn(HIDDEN)
    bias = torch.randn(HIDDEN)
    result = trusted_norm_forward(
        x_tilde=x_tilde,
        n_in_inv=n_in_inv,
        norm_weight=weight,
        norm_bias=bias,
        n_out=n_out,
        norm_type="layernorm",
        eps=EPS,
        pad_in=pad_in,
        pad_out=pad_out,
    )
    expected = layer_norm_reference(flat, weight, bias, EPS)
    assert torch.allclose(result["y_recovered"], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# bias=None path for RMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_rejects_bias() -> None:
    flat, x_tilde, _, n_in_inv, n_out, pad_in, pad_out = _make_state(use_pad=False)
    with pytest.raises(ValueError, match="bias"):
        trusted_norm_forward(
            x_tilde=x_tilde,
            n_in_inv=n_in_inv,
            norm_weight=None,
            norm_bias=torch.zeros(HIDDEN),
            n_out=n_out,
            norm_type="rmsnorm",
            eps=EPS,
        )


def test_rmsnorm_with_no_weight_and_no_bias() -> None:
    """RMSNorm path must run with weight=None (gamma=1)."""
    flat, x_tilde, _, n_in_inv, n_out, _, _ = _make_state(use_pad=False)
    result = trusted_norm_forward(
        x_tilde=x_tilde,
        n_in_inv=n_in_inv,
        norm_weight=None,
        norm_bias=None,
        n_out=n_out,
        norm_type="rmsnorm",
        eps=EPS,
    )
    expected = rms_norm_reference(flat, None, EPS)
    assert torch.allclose(result["y_plain"], expected, atol=1e-5)
    assert result["allclose"] is True


# ---------------------------------------------------------------------------
# TrustedNormConfig dataclass smoke
# ---------------------------------------------------------------------------


def test_trusted_norm_config_defaults() -> None:
    cfg = TrustedNormConfig(norm_type="layernorm", hidden_size=128)
    assert cfg.eps == 1e-5
    assert cfg.use_pad is True
    assert cfg.dtype == "float32"


def test_unknown_norm_type_rejected() -> None:
    flat, x_tilde, _, n_in_inv, n_out, _, _ = _make_state(use_pad=False)
    with pytest.raises(ValueError, match="norm_type"):
        trusted_norm_forward(
            x_tilde=x_tilde,
            n_in_inv=n_in_inv,
            norm_weight=None,
            norm_bias=None,
            n_out=n_out,
            norm_type="batchnorm",
            eps=EPS,
        )
