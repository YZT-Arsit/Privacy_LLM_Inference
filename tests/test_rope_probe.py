"""Stage 6.4 — RoPE probe tests."""

from __future__ import annotations

import math

import pytest
import torch

from pllo.experiments.rope_probe import (
    RopeProbeConfig,
    apply_rope,
    run_rope_probe,
)


# ---------------------------------------------------------------------------
# apply_rope sanity
# ---------------------------------------------------------------------------


def test_apply_rope_preserves_shape_and_norm() -> None:
    x = torch.randn(2, 4, 8, 16)   # [B, H, S, D]
    y = apply_rope(x)
    assert y.shape == x.shape
    # RoPE is unitary per-pair so per-vector L2 norm is preserved.
    per_vec_in = x.reshape(-1, 16).norm(dim=-1)
    per_vec_out = y.reshape(-1, 16).norm(dim=-1)
    assert torch.allclose(per_vec_in, per_vec_out, atol=1e-4)


def test_apply_rope_supports_bsd_layout() -> None:
    x = torch.randn(2, 8, 16)   # [B, S, D]
    y = apply_rope(x)
    assert y.shape == x.shape


def test_apply_rope_rejects_odd_head_dim() -> None:
    x = torch.randn(2, 4, 8, 15)
    with pytest.raises(ValueError):
        apply_rope(x)


def test_apply_rope_zero_position_is_identity() -> None:
    x = torch.randn(2, 4, 1, 16)
    y = apply_rope(x, positions=torch.tensor([0]))
    assert torch.allclose(y, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Probe A — post-RoPE masking invariant
# ---------------------------------------------------------------------------


def test_rope_post_mask_invariant_passes() -> None:
    report = run_rope_probe(
        RopeProbeConfig(batch_size=2, num_heads=4, seq_len=8, head_dim=16)
    )
    assert report["status"] == "ok"
    a = report["probe_a_post_rope_masking_invariant"]
    assert a["allclose"] is True
    assert a["max_abs_error"] < 1e-3
    # QK constraint N_Q N_K^T = I must hold to working precision.
    assert report["qk_constraint_error"] < 1e-3


def test_rope_odd_head_dim_skipped_with_reason() -> None:
    report = run_rope_probe(RopeProbeConfig(head_dim=15))
    assert report["status"] == "skipped"
    assert "even" in report["reason"]


# ---------------------------------------------------------------------------
# Probe B — pre-RoPE mask commutation (feasibility / negative result)
# ---------------------------------------------------------------------------


def test_rope_pre_mask_dense_does_not_commute() -> None:
    report = run_rope_probe(
        RopeProbeConfig(batch_size=2, num_heads=4, seq_len=8, head_dim=16)
    )
    fam = report["probe_b_pre_rope_mask_commutation"]["per_family"]
    dense = fam["dense_invertible"]
    assert dense["expected_behavior"] == "expected_failure"
    assert dense["commutes"] is False, (
        "dense random masks should NOT commute with RoPE"
    )


def test_rope_pre_mask_orthogonal_generally_does_not_commute() -> None:
    report = run_rope_probe(
        RopeProbeConfig(batch_size=2, num_heads=4, seq_len=8, head_dim=16)
    )
    fam = report["probe_b_pre_rope_mask_commutation"]["per_family"]
    orth = fam["orthogonal"]
    assert orth["expected_behavior"] == "expected_failure"
    assert orth["commutes"] is False


def test_rope_pre_mask_block_diagonal_rotation_commutes() -> None:
    report = run_rope_probe(
        RopeProbeConfig(batch_size=2, num_heads=4, seq_len=8, head_dim=16)
    )
    fam = report["probe_b_pre_rope_mask_commutation"]["per_family"]
    bd = fam["block_diagonal_rotation"]
    assert bd["expected_behavior"] == "expected_to_commute"
    assert bd["commutes"] is True, (
        "block-diagonal 2D rotation in RoPE planes must commute"
    )
