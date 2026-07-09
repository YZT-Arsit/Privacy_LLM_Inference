"""Tests for the public-weight prompt-privacy audit.

These lock in the *mechanics* the audit relies on (they do NOT assert the claim
succeeds — V4 is expected to break it):

  * dense orthogonal Q is exactly orthogonal (cond ~1)
  * SiLU/Amulet boundary is exact at fp64; dense keymat cannot cross SiLU
  * orthogonal Procrustes residual is 0 iff token-token Gram matches, and is
    invariant to the choice of orthogonal Q (the V4 crux)
  * known-plaintext recovers a static Q once known rows >= d
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pllo.experiments.public_weight_prompt_audit import (  # noqa: E402
    orthogonal_procrustes_residual,
    qr_orthogonal,
    verify1_silu_amulet_boundary,
)
from pllo.ops.nonlinear_islands import silu_reference  # noqa: E402
from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat  # noqa: E402


def test_qr_orthogonal_is_orthogonal():
    q = qr_orthogonal(128, seed=3, dtype=torch.float64)
    err = float((q @ q.transpose(0, 1) - torch.eye(128, dtype=torch.float64)).abs().max())
    assert err <= 1e-12
    assert float(torch.linalg.cond(q)) < 1.0 + 1e-9


def test_procrustes_zero_iff_gram_matches_and_Q_invariant():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(6, 32, dtype=torch.float64, generator=g)
    # invariant to the choice of orthogonal Q -> residual ~0 for the true prompt
    for s in (1, 2, 3):
        q = qr_orthogonal(32, seed=s, dtype=torch.float64)
        assert orthogonal_procrustes_residual(h, h @ q) <= 1e-9
    # a different prompt (different Gram) -> residual bounded away from 0
    h2 = torch.randn(6, 32, dtype=torch.float64, generator=g)
    q = qr_orthogonal(32, seed=7, dtype=torch.float64)
    assert orthogonal_procrustes_residual(h2, h @ q) > 1e-2


def test_procrustes_length_mismatch_is_inf():
    a = torch.randn(4, 16, dtype=torch.float64)
    b = torch.randn(5, 16, dtype=torch.float64)
    assert orthogonal_procrustes_residual(a, b) == float("inf")


def test_dense_keymat_cannot_cross_silu():
    d = 48
    p, qinv, _ = sample_nonorthogonal_keymat(d, 0.3, seed=5, dtype=torch.float64)
    x = torch.randn(12, d, dtype=torch.float64)
    assert float((silu_reference(x @ p) @ qinv - silu_reference(x)).abs().max()) > 1e-2


def test_v1_boundary_fp64_exact():
    # small sweep for speed; fp64 must be exact and the negative control large
    v1 = verify1_silu_amulet_boundary(dims=(64,), seqs=(1, 16), seeds=(0,),
                                      activations=("silu", "gelu"))
    assert v1["fp64"]["max_abs_error"] <= 1e-9
    assert v1["claim_survives"] is True
    assert v1["negative_control_dense_keymat"]["silu_xP_Pinv_vs_silu_x_max_abs"] > 1e-2
    assert v1["native_fp32_build_probe"]["native_fp32_build"] == "failed_as_expected"


def test_known_plaintext_recovers_static_Q_at_full_rank():
    d = 64
    q = qr_orthogonal(d, seed=1, dtype=torch.float64)
    g = torch.Generator().manual_seed(2)
    # fewer known rows than d -> underdetermined, Q not recovered
    a_small = torch.randn(d // 2, d, dtype=torch.float64, generator=g)
    q_hat_small = torch.linalg.lstsq(a_small, a_small @ q).solution
    assert float((q_hat_small - q).abs().max()) > 1e-3
    # >= d known rows -> Q recovered exactly
    a_full = torch.randn(d + 8, d, dtype=torch.float64, generator=g)
    q_hat_full = torch.linalg.lstsq(a_full, a_full @ q).solution
    assert float((q_hat_full - q).abs().max()) <= 1e-8
