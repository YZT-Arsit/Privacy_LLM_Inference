"""Tests for Variant F — Selective Token-Mixing Shield (STMS).

Lock in the mechanics (they do NOT assert STMS defeats the oracle — orthogonal A
provably does not):
  * boundary roundtrip H N = A^{-1} U is exact (orthogonal / non-orthogonal / block
    / shield); shield rows are dropped before compute
  * orthogonal token-mix leaves the singular-value spectrum invariant (why the
    spectrum oracle still works)
  * non-orthogonal token-mix distorts the spectrum
  * free-A oracle overfits (residual ~0 for any full-rank candidate)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pllo.experiments.public_weight_prompt_audit import qr_orthogonal  # noqa: E402
from pllo.experiments.stms_boundary_shield import (  # noqa: E402
    oracle_free_AQ_residual, oracle_spectrum_residual, sample_shield_rows,
    stms_forward, stms_recover, stms_report_fields, token_mix_blocklocal,
    token_mix_nonorthogonal, token_mix_orthogonal,
)


def _hn(m=8, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(m, d, dtype=torch.float64, generator=g)
    n = qr_orthogonal(d, seed=seed + 1, dtype=torch.float64)
    return h, n


@pytest.mark.parametrize("case", ["orthogonal", "nonorthogonal", "blocklocal"])
def test_roundtrip_exact(case):
    h, n = _hn()
    m = h.shape[0]
    if case == "orthogonal":
        a = token_mix_orthogonal(m, seed=1)
    elif case == "nonorthogonal":
        a = token_mix_nonorthogonal(m, seed=1, cond=10.0)
    else:
        a = token_mix_blocklocal(m, block=4, seed=1)
    fwd = stms_forward(h, n, a)
    rec = stms_recover(fwd["u"], a, num_real_tokens=m)
    assert float((rec - fwd["hn"]).abs().max()) <= 1e-9


def test_roundtrip_with_shield_drops_rows():
    h, n = _hn(m=6, d=32)
    m, d = h.shape
    k = 3
    a = token_mix_orthogonal(m + k, seed=2)
    shield = sample_shield_rows(k, d, scale=7.0, seed=3)
    fwd = stms_forward(h, n, a, shield=shield)
    assert fwd["u"].shape[0] == m + k
    rec = stms_recover(fwd["u"], a, num_real_tokens=m)
    assert rec.shape[0] == m                     # shield rows dropped
    assert float((rec - fwd["hn"]).abs().max()) <= 1e-9


def test_orthogonal_mix_preserves_singular_spectrum():
    h, n = _hn()
    m = h.shape[0]
    a = token_mix_orthogonal(m, seed=5)
    u = stms_forward(h, n, a)["u"]
    # singular values of U == singular values of H  -> spectrum oracle sees the true one
    su = torch.linalg.svdvals(u)
    sh = torch.linalg.svdvals(h)
    assert float((su - sh).abs().max()) <= 1e-10
    # spectrum residual of the true candidate is ~0
    assert oracle_spectrum_residual(h, u) <= 1e-10


def test_nonorthogonal_mix_distorts_spectrum():
    h, n = _hn()
    m = h.shape[0]
    a = token_mix_nonorthogonal(m, seed=5, cond=50.0)
    u = stms_forward(h, n, a)["u"]
    # non-orthogonal A changes the singular values -> spectrum no longer matches
    assert oracle_spectrum_residual(h, u) > 1e-2


def test_free_AQ_oracle_overfits():
    # free [m,m] A AND free orthogonal Q -> residual ~0 for ANY full-rank candidate
    h, n = _hn(m=6, d=32, seed=0)
    m = h.shape[0]
    a = token_mix_orthogonal(m, seed=1)
    u = stms_forward(h, n, a)["u"]
    other, _ = _hn(m=6, d=32, seed=99)           # a DIFFERENT prompt
    assert oracle_free_AQ_residual(other, u) <= 1e-6   # fits the wrong candidate -> useless attack
    assert oracle_free_AQ_residual(h, u) <= 1e-6       # and the true one


def test_report_fields_hard_invariants():
    f = stms_report_fields(roundtrip_max_abs=1e-13, exact_lossless_tests_passed=True)
    assert f["crosses_attention"] is False
    assert f["crosses_kv_cache"] is False
    assert f["crosses_rmsnorm"] is False
    assert f["crosses_rope"] is False
    assert f["protects_compute_visible_HN"] is False
    assert f["uses_token_dim_mixing"] is True
    assert f["formal_security_claim"] is False
    assert f["production_qwen7b_integration"] is False
