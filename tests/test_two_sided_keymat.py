"""Tests for Variant D — exact two-sided non-orthogonal key-matrix transform.

Covers the identities requested for the study:
  1. keymat_inverse_correctness   max|P Q - I| <= tol
  2. two_sided_linear_correctness X_tilde W_tilde + b_tilde == (XW+b) P_out
  3. linear_chain_correctness     chaining P_out(L) = P_in(L+1) stays exact
  4. precision_sweep              fp64/fp32 exact; bf16 reported (not asserted)
  5. scope_negative               a non-orthogonal P does NOT pass a nonlinearity
     exactly (so end-to-end through GELU/SiLU is correctly NOT claimed)
  6. report_fields                exact_lossless_claim only true when passed in
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pllo.ops.two_sided_keymat import (  # noqa: E402
    fold_bias,
    fold_weight,
    sample_nonorthogonal_keymat,
    two_sided_keymat_report_fields,
    two_sided_linear,
)

LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0)


@pytest.mark.parametrize("lam", LAMBDAS)
def test_keymat_inverse_correctness(lam):
    p, q, diag = sample_nonorthogonal_keymat(48, lam, seed=0, dtype=torch.float64)
    err = float((p @ q - torch.eye(48, dtype=torch.float64)).abs().max())
    assert err <= 1e-9
    assert diag.inverse_max_abs_error <= 1e-9
    assert diag.cond >= 1.0


@pytest.mark.parametrize("lam", LAMBDAS)
def test_two_sided_linear_correctness(lam):
    d = 32
    p_in, q_in, _ = sample_nonorthogonal_keymat(d, lam, seed=1, dtype=torch.float64)
    p_out, _, _ = sample_nonorthogonal_keymat(d, lam, seed=2, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, d, dtype=torch.float64, generator=g)
    w = torch.randn(d, d, dtype=torch.float64, generator=g)
    b = torch.randn(d, dtype=torch.float64, generator=g)
    r = two_sided_linear(x, w, b, p_in, q_in, p_out)
    err = float((r["y_tilde"] - r["y_ref"]).abs().max())
    assert err <= 1e-9
    # y_ref must equal the plaintext output masked by P_out (chaining invariant)
    assert float((r["y_ref"] - (x @ w + b) @ p_out).abs().max()) <= 1e-12


@pytest.mark.parametrize("lam", LAMBDAS)
def test_linear_chain_correctness(lam):
    """Two stacked linear layers: the stable state carries P through exactly when
    layer L+1's P_in == layer L's P_out."""
    d = 24
    p0, q0, _ = sample_nonorthogonal_keymat(d, lam, seed=10, dtype=torch.float64)
    p1, q1, _ = sample_nonorthogonal_keymat(d, lam, seed=11, dtype=torch.float64)
    p2, _, _ = sample_nonorthogonal_keymat(d, lam, seed=12, dtype=torch.float64)
    g = torch.Generator().manual_seed(13)
    x = torch.randn(8, d, dtype=torch.float64, generator=g)
    w1 = torch.randn(d, d, dtype=torch.float64, generator=g); b1 = torch.randn(d, dtype=torch.float64, generator=g)
    w2 = torch.randn(d, d, dtype=torch.float64, generator=g); b2 = torch.randn(d, dtype=torch.float64, generator=g)
    # obfuscated chain: P_in=p0, then p1 (=p0's next), then p2
    x_tilde = x @ p0
    w1t, b1t = fold_weight(w1, q0, p1), fold_bias(b1, p1)
    h_tilde = x_tilde @ w1t + b1t                     # = (x w1 + b1) p1
    w2t, b2t = fold_weight(w2, q1, p2), fold_bias(b2, p2)
    y_tilde = h_tilde @ w2t + b2t                     # = ((x w1+b1) w2 + b2) p2
    y_plain = (x @ w1 + b1) @ w2 + b2
    assert float((y_tilde - y_plain @ p2).abs().max()) <= 1e-9


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-9), (torch.float32, 2e-3)])
def test_precision_sweep(dtype, tol):
    d = 32
    for lam in LAMBDAS:
        p_in, q_in, _ = sample_nonorthogonal_keymat(d, lam, seed=5, dtype=dtype)
        p_out, _, _ = sample_nonorthogonal_keymat(d, lam, seed=6, dtype=dtype)
        g = torch.Generator().manual_seed(7)
        x = torch.randn(16, d, dtype=dtype, generator=g)
        w = torch.randn(d, d, dtype=dtype, generator=g)
        b = torch.randn(d, dtype=dtype, generator=g)
        r = two_sided_linear(x, w, b, p_in, q_in, p_out)
        err = float((r["y_tilde"] - r["y_ref"]).abs().max())
        assert err <= tol, f"dtype={dtype} lam={lam} err={err}"


def test_scope_negative_nonlinearity_not_exact():
    """A non-orthogonal P does NOT commute with an elementwise nonlinearity, so we
    must NOT claim end-to-end exactness through GELU/SiLU. This test PROVES the
    non-commutation (the reason exactness is scoped to the linear chain)."""
    d = 32
    p, q, _ = sample_nonorthogonal_keymat(d, 0.3, seed=8, dtype=torch.float64)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(16, d, dtype=torch.float64, generator=g)
    gelu = torch.nn.functional.gelu
    # if P commuted with GELU we'd have gelu(x P) q == gelu(x); it does not.
    lhs = gelu(x @ p) @ q
    rhs = gelu(x)
    assert float((lhs - rhs).abs().max()) > 1e-2   # decisively non-exact


def test_report_fields_scope_honest():
    _, _, diag = sample_nonorthogonal_keymat(16, 0.1, seed=0)
    f_untested = two_sided_keymat_report_fields(diag)
    assert f_untested["exact_lossless_claim"] is False        # not declared without tests
    assert f_untested["uses_noise"] is False
    assert f_untested["uses_two_sided_keymat"] is True
    assert f_untested["uses_nonorthogonal_transform"] is True
    assert f_untested["passes_rmsnorm_exactly"] is False
    assert f_untested["passes_elementwise_nonlinearity_exactly"] is False
    assert f_untested["production_qwen7b_integration"] is False
    assert f_untested["is_aloepri_replication"] is False
    f_tested = two_sided_keymat_report_fields(diag, exact_lossless_tests_passed=True)
    assert f_tested["exact_lossless_claim"] is True
