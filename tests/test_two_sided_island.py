"""Tests for Variant E — two-sided non-orthogonal linear islands with
nonlinear-boundary mask switching.

  A. mask switching correctness   (X N_res)(N_res^{-1} P) == X P ; and back
  B. GPT-2 GELU MLP island        island(X N_res) == MLP(X) N_res  (fp64 exact;
                                   fp32 stable; ±bias; multiple lambda/seed)
  C. negative                     dense keymat across GELU without Amulet fails
  D. SwiGLU elementwise boundary  perm/diagonal commute with Hadamard; dense does not
  E. Qwen SwiGLU FFN island       island(X N_res) == FFN(X) N_res
  F. report fields honesty        hard invariants + exact_lossless only if tested
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pllo.ops.two_sided_island import (  # noqa: E402
    perm_matrix,
    signed_perm_matrix,
    switch_weight,
    two_sided_gelu_mlp_island,
    two_sided_island_report_fields,
    two_sided_swiglu_island,
)
from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat  # noqa: E402

LAMBDAS = (0.03, 0.1, 0.3, 1.0)


def _km(d, lam, s, dt):
    return sample_nonorthogonal_keymat(d, lam, seed=s, dtype=dt)


# --- A. mask switching -------------------------------------------------------
@pytest.mark.parametrize("lam", LAMBDAS)
def test_mask_switch_in_and_out(lam):
    d = 32
    n, n_inv = signed_perm_matrix(d, seed=0, dtype=torch.float64)
    p_in, _, _ = _km(d, lam, 1, torch.float64)
    p_out, _, _ = _km(d, lam, 2, torch.float64)
    p_out_inv = torch.linalg.inv(p_out)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, d, dtype=torch.float64, generator=g)
    # into island: (X N) (N^{-1} P_in) == X P_in
    assert float(((x @ n) @ switch_weight(n_inv, p_in) - x @ p_in).abs().max()) <= 1e-10
    # out of island: (Y P_out) (P_out^{-1} N) == Y N
    y = torch.randn(16, d, dtype=torch.float64, generator=g)
    assert float(((y @ p_out) @ switch_weight(p_out_inv, n) - y @ n).abs().max()) <= 1e-10


# --- B. GPT-2 GELU MLP island ------------------------------------------------
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-8), (torch.float32, 5e-2)])
@pytest.mark.parametrize("with_bias", [True, False])
def test_gpt2_gelu_island(dtype, tol, with_bias):
    dm, dff, m = 32, 64, 8
    n, n_inv = signed_perm_matrix(dm, seed=0, dtype=dtype)
    for lam in LAMBDAS:
        for seed in (1, 2):
            g = torch.Generator().manual_seed(seed)
            p_in, q_in, _ = _km(dm, lam, seed, dtype)
            p_ff, _, _ = _km(dff, lam, seed + 10, dtype)
            p_out, _, _ = _km(dm, lam, seed + 20, dtype)
            p_out_inv = torch.linalg.inv(p_out)
            x = torch.randn(m, dm, dtype=dtype, generator=g)
            w_up = torch.randn(dm, dff, dtype=dtype, generator=g)
            w_down = torch.randn(dff, dm, dtype=dtype, generator=g)
            b_up = torch.randn(dff, dtype=dtype, generator=g) if with_bias else None
            b_down = torch.randn(dm, dtype=dtype, generator=g) if with_bias else None
            r = two_sided_gelu_mlp_island(
                x @ n, w_up, b_up, w_down, b_down, n_res=n, n_res_inv=n_inv,
                p_in=p_in, q_in=q_in, p_ff=p_ff, p_out=p_out, p_out_inv=p_out_inv)
            assert r["max_abs_error"] <= tol, f"dtype={dtype} lam={lam} err={r['max_abs_error']}"


def test_gpt2_gelu_island_fp64_is_exact():
    """fp64 island is exact to rounding across every lambda (tight bound)."""
    dm, dff, m = 32, 64, 8
    n, n_inv = signed_perm_matrix(dm, seed=0, dtype=torch.float64)
    for lam in LAMBDAS:
        g = torch.Generator().manual_seed(0)
        p_in, q_in, _ = _km(dm, lam, 1, torch.float64)
        p_ff, _, _ = _km(dff, lam, 2, torch.float64)
        p_out, _, _ = _km(dm, lam, 3, torch.float64)
        r = two_sided_gelu_mlp_island(
            torch.randn(m, dm, dtype=torch.float64, generator=g) @ n,
            torch.randn(dm, dff, dtype=torch.float64, generator=g), None,
            torch.randn(dff, dm, dtype=torch.float64, generator=g), None,
            n_res=n, n_res_inv=n_inv, p_in=p_in, q_in=q_in, p_ff=p_ff,
            p_out=p_out, p_out_inv=torch.linalg.inv(p_out))
        assert r["max_abs_error"] <= 1e-9


# --- C. negative: dense keymat cannot cross GELU without Amulet --------------
def test_negative_dense_keymat_across_gelu_fails():
    d = 32
    p, q, _ = _km(d, 0.3, 5, torch.float64)
    x = torch.randn(16, d, dtype=torch.float64)
    lhs = torch.nn.functional.gelu(x @ p) @ q      # would need to equal gelu(x)
    assert float((lhs - torch.nn.functional.gelu(x)).abs().max()) > 1e-2


# --- D. SwiGLU elementwise boundary -----------------------------------------
def test_elementwise_boundary_permutation_and_diagonal():
    d = 48
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, d, dtype=torch.float64, generator=g)
    b = torch.randn(8, d, dtype=torch.float64, generator=g)
    s, _ = perm_matrix(d, seed=1, dtype=torch.float64)
    # permutation: (aS)⊙(bS) == (a⊙b)S  AND  SiLU(gS)==SiLU(g)S
    assert float(((a @ s) * (b @ s) - (a * b) @ s).abs().max()) <= 1e-12
    assert float((torch.nn.functional.silu(a @ s) - torch.nn.functional.silu(a) @ s).abs().max()) <= 1e-12
    # diagonal: (aD1)⊙(bD2) == (a⊙b)(D1D2)
    d1 = torch.diag(torch.randn(d, dtype=torch.float64, generator=g))
    d2 = torch.diag(torch.randn(d, dtype=torch.float64, generator=g))
    assert float(((a @ d1) * (b @ d2) - (a * b) @ (d1 @ d2)).abs().max()) <= 1e-12
    # dense P: (aP)⊙(bP) != (a⊙b)P
    p, _, _ = _km(d, 0.3, 2, torch.float64)
    assert float(((a @ p) * (b @ p) - (a * b) @ p).abs().max()) > 1e-2


# --- E. Qwen SwiGLU FFN island ----------------------------------------------
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-7), (torch.float32, 5e-2)])
def test_qwen_swiglu_island(dtype, tol):
    dm, dff, m = 32, 64, 8
    n, n_inv = signed_perm_matrix(dm, seed=0, dtype=dtype)
    for lam in LAMBDAS:
        g = torch.Generator().manual_seed(1)
        p_in, q_in, _ = _km(dm, lam, 1, dtype)
        p_g, _, _ = _km(dff, lam, 4, dtype)
        p_u, _, _ = _km(dff, lam, 5, dtype)
        p_mid, _, _ = _km(dff, lam, 6, dtype)
        p_out, _, _ = _km(dm, lam, 3, dtype)
        s, s_inv = perm_matrix(dff, seed=7, dtype=dtype)
        x = torch.randn(m, dm, dtype=dtype, generator=g)
        w_gate = torch.randn(dm, dff, dtype=dtype, generator=g)
        w_up = torch.randn(dm, dff, dtype=dtype, generator=g)
        w_down = torch.randn(dff, dm, dtype=dtype, generator=g)
        r = two_sided_swiglu_island(
            x @ n, w_gate, w_up, w_down, n_res=n, n_res_inv=n_inv, p_in=p_in, q_in=q_in,
            p_g=p_g, p_u=p_u, s_perm=s, s_perm_inv=s_inv, p_mid=p_mid, p_out=p_out,
            p_out_inv=torch.linalg.inv(p_out))
        assert r["max_abs_error"] <= tol, f"dtype={dtype} lam={lam} err={r['max_abs_error']}"


# --- F. report fields honesty ------------------------------------------------
def test_report_fields_hard_invariants():
    f = two_sided_island_report_fields()
    assert f["dense_keymat_crosses_rmsnorm"] is False
    assert f["dense_keymat_crosses_rope"] is False
    assert f["dense_keymat_crosses_hadamard"] is False
    assert f["uses_mask_switching_at_nonlinear_boundary"] is True
    assert f["uses_noise"] is False
    assert f["changes_stable_state_norm_leakage"] is False
    assert f["exact_lossless_claim"] is False
    assert f["production_qwen7b_integration"] is False
    assert two_sided_island_report_fields(exact_lossless_tests_passed=True)["exact_lossless_claim"] is True
