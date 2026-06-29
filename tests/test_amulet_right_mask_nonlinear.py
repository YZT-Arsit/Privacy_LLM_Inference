"""Tests for the Amulet-style right-mask nonlinear island experiment.

External island contract: ``U N -> phi(U) N`` (right-mask only), for ReLU /
GELU / SiLU and the two-input SwiGLU operator, plus a Qwen-style MLP integration.
"""

from __future__ import annotations

import pytest
import torch

from pllo.ops.amulet_right_mask_islands import (
    amulet_right_mask_activation,
    amulet_right_mask_island_report_fields,
    amulet_right_mask_swiglu,
    make_right_mask_amulet_params,
    run_amulet_right_mask_qwen_mlp,
    sample_amulet_r_factors,
    sample_dense_single_one_rbar,
    selection_e1,
    selection_e2,
    squeeze_select,
)
from pllo.ops.nonlinear_islands import (
    gelu_reference,
    relu_reference,
    silu_reference,
)

DT = torch.float64
TOL = 1e-9


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Test 1: R_bar construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_rbar_dense_single_one_and_factorisation(k: int) -> None:
    g = _gen(100 + k)
    rbar, a, b = sample_dense_single_one_rbar(k, dtype=DT, device="cpu", generator=g)
    # Exactly one entry equal to 1.
    assert rbar[a, b].item() == pytest.approx(1.0, abs=1e-12)
    near_one = (rbar - 1.0).abs() < 1e-3
    assert int(near_one.sum().item()) == 1, "exactly one entry may equal 1"
    # Dense (not a sparse one-hot): the off-(a,b) entries are generally nonzero.
    mask = torch.ones(k, k, dtype=torch.bool)
    mask[a, b] = False
    if k > 1:
        assert bool((rbar[mask].abs() > 1e-6).any()), "R_bar must be dense, not one-hot"
    # Invertible / well-conditioned.
    assert abs(float(torch.linalg.det(rbar).item())) >= 1e-4

    rf = sample_amulet_r_factors(k, dtype=DT, device="cpu", generator=g)
    assert rf.rbar[rf.selected_row, rf.selected_col].item() == pytest.approx(1.0, abs=1e-12)
    prod_err = (rf.r1 @ rf.r2 @ rf.r3 - rf.rbar).abs().max().item()
    assert prod_err <= TOL, f"R1 R2 R3 != R_bar (err {prod_err})"


# ---------------------------------------------------------------------------
# Test 2: selection matrices E1, E2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("act_name,act", [
    ("relu", relu_reference), ("gelu", gelu_reference), ("silu", silu_reference),
])
def test_selection_matrices(act_name: str, act) -> None:
    g = _gen(7)
    m, d, k = 3, 5, 4
    U = torch.randn(m, d, dtype=DT, generator=g)
    rf = sample_amulet_r_factors(k, dtype=DT, device="cpu", generator=g)
    a, b = rf.selected_row, rf.selected_col
    e1 = selection_e1(m, k, a, dtype=DT, device="cpu")
    e2 = selection_e2(d, k, b, dtype=DT, device="cpu")
    lift = torch.kron(U.contiguous(), rf.rbar.contiguous())

    assert (e1 @ lift @ e2 - U).abs().max().item() <= TOL
    assert (e1 @ act(lift) @ e2 - act(U)).abs().max().item() <= TOL
    # Index-based squeeze agrees with the explicit matrices.
    assert (squeeze_select(act(lift), m, d, k, a, b) - e1 @ act(lift) @ e2).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Test 3: right-mask activation correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("act_name,act", [
    ("relu", relu_reference), ("gelu", gelu_reference), ("silu", silu_reference),
])
@pytest.mark.parametrize("shape", [(2, 8), (4, 6), (8, 16)])
@pytest.mark.parametrize("k", [2, 3])
def test_right_mask_activation(act_name, act, shape, k) -> None:
    g = _gen(11)
    m, d = shape
    N = torch.randn(d, d, dtype=DT, generator=g)
    while abs(float(torch.linalg.det(N).item())) < 1e-3:
        N = torch.randn(d, d, dtype=DT, generator=g)
    U = torch.randn(m, d, dtype=DT, generator=g)
    params = make_right_mask_amulet_params(m, d, k, N, generator=g)
    out = amulet_right_mask_activation(U @ N, params, act_name)
    expected = act(U) @ N
    assert (out - expected).abs().max().item() <= 1e-8


# ---------------------------------------------------------------------------
# Test 4: SwiGLU correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 8), (4, 6)])
@pytest.mark.parametrize("k", [2, 3])
def test_right_mask_swiglu(shape, k) -> None:
    g = _gen(13)
    m, d = shape
    N = torch.randn(d, d, dtype=DT, generator=g)
    while abs(float(torch.linalg.det(N).item())) < 1e-3:
        N = torch.randn(d, d, dtype=DT, generator=g)
    G = torch.randn(m, d, dtype=DT, generator=g)
    U = torch.randn(m, d, dtype=DT, generator=g)
    params = make_right_mask_amulet_params(m, d, k, N, generator=g)
    out = amulet_right_mask_swiglu(G @ N, U @ N, params)
    expected = (silu_reference(G) * U) @ N
    assert (out - expected).abs().max().item() <= 1e-8


# ---------------------------------------------------------------------------
# Test 5: Qwen-style MLP correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_pad", [False, True])
def test_qwen_style_mlp(with_pad: bool) -> None:
    g = _gen(17)
    m, d, f, k = 4, 8, 16, 3

    def _inv(dim):
        M = torch.randn(dim, dim, dtype=DT, generator=g)
        while abs(float(torch.linalg.det(M).item())) < 1e-3:
            M = torch.randn(dim, dim, dtype=DT, generator=g)
        return M

    wg = torch.randn(d, f, dtype=DT, generator=g)
    wu = torch.randn(d, f, dtype=DT, generator=g)
    wd = torch.randn(f, d, dtype=DT, generator=g)
    bg = torch.randn(f, dtype=DT, generator=g)
    bu = torch.randn(f, dtype=DT, generator=g)
    bd = torch.randn(d, dtype=DT, generator=g)
    X = torch.randn(m, d, dtype=DT, generator=g)
    n_in = _inv(d)
    n_in_inv = torch.linalg.inv(n_in)
    n_ff = _inv(f)
    n_out = _inv(d)
    pad = torch.randn(m, d, dtype=DT, generator=g) if with_pad else None

    r = run_amulet_right_mask_qwen_mlp(
        X, wg, bg, wu, bu, wd, bd, n_in, n_in_inv, n_ff, n_out,
        k=k, generator=g, pad_in=pad,
    )
    assert (r["y_tilde"] - r["expected_y_tilde"]).abs().max().item() <= 1e-7
    assert (r["y_recovered"] - r["y_plain"]).abs().max().item() <= 1e-7
    # The island always sees the clean masked activations G N_ff / U N_ff,
    # even when an additive Linear-boundary pad is used.
    assert (r["g_tilde"] - r["expected_g_tilde"]).abs().max().item() <= 1e-8
    assert (r["u_tilde"] - r["expected_u_tilde"]).abs().max().item() <= 1e-8
    assert r["metadata"]["pad_enters_nonlinear_island"] is False


# ---------------------------------------------------------------------------
# Test 6: no forbidden behaviour (audit fields)
# ---------------------------------------------------------------------------


def test_audit_fields_no_forbidden_behaviour() -> None:
    g = _gen(23)
    rf = sample_amulet_r_factors(4, dtype=DT, device="cpu", generator=g)
    report = amulet_right_mask_island_report_fields(
        rf, max_abs_error=1e-13, relative_l2_error=1e-14, used_pad=True,
    )
    assert report["uses_left_sequence_mask"] is False
    assert report["intermediate_tee_boundary_calls"] == 0
    assert report["pad_enters_nonlinear_island"] is False
    assert report["selected_coordinate_public"] is False
    assert report["raw_rbar_visible_to_gpu"] is False
    assert report["raw_n_visible_to_gpu"] is False
    assert report["rbar_has_unique_one"] is True
    assert report["rbar_other_entries_not_one"] is True
    assert report["rbar_dense_single_one"] is True
    assert report["r_factor_product_verified"] is True
    assert report["right_mask_output_verified"] is True
    assert report["swiglu_verified"] is True
    # The secret coordinate must never appear in the public report (no key
    # name nor a (row, col) pair is exposed).
    flat = repr(report)
    assert "selected_row" not in flat and "selected_col" not in flat
    assert [rf.selected_row, rf.selected_col] not in list(report.values())
    assert (rf.selected_row, rf.selected_col) not in list(report.values())
    assert not any(k in report for k in ("selected_row", "selected_col", "a", "b"))


# ---------------------------------------------------------------------------
# Test 10 (guard): correctness genuinely relies on the lift/shuffle/squeeze.
# ---------------------------------------------------------------------------


def test_guard_no_plain_activation_fallback() -> None:
    """The activation must run on the lifted/shuffled operand, not recovered U.

    1. The activation input ``Z`` has the expanded shape ``[m k, d k]`` (not the
       plaintext ``[m, d]``), so a plain ``phi(U)`` fallback is structurally
       impossible.
    2. If the secret unit-copy is corrupted (``R_bar[a, b] != 1``), the squeeze
       no longer selects the true value and the output stops matching ``phi(U) N``
       -- proving the result is produced *by* the construction.
    """
    g = _gen(29)
    m, d, k = 4, 6, 3
    N = torch.randn(d, d, dtype=DT, generator=g)
    U = torch.randn(m, d, dtype=DT, generator=g)
    params = make_right_mask_amulet_params(m, d, k, N, generator=g)

    # (1) lifted operand shape.
    z = (
        params.pi3 @ torch.kron(params.pi1.contiguous(), params.r_factors.r1.contiguous())
    ) @ torch.kron((U @ N).contiguous(), params.r_factors.r2.contiguous()) @ (
        torch.kron((params.n_inv @ params.pi2).contiguous(), params.r_factors.r3.contiguous())
        @ params.pi4
    )
    assert z.shape == (m * k, d * k)
    assert z.shape != U.shape

    # baseline correctness
    out_ok = amulet_right_mask_activation(U @ N, params, "gelu")
    assert (out_ok - gelu_reference(U) @ N).abs().max().item() <= 1e-8

    # (2) corrupt the unit coordinate -> output must break.
    rf = params.r_factors
    rf.rbar[rf.selected_row, rf.selected_col] = 1.5  # no longer the unit copy
    rf.r3 = torch.linalg.solve(rf.r1 @ rf.r2, rf.rbar)  # keep R1R2R3 == corrupted rbar
    out_bad = amulet_right_mask_activation(U @ N, params, "gelu")
    assert (out_bad - gelu_reference(U) @ N).abs().max().item() > 1e-2


# ---------------------------------------------------------------------------
# Extra: secret coordinate stability across the factorisation
# ---------------------------------------------------------------------------


def test_rbar_factor_product_fails_loudly_on_bad_tol() -> None:
    g = _gen(31)
    with pytest.raises(RuntimeError):
        sample_amulet_r_factors(
            3, dtype=DT, device="cpu", generator=g, product_tol=-1.0,
        )
