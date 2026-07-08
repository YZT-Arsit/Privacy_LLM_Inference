"""Tests for the Kronecker-lifted-linear variant (variant C).

Covers the identities requested for the study:
  A. Kronecker mixed-product     (X kron R)(W kron S) == (XW) kron (RS)
  B. Lifted linear correctness   X_hat W_hat + bias_hat == Pi ((XW+b) kron R) Omega
  C. Squeeze correctness          unlift(lift(H)) == H   (unit + generic R)
  D. Lifted MLP correctness       LiftedMLP(lift X) == lift(MLP(X))   (GELU + SwiGLU)
  E. Negative test                a non-selectable R breaks pure index-selection
  F. Report fields never leak the secret coordinate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pllo.ops.kronecker_lifted_linear import (  # noqa: E402
    kron_lift,
    kron_unlift,
    lifted_bias,
    lifted_linear_forward,
    lifted_linear_weight,
    lifted_mlp_report_fields,
    lifted_mlp_residual,
    lifted_mlp_squeeze,
    make_lift_params,
    sample_lift_r,
)
from pllo.ops.nonlinear_islands import (  # noqa: E402
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
# A. Kronecker mixed-product identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,d,p,k", [(2, 3, 4, 2), (3, 5, 2, 3), (4, 4, 6, 4)])
@pytest.mark.parametrize("seed", [0, 7, 21])
def test_kron_mixed_product(m, d, p, k, seed) -> None:
    g = _gen(seed)
    X = torch.randn(m, d, dtype=DT, generator=g)
    W = torch.randn(d, p, dtype=DT, generator=g)
    R = torch.randn(k, k, dtype=DT, generator=g)
    S = torch.randn(k, k, dtype=DT, generator=g)
    lhs = torch.kron(X, R) @ torch.kron(W, S)
    rhs = torch.kron(X @ W, R @ S)
    assert (lhs - rhs).abs().max().item() <= TOL


# ---------------------------------------------------------------------------
# B. Lifted linear correctness (no bias + bias), including Pi/Omega mixing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,d,p,k", [(2, 3, 4, 2), (3, 6, 3, 3), (1, 4, 5, 4)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dense_single_one", [True, False])
@pytest.mark.parametrize("token_safe", [True, False])
def test_lifted_linear_correctness(
    m, d, p, k, with_bias, dense_single_one, token_safe
) -> None:
    g = _gen(11)
    X = torch.randn(m, d, dtype=DT, generator=g)
    W = torch.randn(d, p, dtype=DT, generator=g)
    b = torch.randn(p, dtype=DT, generator=g) if with_bias else None

    in_params = make_lift_params(
        m, d, k, generator=g, dense_single_one=dense_single_one,
        token_safe_rows=token_safe,
    )
    # Output space shares the row-side pi (Pi_out = Pi_in) but has its own R/Omega.
    out_params = make_lift_params(
        m, p, k, pi=in_params.pi, generator=g,
        dense_single_one=dense_single_one, token_safe_rows=token_safe,
    )

    x_hat = kron_lift(X, in_params)
    y_hat = lifted_linear_forward(x_hat, W, b, in_params, out_params)

    y_plain = X @ W + (b if b is not None else 0.0)
    expected = kron_lift(y_plain, out_params)
    assert (y_hat - expected).abs().max().item() <= 1e-8
    # And the squeeze recovers the plain output exactly.
    assert (kron_unlift(y_hat, out_params) - y_plain).abs().max().item() <= 1e-8


def test_lifted_bias_shape_and_value() -> None:
    g = _gen(3)
    m, d, p, k = 3, 4, 5, 3
    b = torch.randn(p, dtype=DT, generator=g)
    in_params = make_lift_params(m, d, k, generator=g)
    out_params = make_lift_params(m, p, k, pi=in_params.pi, generator=g)
    bias_hat = lifted_bias(b, in_params, out_params)
    assert bias_hat.shape == (m * k, p * k)
    expected = kron_lift(torch.ones(m, 1, dtype=DT) @ b.reshape(1, -1), out_params)
    assert (bias_hat - expected).abs().max().item() <= TOL


# ---------------------------------------------------------------------------
# C. Squeeze correctness (unit-selectable and generic R)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("dense_single_one", [True, False])
def test_squeeze_roundtrip(k, dense_single_one) -> None:
    g = _gen(5)
    m, d = 4, 6
    H = torch.randn(m, d, dtype=DT, generator=g)
    params = make_lift_params(m, d, k, generator=g, dense_single_one=dense_single_one)
    assert (kron_unlift(kron_lift(H, params), params) - H).abs().max().item() <= 1e-9
    if dense_single_one:
        assert params.unit_selectable
        assert params.r[params.sel_row, params.sel_col].item() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# D. Lifted MLP correctness (GELU single-branch + SwiGLU), squeeze and residual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("swiglu", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("island", [lifted_mlp_squeeze, lifted_mlp_residual])
def test_lifted_mlp_correctness(k, swiglu, with_bias, island) -> None:
    g = _gen(17)
    m, d, f = 3, 5, 8
    X = torch.randn(m, d, dtype=DT, generator=g)
    w_up = torch.randn(d, f, dtype=DT, generator=g)
    w_down = torch.randn(f, d, dtype=DT, generator=g)
    b_up = torch.randn(f, dtype=DT, generator=g) if with_bias else None
    b_down = torch.randn(d, dtype=DT, generator=g) if with_bias else None
    w_gate = torch.randn(d, f, dtype=DT, generator=g) if swiglu else None
    b_gate = (torch.randn(f, dtype=DT, generator=g) if (swiglu and with_bias) else None)

    in_params = make_lift_params(m, d, k, generator=g)
    out_params = make_lift_params(m, d, k, pi=in_params.pi, generator=g)

    # Plain reference.
    up = X @ w_up + (b_up if b_up is not None else 0.0)
    if swiglu:
        gate = X @ w_gate + (b_gate if b_gate is not None else 0.0)
        act = silu_reference(gate) * up
    else:
        act = gelu_reference(up)
    y_plain = act @ w_down + (b_down if b_down is not None else 0.0)

    x_hat = kron_lift(X, in_params)
    res = island(
        x_hat, w_up, b_up, w_down, b_down, in_params, out_params,
        activation="gelu", w_gate=w_gate, b_gate=b_gate, generator=g,
    )
    y_hat = res["y_hat"]
    expected = kron_lift(y_plain, out_params)
    assert (y_hat - expected).abs().max().item() <= 1e-8
    assert (kron_unlift(y_hat, out_params) - y_plain).abs().max().item() <= 1e-8


# ---------------------------------------------------------------------------
# E. Negative test: a non-unit R breaks *pure* index-selection.
# ---------------------------------------------------------------------------


def test_negative_non_selectable_breaks_pure_selection() -> None:
    g = _gen(9)
    m, d, k = 3, 4, 3
    H = torch.randn(m, d, dtype=DT, generator=g)
    # generic (non single-one) R: unit_selectable is False, so a *pure* selection
    # (without the divide-by r[a,b]) does NOT recover H.
    params = make_lift_params(m, d, k, generator=g, dense_single_one=False)
    assert not params.unit_selectable
    core = params.pi_inv @ kron_lift(H, params) @ params.omega_inv
    rows = torch.arange(m) * k + params.sel_row
    cols = torch.arange(d) * k + params.sel_col
    pure_sel = core.index_select(0, rows).index_select(1, cols)   # == H * r[a,b]
    assert (pure_sel - H).abs().max().item() > 1e-3               # scaled, not equal
    # The correct unlift (divide by r[a,b]) does recover H.
    assert (kron_unlift(kron_lift(H, params), params) - H).abs().max().item() <= 1e-9


def test_squeeze_shape_guard() -> None:
    params = make_lift_params(2, 3, 2, generator=_gen(1))
    with pytest.raises(ValueError):
        kron_unlift(torch.zeros(3, 3, dtype=DT), params)


# ---------------------------------------------------------------------------
# F. Report fields never leak the secret coordinate.
# ---------------------------------------------------------------------------


def test_report_fields_no_leak() -> None:
    params = make_lift_params(3, 4, 3, generator=_gen(2))
    fields = lifted_mlp_report_fields(
        params, variant_stage="C2", lifted_residual_stream=True,
        max_abs_error=1e-12, relative_l2_error=1e-13,
    )
    assert fields["variant"] == "kronecker_lifted_linear"
    assert fields["uses_kronecker_lifted_state"] is True
    assert fields["formal_security_claim"] is False
    assert fields["production_qwen7b_integration"] is False
    assert fields["experiment_only"] is True
    assert fields["stable_state_form"] == "Pi (H (x) R) Omega"
    assert fields["hidden_dim_after"] == 4 * 3
    # No key exposes the selected coordinate or raw factors.
    blob = str(fields).lower()
    assert "sel_row" not in blob and "sel_col" not in blob
    assert str(params.sel_row) not in {v for v in fields.values() if isinstance(v, str)}
    assert fields["selected_coordinate_public"] is False
