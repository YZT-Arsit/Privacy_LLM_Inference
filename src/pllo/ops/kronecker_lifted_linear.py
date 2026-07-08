"""Kronecker-*lifted* linear layers / residual stream (variant C prototype).

Motivation
----------

Variants A (``right_pad_pure_right``) and B (``right_pad_amulet_gelu``) both keep
the *stable* decoder state in the plain right-masked space::

    H_tilde = H N

and offload plain folded weights ``W_tilde = N_in^{-1} W N_out``.  The Amulet
right-mask GELU primitive (variant B) only lifts a *transient* nonlinear tensor
``Z``; after the squeeze the stable state is again ``H N``.  Consequently the
Gram / norm / weight-alignment / token-recovery attacks -- which observe ``H N``
and ``W_tilde`` -- are **unaffected** by the Amulet ``R_bar``.

Variant C (``kronecker_lifted_linear``) pushes the Kronecker lift *out* of the
nonlinear core and into the linear layers / residual stream, so the public attack
surface itself (stable state and/or folded linear weights) lives in a lifted
Kronecker space rather than the plain ``H N`` space.

    Plain stable state:   H_tilde = H N
    Lifted stable state:  H_hat   = Pi (H (x) R) Omega

Honest scope
------------

* This is an **experiment**, not the production Qwen7B path.
* ``formal_security_claim = False``.  Kronecker expansion does **not**
  information-theoretically remove norm/Gram structure -- ``||X (x) R|| =
  ||X|| ||R||`` and ``(X (x) R)(X (x) R)^T = (X X^T) (x) (R R^T)``.  Security
  relies on hidden block structure + permutation/mixing + the attacker's
  inability to align lifted channels.  If the block structure is recovered, the
  Gram/norm signal reappears.
* First stage lifts the **hidden/feature** dimension only.  The row (token/
  sequence) side is kept token-order-preserving (``Pi = I_m (x) P_k`` block form,
  or ``Pi = I``) so causal order and KV-cache append are not broken.  Attention /
  KV cache are **not** lifted in this stage.
* The lifted MLP island comes in two flavours:
    - ``lifted_mlp_squeeze`` (C1): lifts the up/gate/down linear weights but
      squeezes to the plain activation *around* the nonlinearity.  Protects the
      folded MLP weights only; the stable residual state between blocks is plain
      ``H N`` -- it does **not** defend a stable-residual attack.
    - ``lifted_mlp_residual`` (C2, experimental): keeps the residual stream in the
      lifted space across the whole MLP; only the elementwise nonlinearity
      internally squeezes to the plain activation *transiently* (unavoidable for
      an exact elementwise nonlinearity), then re-lifts.

Conventions
-----------

* Row-vector convention; tensors are ``[m, d]`` (m = token rows, d = features).
* ``torch.kron`` operands are made ``.contiguous()``.
* fp64 is the default for correctness checks (kron is exact up to ~1e-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from pllo.ops.nonlinear_islands import (
    gelu_reference,
    relu_reference,
    silu_reference,
)

__all__ = [
    "LiftParams",
    "LiftedLinear",
    "kron_lift",
    "kron_unlift",
    "lifted_bias",
    "lifted_linear_forward",
    "lifted_linear_weight",
    "lifted_mlp_report_fields",
    "lifted_mlp_residual",
    "lifted_mlp_squeeze",
    "make_lift_params",
    "sample_lift_r",
]

_ACTIVATIONS = {
    "relu": relu_reference,
    "gelu": gelu_reference,
    "silu": silu_reference,
}


# ---------------------------------------------------------------------------
# Generators / small matrix samplers (kept local; mirror amulet_right_mask)
# ---------------------------------------------------------------------------


def _gen(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _rand(shape, *, dtype, device, generator):
    if generator is not None:
        return torch.randn(*shape, generator=generator, dtype=dtype, device=device)
    return torch.randn(*shape, dtype=dtype, device=device)


def _permutation_matrix(
    s: int, *, dtype, device, generator: torch.Generator | None
) -> torch.Tensor:
    """Dense ``[s, s]`` permutation matrix (orthogonal; ``P^T P = I``)."""
    if generator is not None:
        perm = torch.randperm(s, generator=generator, device=device)
    else:
        perm = torch.randperm(s, device=device)
    p = torch.zeros(s, s, dtype=dtype, device=device)
    p[torch.arange(s, device=device), perm] = 1.0
    return p


def _sample_invertible(
    k: int, *, dtype, device, generator, min_abs_det: float, max_tries: int = 200
) -> torch.Tensor:
    for _ in range(max_tries):
        m = _rand((k, k), dtype=dtype, device=device, generator=generator)
        if abs(float(torch.linalg.det(m).item())) >= min_abs_det:
            return m
    raise RuntimeError(
        f"failed to sample an invertible {k}x{k} matrix with |det| >= {min_abs_det}"
    )


def sample_lift_r(
    k: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    dense_single_one: bool = True,
    min_abs_det: float = 1e-4,
    avoid_eps: float = 1e-3,
    max_tries: int = 1000,
) -> tuple[torch.Tensor, int, int, bool]:
    """Sample the lift factor ``R in R^{k x k}``.

    Returns ``(R, a, b, unit_selectable)``.

    * ``dense_single_one=True``: dense invertible ``R`` with exactly one entry
      ``R[a, b] == 1`` and all others far from 1 (Amulet-style; ``unit_selectable``
      is True so the squeeze is a pure index-selection).
    * ``dense_single_one=False``: a generic invertible ``R``; ``(a, b)`` is the
      largest-magnitude entry and the squeeze divides by ``R[a, b]``
      (``unit_selectable`` is False).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    device = torch.device(device)
    if not dense_single_one:
        r = _sample_invertible(
            k, dtype=dtype, device=device, generator=generator, min_abs_det=min_abs_det
        )
        idx = int(r.abs().argmax().item())
        a, b = idx // k, idx % k
        return r, a, b, False
    a = int(torch.randint(0, k, (1,), generator=generator, device=device).item()) if generator is not None else int(torch.randint(0, k, (1,), device=device).item())
    b = int(torch.randint(0, k, (1,), generator=generator, device=device).item()) if generator is not None else int(torch.randint(0, k, (1,), device=device).item())
    eye_idx = (a, b)
    for _ in range(max_tries):
        r = _rand((k, k), dtype=dtype, device=device, generator=generator)
        r[eye_idx] = 1.0
        mask = torch.ones(k, k, dtype=torch.bool, device=device)
        mask[eye_idx] = False
        if k > 1 and bool(((r[mask] - 1.0).abs() < avoid_eps).any()):
            continue
        if abs(float(torch.linalg.det(r).item())) < min_abs_det:
            continue
        return r, a, b, True
    raise RuntimeError(f"failed to sample a dense single-one R (k={k})")


# ---------------------------------------------------------------------------
# Lift parameters for a single tensor width
# ---------------------------------------------------------------------------


@dataclass
class LiftParams:
    """Parameters describing the lifted space for a tensor of width ``d``.

    ``H_hat = pi (H (x) r) omega`` for ``H`` of shape ``[m, d]``, giving an
    ``[m k, d k]`` lifted tensor.  ``pi`` is the shared row-side operator
    (token-order-preserving), ``omega`` the feature-side operator.
    """

    r: torch.Tensor            # [k, k] invertible lift factor
    pi: torch.Tensor           # [m k, m k] row-side operator (shared across widths)
    omega: torch.Tensor        # [d k, d k] feature-side operator
    r_inv: torch.Tensor
    pi_inv: torch.Tensor
    omega_inv: torch.Tensor
    m: int
    d: int
    k: int
    sel_row: int               # (a, b): the coordinate used by the squeeze
    sel_col: int
    unit_selectable: bool      # True iff r[a, b] == 1 (pure index-selection)


def make_lift_params(
    m: int,
    d: int,
    k: int,
    *,
    r: torch.Tensor | None = None,
    sel_row: int | None = None,
    sel_col: int | None = None,
    unit_selectable: bool | None = None,
    pi: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    seed: int | None = None,
    dense_single_one: bool = True,
    token_safe_rows: bool = True,
) -> LiftParams:
    """Build lift params for a ``[m, d]`` tensor with lift factor ``k``.

    ``pi`` (row side) is shared across the widths of a linear chain -- pass the
    same ``pi`` (and ``r`` for a matching space) to keep ``Pi_out = Pi_in``.  When
    ``token_safe_rows`` the row operator is ``I_m (x) P_k`` (block-diagonal per
    token: permutes sub-rows, never mixes token order -> KV/causal-safe); set it
    False for a full row permutation (toy only).
    """
    device = torch.device(device)
    if generator is None and seed is not None:
        generator = _gen(seed, device)
    if r is None:
        r, a, b, unit = sample_lift_r(
            k, dtype=dtype, device=device, generator=generator,
            dense_single_one=dense_single_one,
        )
        sel_row, sel_col, unit_selectable = a, b, unit
    else:
        if sel_row is None or sel_col is None:
            idx = int(r.abs().argmax().item())
            sel_row, sel_col = idx // k, idx % k
        if unit_selectable is None:
            unit_selectable = bool(
                torch.isclose(
                    r[sel_row, sel_col],
                    torch.ones((), dtype=r.dtype, device=r.device),
                ).item()
            )
    if pi is None:
        if token_safe_rows:
            p_k = _permutation_matrix(k, dtype=dtype, device=device, generator=generator)
            pi = torch.kron(
                torch.eye(m, dtype=dtype, device=device).contiguous(), p_k.contiguous()
            )
        else:
            pi = _permutation_matrix(m * k, dtype=dtype, device=device, generator=generator)
    omega = _permutation_matrix(d * k, dtype=dtype, device=device, generator=generator)
    # pi / omega are permutation matrices (orthogonal): inverse == transpose. Use
    # ``.T`` directly -- exact and far cheaper than a general inverse on the large
    # (d k)-dim omega (torch.linalg.inv would be O((d k)^3)). Passing a custom
    # non-orthogonal pi is a stage-2 extension; assert orthogonality if given.
    if pi.shape[0] <= 512:  # cheap guard; skip for large custom pi
        _eye = torch.eye(pi.shape[0], dtype=dtype, device=device)
        assert torch.allclose(pi @ pi.T, _eye, atol=1e-6), "pi must be orthogonal (stage 1)"
    return LiftParams(
        r=r, pi=pi, omega=omega,
        r_inv=torch.linalg.inv(r),
        pi_inv=pi.T.contiguous(),
        omega_inv=omega.T.contiguous(),
        m=int(m), d=int(d), k=int(k),
        sel_row=int(sel_row), sel_col=int(sel_col),
        unit_selectable=bool(unit_selectable),
    )


# ---------------------------------------------------------------------------
# Lift / unlift (squeeze)
# ---------------------------------------------------------------------------


def kron_lift(h: torch.Tensor, params: LiftParams) -> torch.Tensor:
    """``H -> H_hat = pi (H (x) r) omega`` (``[m, d] -> [m k, d k]``)."""
    if h.shape != (params.m, params.d):
        raise ValueError(f"h shape {tuple(h.shape)} != ({params.m}, {params.d})")
    return params.pi @ torch.kron(h.contiguous(), params.r.contiguous()) @ params.omega


def kron_unlift(h_hat: torch.Tensor, params: LiftParams) -> torch.Tensor:
    """``H_hat -> H`` (squeeze).

    Undoes ``pi`` / ``omega``, then selects sub-row ``a`` / sub-col ``b`` of every
    ``k x k`` Kronecker block and (if ``r[a, b] != 1``) divides by ``r[a, b]``.
    """
    m, d, k = params.m, params.d, params.k
    a, b = params.sel_row, params.sel_col
    if h_hat.shape != (m * k, d * k):
        raise ValueError(f"h_hat shape {tuple(h_hat.shape)} != ({m * k}, {d * k})")
    core = params.pi_inv @ h_hat @ params.omega_inv     # == H (x) r
    rows = torch.arange(m, device=h_hat.device) * k + a
    cols = torch.arange(d, device=h_hat.device) * k + b
    sel = core.index_select(0, rows).index_select(1, cols)
    scale = params.r[a, b]
    if params.unit_selectable:
        return sel
    return sel / scale


# ---------------------------------------------------------------------------
# Lifted linear layer
# ---------------------------------------------------------------------------


def lifted_linear_weight(
    w: torch.Tensor, in_params: LiftParams, out_params: LiftParams
) -> torch.Tensor:
    """``W -> W_hat = omega_in^{-1} (W (x) S) omega_out``, ``S = r_in^{-1} r_out``.

    ``w`` maps width ``in_params.d`` -> ``out_params.d`` (shape ``[d_in, d_out]``).
    Requires the same row-side ``pi`` on both spaces (``Pi_out = Pi_in``).
    """
    if w.shape != (in_params.d, out_params.d):
        raise ValueError(
            f"w shape {tuple(w.shape)} != ({in_params.d}, {out_params.d})"
        )
    s = in_params.r_inv @ out_params.r                  # R_in S = R_out
    return (
        in_params.omega_inv
        @ torch.kron(w.contiguous(), s.contiguous())
        @ out_params.omega
    )


def lifted_bias(
    b: torch.Tensor, in_params: LiftParams, out_params: LiftParams
) -> torch.Tensor:
    """``bias_hat = pi_out ((1 b^T) (x) r_out) omega_out`` (``[m k, d_out k]``).

    Broadcasts the length-``d_out`` bias across the ``m`` rows before lifting.
    """
    m = out_params.m
    ones = torch.ones(m, 1, dtype=b.dtype, device=b.device)
    plain = ones @ b.reshape(1, -1)                     # [m, d_out]
    return (
        out_params.pi
        @ torch.kron(plain.contiguous(), out_params.r.contiguous())
        @ out_params.omega
    )


def lifted_linear_forward(
    x_hat: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    in_params: LiftParams,
    out_params: LiftParams,
) -> torch.Tensor:
    """Lifted linear: ``X_hat -> Y_hat = X_hat W_hat + bias_hat``.

    Equals ``Pi (( X W + b ) (x) R_out) Omega_out`` when ``x_hat`` is a valid lift
    of ``X`` (i.e. ``Pi_out = Pi_in`` and the input factor is ``r_in``).
    """
    y_hat = x_hat @ lifted_linear_weight(w, in_params, out_params)
    if b is not None:
        y_hat = y_hat + lifted_bias(b, in_params, out_params)
    return y_hat


@dataclass
class LiftedLinear:
    """Convenience wrapper bundling a lifted weight + optional lifted bias."""

    w_hat: torch.Tensor
    bias_hat: torch.Tensor | None
    in_params: LiftParams
    out_params: LiftParams

    @classmethod
    def from_plain(
        cls,
        w: torch.Tensor,
        b: torch.Tensor | None,
        in_params: LiftParams,
        out_params: LiftParams,
    ) -> "LiftedLinear":
        return cls(
            w_hat=lifted_linear_weight(w, in_params, out_params),
            bias_hat=(lifted_bias(b, in_params, out_params) if b is not None else None),
            in_params=in_params,
            out_params=out_params,
        )

    def __call__(self, x_hat: torch.Tensor) -> torch.Tensor:
        y_hat = x_hat @ self.w_hat
        if self.bias_hat is not None:
            y_hat = y_hat + self.bias_hat
        return y_hat


# ---------------------------------------------------------------------------
# Lifted MLP islands (C1 squeeze / C2 residual)
# ---------------------------------------------------------------------------


def _plain_mlp(
    x, w_up, b_up, w_down, b_down, activation, w_gate=None, b_gate=None,
):
    """Plain reference MLP. SwiGLU when ``w_gate`` given, else single-branch."""
    up = x @ w_up + (b_up if b_up is not None else 0.0)
    if w_gate is not None:
        gate = x @ w_gate + (b_gate if b_gate is not None else 0.0)
        act = silu_reference(gate) * up
    else:
        act = _ACTIVATIONS[activation](up)
    y = act @ w_down + (b_down if b_down is not None else 0.0)
    return y, act


def _mid_params_for(
    h_params: LiftParams, f: int, *, generator, dense_single_one: bool,
) -> LiftParams:
    """Build the hidden-width (``f``) lift params sharing the row side ``pi``."""
    dtype, device = h_params.r.dtype, h_params.r.device
    r_mid, a, b, unit = sample_lift_r(
        h_params.k, dtype=dtype, device=device, generator=generator,
        dense_single_one=dense_single_one,
    )
    return make_lift_params(
        h_params.m, f, h_params.k, r=r_mid, sel_row=a, sel_col=b,
        unit_selectable=unit, pi=h_params.pi, dtype=dtype, device=device,
        generator=generator,
    )


def lifted_mlp_squeeze(
    x_hat: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    in_params: LiftParams,
    out_params: LiftParams,
    *,
    activation: Literal["relu", "gelu", "silu"] = "gelu",
    w_gate: torch.Tensor | None = None,
    b_gate: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    dense_single_one: bool = True,
) -> dict[str, Any]:
    """C1: lifted up/gate/down linear weights, **squeeze around the nonlinearity**.

    The folded MLP linear weights are lifted, but the activation is computed on the
    plain squeezed value, and the block output is ``Y_hat`` in ``out_params`` space.
    This protects the *folded MLP weights* only -- the surrounding stable residual
    stream (input/output of this call) is a valid lift, but the nonlinear point is
    plain.  Does **not** defend a stable-residual attack on the plain activation.
    """
    f = w_up.shape[1]
    mid = _mid_params_for(in_params, f, generator=generator,
                          dense_single_one=dense_single_one)

    up_hat = lifted_linear_forward(x_hat, w_up, b_up, in_params, mid)
    u = kron_unlift(up_hat, mid)                        # plain U
    if w_gate is not None:
        gate_hat = lifted_linear_forward(x_hat, w_gate, b_gate, in_params, mid)
        g = kron_unlift(gate_hat, mid)
        act = silu_reference(g) * u
    else:
        act = _ACTIVATIONS[activation](u)
    act_hat = kron_lift(act, mid)                       # re-lift the plain activation
    y_hat = lifted_linear_forward(act_hat, w_down, b_down, mid, out_params)
    return {
        "y_hat": y_hat,
        "mid_params": mid,
        "metadata": {
            "island": "lifted_mlp_squeeze",
            "activation": "swiglu" if w_gate is not None else activation,
            "lift_factor_k": in_params.k,
            "lifted_linear_weights": True,
            "lifted_residual_stream": False,
            "nonlinear_point_is_plain": True,
            "protects": "folded_mlp_weights_only",
        },
    }


def lifted_mlp_residual(
    x_hat: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    in_params: LiftParams,
    out_params: LiftParams,
    *,
    activation: Literal["relu", "gelu", "silu"] = "gelu",
    w_gate: torch.Tensor | None = None,
    b_gate: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    dense_single_one: bool = True,
) -> dict[str, Any]:
    """C2 (experimental): keep the residual stream lifted across the whole MLP.

    Both the input ``x_hat`` and output ``y_hat`` are lifted, and the up/gate/down
    weights are lifted.  Only the elementwise nonlinearity internally squeezes to
    the plain activation *transiently* (unavoidable for an exact elementwise
    nonlinearity) and immediately re-lifts.  Numerically identical to
    :func:`lifted_mlp_squeeze`; the difference is the *contract*: here the stable
    residual state (this call's input and output) never leaves the lifted space,
    so a stable-residual attacker sees ``Pi (H (x) R) Omega`` rather than ``H N``.

    NOTE: exact, but ``formal_security_claim = False`` -- the transient plain
    activation still exists inside the island, exactly as in any nonlinear island.
    """
    out = lifted_mlp_squeeze(
        x_hat, w_up, b_up, w_down, b_down, in_params, out_params,
        activation=activation, w_gate=w_gate, b_gate=b_gate,
        generator=generator, dense_single_one=dense_single_one,
    )
    out["metadata"].update({
        "island": "lifted_mlp_residual",
        "lifted_residual_stream": True,
        "nonlinear_point_is_plain": True,      # transient only, inside the island
        "protects": "folded_mlp_weights_and_stable_residual_stream",
        "attention_kv_lifted": False,
        "stage": "C2_experimental",
    })
    return out


# ---------------------------------------------------------------------------
# Report fields
# ---------------------------------------------------------------------------


def lifted_mlp_report_fields(
    params: LiftParams,
    *,
    variant_stage: str = "C1",
    lifted_residual_stream: bool = False,
    max_abs_error: float | None = None,
    relative_l2_error: float | None = None,
    approximate: bool = False,
) -> dict[str, Any]:
    """Audit fields for the Kronecker-lifted-linear variant.

    Never publishes the selected coordinate ``(a, b)`` or raw ``R`` / ``Pi`` /
    ``Omega``.
    """
    k = params.k
    fields: dict[str, Any] = {
        "variant": "kronecker_lifted_linear",
        "variant_stage": variant_stage,
        "uses_right_mask": "partial",
        "uses_kronecker_lifted_state": True,
        "lift_factor": int(k),
        "stable_state_form": "Pi (H (x) R) Omega",
        "lifted_linear_weights": True,
        "lifted_residual_stream": bool(lifted_residual_stream),
        "attention_kv_lifted": False,
        "hidden_dim_before": int(params.d),
        "hidden_dim_after": int(params.d * k),
        "row_side_token_order_preserving": True,
        "selected_coordinate_public": False,
        "raw_lift_factor_visible_to_gpu": False,
        "unit_selectable_squeeze": bool(params.unit_selectable),
        "approximate_nonlinear": bool(approximate),
        "formal_security_claim": False,
        "production_qwen7b_integration": False,
        "experiment_only": True,
        "paper_scope": "kronecker_lifted_linear_correctness_experiment",
        "security_note": (
            "Kronecker expansion does NOT information-theoretically remove "
            "norm/Gram: ||X kron R|| = ||X|| ||R|| and (X kron R)(X kron R)^T = "
            "(X X^T) kron (R R^T). Security relies on hidden block structure + "
            "permutation/mixing; if block structure is recovered, Gram/norm "
            "reappear."
        ),
    }
    if max_abs_error is not None:
        fields["max_abs_error"] = float(max_abs_error)
    if relative_l2_error is not None:
        fields["relative_l2_error"] = float(relative_l2_error)
    return fields
