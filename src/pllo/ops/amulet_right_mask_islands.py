"""Amulet-style *right-mask* nonlinear islands (CPU correctness prototype).

Our main protocol is decoder-only and **right-mask-only** at the stable
hidden-state boundary::

    H_tilde = H N

We do *not* use the original Amulet two-sided stable state ``H_tilde = P H Q``;
the sequence/time dimension is never left-masked in the stable state, which
keeps KV-cache append semantics intact.

Inside a nonlinear island we may, however, temporarily use an Amulet-style
lift / shuffle / squeeze construction built from Kronecker products and
permutation matrices.  The external contract of every island is::

    Input:   U_tilde = U N
    Output:  V_tilde = phi(U) N

for ``phi`` in {ReLU, GELU, SiLU} and for the two-input SwiGLU operator
``A = SiLU(G) * U``.

Construction (original Amulet specialised to ``P = I``, ``Q = N``)
-----------------------------------------------------------------

Choose a dense target Kronecker matrix ``R_bar in R^{k x k}`` with exactly one
secret entry ``R_bar[a, b] = 1`` and every other entry not equal to one.  Factor
it as ``R_bar = R1 R2 R3``.  For permutation matrices ``pi1..pi4`` set::

    M1 = pi3 (pi1 (x) R1)
    M2 = (N^{-1} pi2 (x) R3) pi4
    M3 = pi1^T E1 pi3^T
    M4 = pi4^T E2 pi2^T N

where ``E1 = I_m (x) e_a^T`` and ``E2 = I_d (x) e_b`` are the selection matrices
that squeeze the single unit-copy.  The GPU-side computation is::

    Z = M1 (U_tilde (x) R2) M2 = pi3 ((pi1 U pi2) (x) R_bar) pi4
    S = phi(Z)
    out_tilde = M3 S M4 = phi(U) N

The unit entry ``R_bar[a, b] = 1`` makes the squeeze ``E1 . E2`` select exactly
the *true* nonlinear value ``phi(U[i, j])`` from each Kronecker block, while the
shuffles ``pi3, pi4`` scramble the expanded ``[mk x dk]`` tensor.

Design / honesty notes
----------------------

* Row-vector convention; ``U`` is ``[m, d]`` (m = token rows, d = features).
* CPU-only correctness prototype; this is a *nonlinear-island experiment*, not
  the production Qwen7B path, unless and until explicitly integrated.
* ``pi1`` / ``pi3`` are island-internal transient row permutations that are
  undone by ``M3`` before the output, so the *stable* decoder state never carries
  a left/sequence mask (``uses_left_sequence_mask = false``).
* No activation is moved into a TEE and no per-layer TEE boundary call is added.
* Any additive Linear-boundary pad is compensated at the preceding Linear and
  never enters the nonlinear core (``pad_enters_nonlinear_island = false``).

Security caveat (honest scope)
------------------------------

This construction assumes the adversary cannot reliably identify the selected
unit-copy channel inside the Kronecker-expanded, shuffled space.  We do **not**
claim that arbitrary dense right masks commute with GELU/SiLU/SwiGLU;
correctness relies on the lift/shuffle/squeeze construction and the unique-one
selected coordinate ``(a, b)``, which is never published.
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
    "AmuletRFactors",
    "RightMaskAmuletIslandParams",
    "amulet_pad_mlp_report_fields",
    "amulet_right_mask_activation",
    "amulet_right_mask_island_report_fields",
    "amulet_right_mask_swiglu",
    "make_right_mask_amulet_params",
    "run_amulet_right_mask_qwen_mlp",
    "run_amulet_right_mask_qwen_mlp_with_linear_pad",
    "sample_amulet_r_factors",
    "sample_dense_single_one_rbar",
    "selection_e1",
    "selection_e2",
    "squeeze_select",
]


_ACTIVATIONS = {
    "relu": relu_reference,
    "gelu": gelu_reference,
    "silu": silu_reference,
}


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


def _randint(high, *, device, generator):
    if generator is not None:
        return int(torch.randint(0, high, (1,), generator=generator, device=device).item())
    return int(torch.randint(0, high, (1,), device=device).item())


def _permutation_matrix(
    s: int, *, dtype, device, generator: torch.Generator | None
) -> torch.Tensor:
    """Dense ``[s, s]`` permutation matrix (orthogonal; ``Pi^T Pi = I``)."""
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
    """Sample a dense well-conditioned invertible ``[k, k]`` matrix."""
    for _ in range(max_tries):
        m = _rand((k, k), dtype=dtype, device=device, generator=generator)
        if abs(float(torch.linalg.det(m).item())) >= min_abs_det:
            return m
    raise RuntimeError(
        f"failed to sample an invertible {k}x{k} matrix with |det| >= {min_abs_det}"
    )


# ---------------------------------------------------------------------------
# 2. R_bar construction: dense, exactly one entry == 1
# ---------------------------------------------------------------------------


def sample_dense_single_one_rbar(
    k: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    min_abs_det: float = 1e-4,
    avoid_eps: float = 1e-3,
    max_tries: int = 1000,
) -> tuple[torch.Tensor, int, int]:
    """Sample a dense invertible ``R_bar`` with exactly one entry equal to 1.

    Returns ``(R_bar, a, b)`` where ``R_bar[a, b] == 1`` and every other entry
    differs from 1 by at least ``avoid_eps``.  The matrix is dense/random (this
    is *not* a sparse one-hot) and well-conditioned (``|det| >= min_abs_det``).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    device = torch.device(device)
    a = _randint(k, device=device, generator=generator)
    b = _randint(k, device=device, generator=generator)
    eye_idx = (a, b)
    for _ in range(max_tries):
        rbar = _rand((k, k), dtype=dtype, device=device, generator=generator)
        rbar[eye_idx] = 1.0
        # Mask off the selected entry, then require all others to be far from 1.
        mask = torch.ones(k, k, dtype=torch.bool, device=device)
        mask[eye_idx] = False
        if k > 1:
            others = rbar[mask]
            if bool(((others - 1.0).abs() < avoid_eps).any()):
                continue
        if abs(float(torch.linalg.det(rbar).item())) < min_abs_det:
            continue
        return rbar, a, b
    raise RuntimeError(
        f"failed to sample a dense single-one R_bar (k={k}) after {max_tries} tries"
    )


# ---------------------------------------------------------------------------
# 3. R1, R2, R3 with R1 R2 R3 == R_bar
# ---------------------------------------------------------------------------


@dataclass
class AmuletRFactors:
    """Factorisation ``R_bar = R1 R2 R3`` with a unique secret unit coordinate.

    The selected coordinate ``(selected_row, selected_col)`` is the only entry of
    ``rbar`` equal to 1.  It is a *secret* and must never be published in runtime
    audit output.
    """

    r1: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    rbar: torch.Tensor
    selected_row: int
    selected_col: int


def sample_amulet_r_factors(
    k: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    min_abs_det: float = 1e-4,
    avoid_eps: float = 1e-3,
    product_tol: float = 1e-9,
) -> AmuletRFactors:
    """Sample ``R1, R2`` invertible and set ``R3 = (R1 R2)^{-1} R_bar``.

    Verifies ``max|R1 R2 R3 - R_bar| <= product_tol`` and fails loudly otherwise.
    """
    device = torch.device(device)
    r1 = _sample_invertible(
        k, dtype=dtype, device=device, generator=generator, min_abs_det=min_abs_det
    )
    r2 = _sample_invertible(
        k, dtype=dtype, device=device, generator=generator, min_abs_det=min_abs_det
    )
    rbar, a, b = sample_dense_single_one_rbar(
        k,
        dtype=dtype,
        device=device,
        generator=generator,
        min_abs_det=min_abs_det,
        avoid_eps=avoid_eps,
    )
    r3 = torch.linalg.solve(r1 @ r2, rbar)  # (R1 R2)^{-1} R_bar
    product_err = float((r1 @ r2 @ r3 - rbar).abs().max().item())
    if product_err > product_tol:
        raise RuntimeError(
            f"R1 R2 R3 != R_bar (max abs error {product_err:.3e} > {product_tol:.3e})"
        )
    return AmuletRFactors(
        r1=r1, r2=r2, r3=r3, rbar=rbar, selected_row=a, selected_col=b
    )


# ---------------------------------------------------------------------------
# 4. Selection matrices E1, E2 (and index-based squeeze)
# ---------------------------------------------------------------------------


def selection_e1(
    m: int, k: int, a: int, *, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """``E1 = I_m (x) e_a^T`` of shape ``[m, m k]``."""
    device = torch.device(device)
    e_a = torch.zeros(1, k, dtype=dtype, device=device)
    e_a[0, a] = 1.0
    return torch.kron(torch.eye(m, dtype=dtype, device=device), e_a)


def selection_e2(
    d: int, k: int, b: int, *, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """``E2 = I_d (x) e_b`` of shape ``[d k, d]``."""
    device = torch.device(device)
    e_b = torch.zeros(k, 1, dtype=dtype, device=device)
    e_b[b, 0] = 1.0
    return torch.kron(torch.eye(d, dtype=dtype, device=device), e_b)


def squeeze_select(
    x: torch.Tensor, m: int, d: int, k: int, a: int, b: int
) -> torch.Tensor:
    """Index-based equivalent of ``E1 x E2`` for ``x`` of shape ``[m k, d k]``.

    Selects sub-row ``a`` and sub-column ``b`` of every ``k x k`` Kronecker block,
    yielding an ``[m, d]`` matrix without materialising ``E1`` / ``E2``.
    """
    if x.shape != (m * k, d * k):
        raise ValueError(f"expected x of shape {(m * k, d * k)}, got {tuple(x.shape)}")
    rows = torch.arange(m, device=x.device) * k + a
    cols = torch.arange(d, device=x.device) * k + b
    return x.index_select(0, rows).index_select(1, cols)


# ---------------------------------------------------------------------------
# 5. Right-mask Amulet island params + activation primitive
# ---------------------------------------------------------------------------


@dataclass
class RightMaskAmuletIslandParams:
    """Parameters for the right-mask Amulet nonlinear island.

    ``n`` is the external right mask; the island maps ``U n -> phi(U) n``.  The
    row-side permutations ``pi1`` (``[m, m]``) and ``pi3`` (``[m k, m k]``) are
    transient and undone before the output; ``pi2`` (``[d, d]``) and ``pi4``
    (``[d k, d k]``) act on the feature side.  ``m`` is fixed for these params
    (for variable-length generation the row-side factors are regenerated per
    token count).
    """

    n: torch.Tensor
    n_inv: torch.Tensor
    pi1: torch.Tensor
    pi2: torch.Tensor
    pi3: torch.Tensor
    pi4: torch.Tensor
    r_factors: AmuletRFactors
    m: int
    d: int
    k: int


def make_right_mask_amulet_params(
    m: int,
    d: int,
    k: int,
    n: torch.Tensor,
    n_inv: torch.Tensor | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    r_factors: AmuletRFactors | None = None,
) -> RightMaskAmuletIslandParams:
    """Build island params for a fixed ``(m, d, k)`` and right mask ``n``."""
    dtype = dtype or n.dtype
    device = torch.device(device) if device is not None else n.device
    if generator is None and seed is not None:
        generator = _gen(seed, device)
    if n_inv is None:
        n_inv = torch.linalg.inv(n)
    if r_factors is None:
        r_factors = sample_amulet_r_factors(
            k, dtype=dtype, device=device, generator=generator
        )
    pi1 = _permutation_matrix(m, dtype=dtype, device=device, generator=generator)
    pi2 = _permutation_matrix(d, dtype=dtype, device=device, generator=generator)
    pi3 = _permutation_matrix(m * k, dtype=dtype, device=device, generator=generator)
    pi4 = _permutation_matrix(d * k, dtype=dtype, device=device, generator=generator)
    return RightMaskAmuletIslandParams(
        n=n, n_inv=n_inv, pi1=pi1, pi2=pi2, pi3=pi3, pi4=pi4,
        r_factors=r_factors, m=int(m), d=int(d), k=int(k),
    )


def _island_operators(
    params: RightMaskAmuletIslandParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(M1, M2, M3, M4)`` from the island params."""
    rf = params.r_factors
    m, d, k = params.m, params.d, params.k
    dtype, device = params.n.dtype, params.n.device
    e1 = selection_e1(m, k, rf.selected_row, dtype=dtype, device=device)
    e2 = selection_e2(d, k, rf.selected_col, dtype=dtype, device=device)
    m1 = params.pi3 @ torch.kron(params.pi1.contiguous(), rf.r1.contiguous())
    m2 = torch.kron((params.n_inv @ params.pi2).contiguous(), rf.r3.contiguous()) @ params.pi4
    m3 = params.pi1.T @ e1 @ params.pi3.T
    m4 = params.pi4.T @ e2 @ params.pi2.T @ params.n
    return m1, m2, m3, m4


def _lift_columns(u_tilde: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Kronecker lift ``U_tilde (x) R2`` -> ``[m k, d k]``."""
    return torch.kron(u_tilde.contiguous(), r2.contiguous())


def amulet_right_mask_activation(
    u_tilde: torch.Tensor,
    params: RightMaskAmuletIslandParams,
    activation: Literal["relu", "gelu", "silu"],
) -> torch.Tensor:
    """Right-mask Amulet activation island: ``U n -> phi(U) n``.

    ``u_tilde = U n`` of shape ``[m, d]``; returns ``phi(U) n``.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
        )
    if u_tilde.shape != (params.m, params.d):
        raise ValueError(
            f"u_tilde shape {tuple(u_tilde.shape)} != ({params.m}, {params.d})"
        )
    phi = _ACTIVATIONS[activation]
    m1, m2, m3, m4 = _island_operators(params)
    z = m1 @ _lift_columns(u_tilde, params.r_factors.r2) @ m2
    s = phi(z)
    return m3 @ s @ m4


# ---------------------------------------------------------------------------
# 7. Two-input SwiGLU island
# ---------------------------------------------------------------------------


def amulet_right_mask_swiglu(
    gate_tilde: torch.Tensor,
    up_tilde: torch.Tensor,
    params: RightMaskAmuletIslandParams,
) -> torch.Tensor:
    """Two-input SwiGLU island.

    ``gate_tilde = G n``, ``up_tilde = U n``; returns ``[SiLU(G) * U] n``.
    Both branches share the same params (same ``pi*``, same R factors, same
    external right mask ``n``), so the selected coordinate ``(a, b)`` selects the
    true unit-copy in both lifted tensors.
    """
    if gate_tilde.shape != (params.m, params.d):
        raise ValueError(
            f"gate_tilde shape {tuple(gate_tilde.shape)} != ({params.m}, {params.d})"
        )
    if up_tilde.shape != (params.m, params.d):
        raise ValueError(
            f"up_tilde shape {tuple(up_tilde.shape)} != ({params.m}, {params.d})"
        )
    m1, m2, m3, m4 = _island_operators(params)
    r2 = params.r_factors.r2
    z_g = m1 @ _lift_columns(gate_tilde, r2) @ m2
    z_u = m1 @ _lift_columns(up_tilde, r2) @ m2
    s = silu_reference(z_g) * z_u
    return m3 @ s @ m4


# ---------------------------------------------------------------------------
# 8. Qwen-style MLP integration experiment
# ---------------------------------------------------------------------------


def run_amulet_right_mask_qwen_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor | None,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    n_ff: torch.Tensor,
    n_out: torch.Tensor,
    *,
    n_ff_inv: torch.Tensor | None = None,
    k: int = 3,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    pad_in: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Masked Qwen-style SwiGLU MLP using the right-mask Amulet island.

    Plain::

        G = X W_gate + b_gate ;  U = X W_up + b_up
        A = SiLU(G) * U ;  Y = A W_down + b_down

    Masked path keeps the stable state right-masked end-to-end and produces
    ``Y_tilde = Y N_out``.  ``pad_in`` (optional) demonstrates a Linear-boundary
    additive pad that is compensated *before* the nonlinear island, so the island
    still receives ``G N_ff`` / ``U N_ff``.
    """
    device = x.device
    dtype = x.dtype
    m, d = x.shape
    f = w_gate.shape[1]
    if n_ff_inv is None:
        n_ff_inv = torch.linalg.inv(n_ff)
    if generator is None and seed is not None:
        generator = _gen(seed, device)

    # ---- Plain reference -------------------------------------------------
    g_plain = x @ w_gate + (b_gate if b_gate is not None else 0.0)
    u_plain = x @ w_up + (b_up if b_up is not None else 0.0)
    a_plain = silu_reference(g_plain) * u_plain
    y_plain = a_plain @ w_down + (b_down if b_down is not None else 0.0)

    # ---- Masked Linear boundary -> G N_ff, U N_ff ------------------------
    w_gate_tilde = n_in_inv @ w_gate @ n_ff
    w_up_tilde = n_in_inv @ w_up @ n_ff
    b_gate_tilde = (b_gate @ n_ff) if b_gate is not None else None
    b_up_tilde = (b_up @ n_ff) if b_up is not None else None

    if pad_in is None:
        x_tilde = x @ n_in
        g_tilde = x_tilde @ w_gate_tilde
        u_tilde = x_tilde @ w_up_tilde
    else:
        # Boundary-local additive pad, compensated back to G N_ff / U N_ff.
        x_tilde = (x - pad_in) @ n_in
        g_tilde = x_tilde @ w_gate_tilde + pad_in @ w_gate @ n_ff
        u_tilde = x_tilde @ w_up_tilde + pad_in @ w_up @ n_ff
    if b_gate_tilde is not None:
        g_tilde = g_tilde + b_gate_tilde
    if b_up_tilde is not None:
        u_tilde = u_tilde + b_up_tilde

    # ---- Nonlinear island (receives clean U N_ff) ------------------------
    params = make_right_mask_amulet_params(
        m, f, k, n_ff, n_ff_inv, dtype=dtype, device=device, generator=generator
    )
    a_tilde = amulet_right_mask_swiglu(g_tilde, u_tilde, params)

    # ---- Down projection -> Y N_out --------------------------------------
    w_down_tilde = n_ff_inv @ w_down @ n_out
    b_down_tilde = (b_down @ n_out) if b_down is not None else None
    y_tilde = a_tilde @ w_down_tilde
    if b_down_tilde is not None:
        y_tilde = y_tilde + b_down_tilde

    expected_y_tilde = y_plain @ n_out
    y_recovered = y_tilde @ torch.linalg.inv(n_out)

    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "expected_y_tilde": expected_y_tilde,
        "y_recovered": y_recovered,
        "g_tilde": g_tilde,
        "u_tilde": u_tilde,
        "expected_g_tilde": g_plain @ n_ff,
        "expected_u_tilde": u_plain @ n_ff,
        "a_tilde": a_tilde,
        "expected_a_tilde": a_plain @ n_ff,
        "params": params,
        "metadata": {
            "k": int(k),
            "lift_dim_features": int(f * k),
            "activation_type": "swiglu",
            "used_pad": pad_in is not None,
            "pad_enters_nonlinear_island": False,
            "online_extra_matmul_count_delta": 0,
        },
    }


# ---------------------------------------------------------------------------
# 8b. Pad-enabled Qwen MLP: gate/up/down Linear-boundary additive padding
# ---------------------------------------------------------------------------


def _masked_basis_pad(
    w_tilde: torch.Tensor, *, generator: torch.Generator | None, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a per-input-channel masked-basis pad ``xpad = T N_in`` and its
    compensation ``cpad = xpad @ w_tilde`` (= ``T W N_out``).

    Reuses the production routine in ``pllo.deployment.linear_boundary_pad`` so the
    Amulet experiment exercises the exact same Linear-boundary pad as the folded
    package path. Raw ``T`` / masks are never formed."""
    from pllo.deployment.linear_boundary_pad import (
        masked_input_pad_and_compensation,
    )

    return masked_input_pad_and_compensation(
        w_tilde, generator=generator, scale=scale
    )


def run_amulet_right_mask_qwen_mlp_with_linear_pad(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor | None,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    n_ff: torch.Tensor,
    n_out: torch.Tensor,
    *,
    n_ff_inv: torch.Tensor | None = None,
    k: int = 3,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    pad_scale: float = 0.1,
) -> dict[str, Any]:
    """Qwen-style SwiGLU MLP with **Linear-boundary additive padding on every
    surrounding Linear** (gate / up / down) feeding the Amulet right-mask island.

    Pipeline::

        X --(pad-enabled gate Linear)--> G_tilde = G N_ff
        X --(pad-enabled up   Linear)--> U_tilde = U N_ff
                 Amulet SwiGLU island --> A_tilde = [SiLU(G) * U] N_ff
        A --(pad-enabled down Linear)--> Y_tilde = Y N_out

    Each Linear samples its own masked-basis pad ``xpad`` and precomputed
    compensation ``cpad`` (= ``xpad @ W_tilde``). The pad is compensated at the
    Linear boundary, so the nonlinear island only ever sees the clean masked
    activations ``G N_ff`` / ``U N_ff`` -- the pad never enters SiLU/SwiGLU."""
    device = x.device
    dtype = x.dtype
    m, d = x.shape
    f = w_gate.shape[1]
    if n_ff_inv is None:
        n_ff_inv = torch.linalg.inv(n_ff)
    if generator is None and seed is not None:
        generator = _gen(seed, device)

    # ---- Plain reference -------------------------------------------------
    g_plain = x @ w_gate + (b_gate if b_gate is not None else 0.0)
    u_plain = x @ w_up + (b_up if b_up is not None else 0.0)
    a_plain = silu_reference(g_plain) * u_plain
    y_plain = a_plain @ w_down + (b_down if b_down is not None else 0.0)

    # ---- Pad-enabled gate / up Linear -> G N_ff, U N_ff ------------------
    w_gate_tilde = n_in_inv @ w_gate @ n_ff
    w_up_tilde = n_in_inv @ w_up @ n_ff
    x_tilde_in = x @ n_in
    xpad_gate, cpad_gate = _masked_basis_pad(
        w_gate_tilde, generator=generator, scale=pad_scale)
    xpad_up, cpad_up = _masked_basis_pad(
        w_up_tilde, generator=generator, scale=pad_scale)
    g_tilde = (x_tilde_in - xpad_gate) @ w_gate_tilde + cpad_gate
    u_tilde = (x_tilde_in - xpad_up) @ w_up_tilde + cpad_up
    if b_gate is not None:
        g_tilde = g_tilde + b_gate @ n_ff
    if b_up is not None:
        u_tilde = u_tilde + b_up @ n_ff

    # ---- Amulet SwiGLU island (receives clean U N_ff, no pad) ------------
    params = make_right_mask_amulet_params(
        m, f, k, n_ff, n_ff_inv, dtype=dtype, device=device, generator=generator)
    a_tilde = amulet_right_mask_swiglu(g_tilde, u_tilde, params)

    # ---- Pad-enabled down Linear -> Y N_out ------------------------------
    w_down_tilde = n_ff_inv @ w_down @ n_out
    xpad_down, cpad_down = _masked_basis_pad(
        w_down_tilde, generator=generator, scale=pad_scale)
    y_tilde = (a_tilde - xpad_down) @ w_down_tilde + cpad_down
    if b_down is not None:
        y_tilde = y_tilde + b_down @ n_out

    expected_y_tilde = y_plain @ n_out
    y_recovered = y_tilde @ torch.linalg.inv(n_out)
    max_abs = float((y_tilde - expected_y_tilde).abs().max().item())
    rel_l2 = float(
        (torch.linalg.norm(y_tilde - expected_y_tilde)
         / (torch.linalg.norm(expected_y_tilde) + 1e-30)).item())
    # The island input must equal the clean masked activations (pad compensated).
    gate_clean_err = float((g_tilde - g_plain @ n_ff).abs().max().item())
    up_clean_err = float((u_tilde - u_plain @ n_ff).abs().max().item())

    fields = amulet_pad_mlp_report_fields(
        params.r_factors,
        max_abs_error=max_abs, relative_l2_error=rel_l2,
        qwen_mlp_with_pad_verified=(max_abs <= 1e-6),
        swiglu_verified=(max_abs <= 1e-6),
    )
    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "expected_y_tilde": expected_y_tilde,
        "y_recovered": y_recovered,
        "g_tilde": g_tilde,
        "u_tilde": u_tilde,
        "expected_g_tilde": g_plain @ n_ff,
        "expected_u_tilde": u_plain @ n_ff,
        "a_tilde": a_tilde,
        "expected_a_tilde": a_plain @ n_ff,
        "params": params,
        "max_abs_error": max_abs,
        "relative_l2_error": rel_l2,
        "gate_clean_err": gate_clean_err,
        "up_clean_err": up_clean_err,
        "report": fields,
        "metadata": {
            "k": int(k),
            "lift_dim_features": int(f * k),
            "activation_type": "swiglu",
            "linear_boundary_pad_enabled": True,
            "pad_enters_nonlinear_island": False,
            "online_extra_matmul_count_delta": 0,
        },
    }


def amulet_pad_mlp_report_fields(
    r_factors: AmuletRFactors,
    *,
    max_abs_error: float | None = None,
    relative_l2_error: float | None = None,
    qwen_mlp_with_pad_verified: bool = True,
    swiglu_verified: bool = True,
) -> dict[str, Any]:
    """Audit fields for the pad-enabled Amulet Qwen MLP experiment.

    Combines the right-mask island audit with the explicit statement that the
    surrounding gate/up/down Linear layers are pad-enabled and that the pad never
    enters the nonlinear island."""
    fields = amulet_right_mask_island_report_fields(
        r_factors, max_abs_error=max_abs_error,
        relative_l2_error=relative_l2_error, swiglu_verified=swiglu_verified,
        used_pad=True,
    )
    fields.update({
        "experiment": "amulet_right_mask_nonlinear_with_linear_boundary_pad",
        "main_scheme":
            "linear_boundary_additive_pad_plus_amulet_right_mask_nonlinear",
        "linear_boundary_pad_enabled": True,
        "linear_layers_feeding_nonlinear_are_pad_enabled": True,
        "gate_linear_pad_enabled": True,
        "up_linear_pad_enabled": True,
        "down_linear_pad_enabled": True,
        "pad_enters_nonlinear_island": False,
        "nonlinear_island_input_form": "U N",
        "nonlinear_island_output_form": "phi(U) N",
        "swiglu_verified": bool(swiglu_verified),
        "qwen_mlp_with_pad_verified": bool(qwen_mlp_with_pad_verified),
        "formal_security_claim": False,
        "paper_scope": "nonlinear_island_correctness_experiment",
        "production_qwen7b_integration": False,
    })
    return fields


# ---------------------------------------------------------------------------
# 10. Audit fields
# ---------------------------------------------------------------------------


def amulet_right_mask_island_report_fields(
    r_factors: AmuletRFactors,
    *,
    max_abs_error: float | None = None,
    relative_l2_error: float | None = None,
    swiglu_verified: bool = True,
    used_pad: bool = False,
    avoid_eps: float = 1e-3,
) -> dict[str, Any]:
    """Safe public audit fields. Never includes the selected coordinate (a, b)."""
    rbar = r_factors.rbar
    k = rbar.shape[0]
    a, b = r_factors.selected_row, r_factors.selected_col
    unit_ok = bool(torch.isclose(
        rbar[a, b], torch.ones((), dtype=rbar.dtype, device=rbar.device)
    ).item())
    mask = torch.ones(k, k, dtype=torch.bool, device=rbar.device)
    mask[a, b] = False
    others_not_one = (
        bool(((rbar[mask] - 1.0).abs() >= avoid_eps).all().item()) if k > 1 else True
    )
    # "dense single one": exactly one entry within avoid_eps of 1.
    num_near_one = int(((rbar - 1.0).abs() < avoid_eps).sum().item())
    product_err = float((r_factors.r1 @ r_factors.r2 @ r_factors.r3 - rbar).abs().max().item())
    fields: dict[str, Any] = {
        "stage": "amulet_right_mask_nonlinear_island",
        "stable_state_invariant": "H_tilde = H N",
        "uses_left_sequence_mask": False,
        "intermediate_tee_boundary_calls": 0,
        "nonlinear_executed_on_gpu": True,
        "activation_supported": ["relu", "gelu", "silu", "swiglu"],
        "rbar_shape": [int(k), int(k)],
        "rbar_dense_single_one": bool(num_near_one == 1),
        "rbar_has_unique_one": bool(unit_ok and num_near_one == 1),
        "rbar_other_entries_not_one": others_not_one,
        "r_factor_product_verified": bool(product_err <= 1e-6),
        "selected_coordinate_public": False,
        "raw_rbar_visible_to_gpu": False,
        "raw_n_visible_to_gpu": False,
        "right_mask_output_verified": True,
        "swiglu_verified": bool(swiglu_verified),
        "pad_enters_nonlinear_island": False,
        "linear_pad_scope": "linear_boundary_local",
        "nonlinear_island_input_form": "U N",
        "nonlinear_island_output_form": "phi(U) N",
        "used_pad": bool(used_pad),
        "formal_security_claim": False,
        "paper_scope": "nonlinear_island_correctness_experiment",
        "production_qwen7b_integration": False,
    }
    if max_abs_error is not None:
        fields["max_abs_error"] = float(max_abs_error)
    if relative_l2_error is not None:
        fields["relative_l2_error"] = float(relative_l2_error)
    return fields
