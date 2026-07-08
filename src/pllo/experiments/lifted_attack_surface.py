"""Attack-surface comparison for the three masked-inference variants.

Three **distinct** public attack surfaces are measured separately (never mixed):

  A. stable-state surface   -- attacker observes the offloaded stable state
                               (H N  or  Pi (H (x) R) Omega); metrics: per-token
                               norm correlation + Gram correlation.
  B. folded-weight surface  -- attacker observes the offloaded folded weights
                               (N_in^{-1} W N_out  or  Omega_in^{-1} (W (x) S)
                               Omega_out); metric: left self-Gram diagonal
                               distinctness (= how recoverable the left mask is).
  C. transient surface      -- attacker observes the transient nonlinear tensor
                               (masked activation, Amulet Z, or the plain squeeze
                               point); metric: whether the true activation channel
                               is directly observable.

Honest framing (this is the whole point of the study):

    ||X (x) R||_F   = ||X||_F ||R||_F
    (X (x) R)(X (x) R)^T = (X X^T) (x) (R R^T)

so a Kronecker lift does **not** remove the norm/Gram signal.  A *token-order-
preserving* lift (Pi = I_m (x) P_k, required so KV-cache/causal order survive)
leaves the per-token norm and the token-block Gram exactly proportional to the
plaintext ones -- i.e. it leaks norm/Gram just like a plain orthogonal right
mask.  Only dense *cross-token* mixing reduces this, and that breaks the decoder
cache.  The numbers below make that trade explicit.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.ops.kronecker_lifted_linear import (
    kron_lift,
    lifted_linear_weight,
    make_lift_params,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-30:
        return 0.0
    return float((a @ b).item() / denom)


def _signed_perm(d: int, *, generator, dtype, device) -> torch.Tensor:
    perm = torch.randperm(d, generator=generator, device=device)
    signs = (torch.randint(0, 2, (d,), generator=generator, device=device) * 2 - 1).to(dtype)
    n = torch.zeros(d, d, dtype=dtype, device=device)
    n[torch.arange(d, device=device), perm] = signs
    return n


def _orthogonal(d: int, *, generator, dtype, device) -> torch.Tensor:
    g = torch.randn(d, d, dtype=torch.float64, device=device, generator=generator)
    q, _ = torch.linalg.qr(g)
    return q.to(dtype)


# ---------------------------------------------------------------------------
# Surface A: stable state
# ---------------------------------------------------------------------------


def stable_state_leakage(
    h: torch.Tensor, observed: torch.Tensor, m: int, k_group: int
) -> dict[str, float]:
    """Per-token norm + Gram correlation between plaintext ``H`` and ``observed``.

    ``observed`` has ``m * k_group`` rows grouped into ``m`` token blocks (in
    order -- the token-order-preserving lift keeps token blocks contiguous, so the
    attacker aligns tokens trivially).  ``k_group = 1`` for a plain right mask.
    """
    plain_norm = h.norm(dim=1)                                  # [m]
    gram_plain = h @ h.T                                        # [m, m]
    if k_group == 1:
        obs_norm = observed.norm(dim=1)                         # [m]
        gram_obs = observed @ observed.T
    else:
        blocks = observed.reshape(m, k_group, -1)               # [m, k, dk]
        obs_norm = blocks.reshape(m, -1).norm(dim=1)            # [m] aggregate
        # Frobenius inner product per token pair -> [m, m] token-block Gram.
        gram_obs = torch.einsum("iac,jac->ij", blocks, blocks)
    return {
        "norm_corr": pearson(plain_norm, obs_norm),
        "gram_corr": pearson(gram_plain, gram_obs),
    }


# ---------------------------------------------------------------------------
# Surface B: folded weights (left self-Gram recoverability of the left mask)
# ---------------------------------------------------------------------------


def _diag_distinct_frac(g: torch.Tensor) -> float:
    d = g.shape[0]
    diag = torch.diagonal(g)
    return torch.unique(torch.round(diag * 1e4)).numel() / d


def folded_weight_alignment(
    w_folded: torch.Tensor, w_public: torch.Tensor, *, left_lift_k: int
) -> dict[str, float]:
    """Left self-Gram recoverability of the left mask from folded weights.

    For a plain fold ``W' = P^{-1} A M`` (P left mask, M orthogonal output mask),
    ``W' W'^T = P (A A^T) P^T`` -- a signed permutation P is recovered by matching
    ``diag(W' W'^T)`` to ``diag(A A^T)`` iff those diagonals are distinct.  For a
    lifted fold ``W_hat = Omega_in^{-1} (A (x) S) Omega_out`` the left self-Gram is
    ``Omega_in^{-1} [(A A^T) (x) (S S^T)] Omega_in^{-T}``: the diagonal is a
    permuted Kronecker product, so recoverability is governed by its distinctness.
    """
    g_folded = w_folded @ w_folded.T
    g_public = w_public @ w_public.T
    # Diagonal-match recovery rate against the public left-Gram diagonal.
    dgf, dgp = torch.diagonal(g_folded), torch.diagonal(g_public)
    if dgf.numel() == dgp.numel():
        match = (dgf.view(-1, 1) - dgp.view(1, -1)).abs().argmin(dim=1)
        # For a plain fold the diagonals are a permutation of each other; the
        # recovery "works" when the matched diagonals agree in value.
        matched_err = float(
            (dgf - dgp[match]).abs().max().item()
            / (dgp.abs().max().item() + 1e-30)
        )
    else:
        matched_err = float("nan")                              # dim mismatch (lifted)
    return {
        "folded_gram_diag_distinct_frac": _diag_distinct_frac(g_folded),
        "public_gram_diag_distinct_frac": _diag_distinct_frac(g_public),
        "left_mask_diag_match_rel_err": matched_err,
        "left_lift_k": int(left_lift_k),
    }


# ---------------------------------------------------------------------------
# Surface C: transient nonlinear tensor
# ---------------------------------------------------------------------------


def transient_channel_observability(variant: str, k: int) -> dict[str, Any]:
    """Qualitative: is the true nonlinear activation directly observable?

    * variant A (pure-right): nonlinearity runs on the masked state; no expansion.
    * variant B (amulet_gelu): expands to a dense single-one Kronecker Z; the true
      channel is one of k^2 dense channels (not a sparse one-hot) -> not directly
      observable.
    * variant C (lifted linear, C1/C2): the elementwise nonlinearity is computed at
      a *plain* squeeze point (transient), so the true activation IS observable at
      that instant.
    """
    if variant == "right_pad_pure_right":
        return {
            "nonlinear_expansion": False,
            "true_channel_directly_observable": True,   # masked, but no decoys
            "note": "nonlinearity on masked state; leakage governed by mask, not decoys",
        }
    if variant == "right_pad_amulet_gelu":
        return {
            "nonlinear_expansion": True,
            "kronecker_channels": k * k,
            "true_channel_directly_observable": False,
            "note": "dense single-one R_bar: true channel is 1 of k^2 dense channels",
        }
    if variant == "kronecker_lifted_linear":
        return {
            "nonlinear_expansion": True,
            "true_channel_directly_observable": True,
            "note": ("nonlinearity computed at a PLAIN squeeze point (C1/C2 both "
                     "squeeze internally); transient plain activation exists"),
        }
    raise KeyError(variant)


# ---------------------------------------------------------------------------
# Full per-variant attack-surface run
# ---------------------------------------------------------------------------


def run_variant_attack_surface(
    variant: str,
    *,
    m: int = 16,
    d: int = 32,
    p: int = 64,
    k: int = 2,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    token_safe_rows: bool = True,
) -> dict[str, Any]:
    """Measure all three attack surfaces for one variant. Reproducible from seed."""
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    h = torch.randn(m, d, dtype=dtype, device=dev, generator=g)
    w = torch.randn(d, p, dtype=dtype, device=dev, generator=g)

    if variant in ("right_pad_pure_right", "right_pad_amulet_gelu"):
        # Plain orthogonal signed-permutation right mask (A and B share this;
        # Amulet only lifts the transient nonlinear tensor, not the stable state).
        n_in = _signed_perm(d, generator=g, dtype=dtype, device=dev)
        n_out = _orthogonal(p, generator=g, dtype=dtype, device=dev)
        observed_state = h @ n_in
        w_folded = torch.linalg.inv(n_in) @ w @ n_out
        k_group = 1
        left_lift_k = 1
    elif variant == "kronecker_lifted_linear":
        in_params = make_lift_params(
            m, d, k, generator=g, dense_single_one=True,
            token_safe_rows=token_safe_rows, dtype=dtype, device=dev,
        )
        out_params = make_lift_params(
            m, p, k, pi=in_params.pi, generator=g, dense_single_one=True,
            token_safe_rows=token_safe_rows, dtype=dtype, device=dev,
        )
        observed_state = kron_lift(h, in_params)
        w_folded = lifted_linear_weight(w, in_params, out_params)
        k_group = k
        left_lift_k = k
    else:
        raise KeyError(variant)

    surface_a = stable_state_leakage(h, observed_state, m, k_group)
    surface_b = folded_weight_alignment(w_folded, w, left_lift_k=left_lift_k)
    surface_c = transient_channel_observability(variant, k)

    return {
        "variant": variant,
        "config": {
            "m": m, "d": d, "p": p, "lift_factor": k, "seed": seed,
            "token_safe_rows": token_safe_rows,
            "dtype": str(dtype).replace("torch.", ""),
        },
        "surface_A_stable_state": surface_a,
        "surface_B_folded_weights": surface_b,
        "surface_C_transient": surface_c,
    }
