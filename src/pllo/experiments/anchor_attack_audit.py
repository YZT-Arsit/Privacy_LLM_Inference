"""Anchor-based weight-alignment attack audit (AloePri/ArrowMatch/VMA family).

Threat model: the attacker holds the PUBLIC weight ``W`` (anchor), the offloaded
obfuscated weight ``W_tilde``, and knows the mechanism; the secret keys
(perm/signs or P/Q) are unknown. Every attack here REQUIRES the public-weight
anchor (``requires_plaintext_weight_anchor = True``) — none of them applies under
a proprietary (private-weights) deployment. We report weight-alignment recovery
SEPARATELY from token/prompt recovery and never conflate the two.

Attacks:
  * Gram alignment       — left/right column-Gram diagonal match (recovers a
    permutation fold; a dense congruence resists it). Plus a Gram correlation.
  * VMA (vector matching) — row-norm and column-norm sort/rank matching and
    nearest-neighbour cosine matching with top-k recovery.
  * ArrowMatch           — cosine column matching + length ratio (reuses
    ``pllo.attacks.arrowmatch_attack``).
  * Mask-recovery proxy  — if a permutation was recovered, un-mask the stable
    state and score hidden-state / token recovery; otherwise report "no mapping
    recovered" (NOT a token-privacy failure).

Compared variants (all fold the SAME public W, so the comparison is like-for-like):
  * ``signed_perm``      — column signed permutation ``W[:,pi]·s`` (the A_rightmul
    production residual-mask surface; right_pad_pure_right and right_pad_amulet_gelu
    expose this identical folded-weight surface — Amulet only lifts the transient).
  * ``dense_right_orthogonal`` — ``W R`` with R orthogonal (ObfuscaTune reference).
  * ``two_sided_nonorthogonal_exact`` — ``Q_in W P_out`` (Variant D).
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.baselines.obfuscatune.random_matrices import orthogonal_matrix
from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat


# ---------------------------------------------------------------------------
# variant folds: given public W, produce (W_tilde, secret) for each variant
# ---------------------------------------------------------------------------


def _signed_perm(d: int, g: torch.Generator, dtype, device):
    perm = torch.randperm(d, generator=g, device=device)
    signs = torch.where(torch.rand(d, generator=g, device=device) < 0.5,
                        torch.tensor(-1.0), torch.tensor(1.0)).to(dtype)
    return perm, signs


def build_variant_fold(variant: str, w: torch.Tensor, *, lam: float, seed: int,
                       dtype, device) -> dict[str, Any]:
    """Return {'w_obf', 'true_perm' or None, 'p_in'/'p_out' for stable-state proxy}."""
    d_in, d_out = w.shape
    g = torch.Generator(device=device); g.manual_seed(seed)
    if variant == "signed_perm":
        # column signed-perm fold on the OUTPUT dim (matches the main-table Gram target)
        perm, signs = _signed_perm(d_out, g, dtype, device)
        w_obf = w[:, perm] * signs
        return {"w_obf": w_obf, "true_perm": perm, "signs": signs, "p_out": None}
    if variant == "dense_right_orthogonal":
        r, _ = orthogonal_matrix(d_out, seed=seed, dtype=dtype, device=device)
        return {"w_obf": w @ r, "true_perm": None, "p_out": r}
    if variant == "two_sided_nonorthogonal_exact":
        p_in, q_in, _ = sample_nonorthogonal_keymat(d_in, lam, seed=seed, dtype=dtype, device=device)
        p_out, _, _ = sample_nonorthogonal_keymat(d_out, lam, seed=seed + 100, dtype=dtype, device=device)
        return {"w_obf": q_in @ w @ p_out, "true_perm": None, "p_in": p_in, "p_out": p_out}
    raise KeyError(variant)


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float64); b = b.reshape(-1).to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    den = (a.norm() * b.norm()).item()
    return 0.0 if den < 1e-30 else float((a @ b).item() / den)


def gram_alignment(w_pub: torch.Tensor, w_obf: torch.Tensor) -> dict[str, float]:
    """Column-Gram diagonal match (both sides) + Gram correlation.

    diag-match rel err ~0 => the fold is a permutation and is recovered; a dense
    congruence gives a large rel err (not recoverable by diagonal matching)."""
    def diag_match_rel_err(a, b):                      # a,b: (n,) diagonals
        m = (a.view(-1, 1) - b.view(1, -1)).abs().argmin(dim=1)
        return float((a - b[m]).abs().max() / (b.abs().max() + 1e-30))
    wp, wo = w_pub.to(torch.float64), w_obf.to(torch.float64)
    # right column-Gram: diag = squared column norms
    gpr, gor = (wp.T @ wp), (wo.T @ wo)
    # left row-Gram
    gpl, gol = (wp @ wp.T), (wo @ wo.T)
    return {
        "gram_right_diag_match_rel_err": diag_match_rel_err(torch.diagonal(gor), torch.diagonal(gpr)),
        "gram_left_diag_match_rel_err": diag_match_rel_err(torch.diagonal(gol), torch.diagonal(gpl)),
        # sorted-spectrum correlation (permutation- & basis-agnostic): stays high
        # for any congruence, so it is NOT a recovery metric — reported for context.
        "gram_right_spectrum_corr": _pearson(torch.linalg.eigvalsh(gpr), torch.linalg.eigvalsh(gor)),
    }


def vma_matching(w_pub: torch.Tensor, w_obf: torch.Tensor,
                 true_perm: torch.Tensor | None) -> dict[str, float]:
    """Vector-matching attack on COLUMNS: norm-rank match + cosine NN top-k.

    Recovers the column correspondence when the fold is a permutation (+sign);
    dense mixing changes column directions and norms, defeating it."""
    wp, wo = w_pub.to(torch.float64), w_obf.to(torch.float64)
    n = wp.shape[1]
    # (1) column-norm rank matching: sort both by column norm, match by rank
    pn, on = wp.norm(dim=0), wo.norm(dim=0)
    rank_map = on.argsort().argsort()                      # obf col -> rank
    pub_by_rank = pn.argsort()                             # rank -> pub col
    norm_pred = pub_by_rank[rank_map]                      # obf col -> pub col
    # (2) cosine NN matching (sign-invariant, so it survives sign flips)
    a = torch.nn.functional.normalize(wo, dim=0)
    b = torch.nn.functional.normalize(wp, dim=0)
    cos = (a.T @ b).abs()                                  # (obf, pub)
    nn1 = cos.argmax(dim=1)
    top5 = cos.topk(min(5, n), dim=1).indices
    if true_perm is not None:
        tp = true_perm.to(norm_pred.device)
        col_norm_acc = float((norm_pred == tp).float().mean())
        col_cos_top1 = float((nn1 == tp).float().mean())
        col_cos_top5 = float((top5 == tp.view(-1, 1)).any(dim=1).float().mean())
    else:
        # no ground-truth permutation (dense fold): score self-consistency of the
        # recovered map as an UPPER bound on what the attacker could exploit —
        # here, how strongly cosine picks a unique match (max |cos|). Low => dense
        # mixing left no column direction to latch onto.
        col_norm_acc = float("nan")
        col_cos_top1 = float(cos.max(dim=1).values.mean())   # mean best |cos|
        col_cos_top5 = float("nan")
    return {
        "vma_col_norm_rank_acc": col_norm_acc,
        "vma_col_cos_top1": col_cos_top1,
        "vma_col_cos_top5": col_cos_top5,
    }


def run_anchor_audit(
    variant: str,
    *,
    d_in: int = 64,
    d_out: int = 64,
    lam: float = 0.1,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    m_tokens: int = 32,
) -> dict[str, Any]:
    """Run Gram + VMA + ArrowMatch on one variant's folded weight, plus a
    mask-recovery proxy on the stable state. Reproducible from seed."""
    from pllo.attacks.arrowmatch_attack import arrowmatch_single
    from pllo.attacks.metrics import permutation_recovery_accuracy

    g = torch.Generator(device=device); g.manual_seed(seed)
    w = torch.randn(d_in, d_out, dtype=dtype, device=device, generator=g)
    fold = build_variant_fold(variant, w, lam=lam, seed=seed, dtype=dtype, device=device)
    w_obf, true_perm = fold["w_obf"], fold["true_perm"]

    gram = gram_alignment(w, w_obf)
    vma = vma_matching(w, w_obf, true_perm)
    # ArrowMatch: cosine column matching. Score against true perm if known, else
    # the reconstruction-cosine alignment fraction (how many matched columns align).
    sigma, _ = arrowmatch_single(w_obf, w, bijection=True)
    if true_perm is not None:
        arrow_acc = float(permutation_recovery_accuracy(sigma, true_perm))
    else:
        cd = 1.0 - torch.nn.functional.normalize(w_obf.to(torch.float32), dim=0).T @ \
            torch.nn.functional.normalize(w.to(torch.float32), dim=0)
        arrow_acc = float((cd.gather(1, sigma.view(-1, 1)).squeeze(1) < 0.05).float().mean())

    # --- mask-recovery proxy on the stable state -------------------------------
    # Only meaningful when the column mapping was actually recovered. For dense
    # folds no mapping is recovered, so hidden-state recovery is NOT attempted (we
    # do NOT claim token/prompt privacy failure without a measured recovery). For a
    # permutation fold we recover perm via cosine-NN (sign-invariant) and the signs
    # via the matched-column dot-product sign, then demonstrate that the SAME mask
    # governs the residual stream (A_rightmul: the weight fold and the H·N mask
    # share the mask), so un-masking H·N recovers H exactly.
    hidden_state_unmask_err = None
    mapping_recovery_acc = float("nan")
    if true_perm is not None:
        wp = w.to(torch.float64); wo = w_obf.to(torch.float64)
        cosmap = (torch.nn.functional.normalize(wo, dim=0).T
                  @ torch.nn.functional.normalize(wp, dim=0))       # (obf,pub)
        perm_hat = cosmap.abs().argmax(dim=1)                       # obf col -> pub col
        mapping_recovery_acc = float((perm_hat == true_perm.to(perm_hat.device)).float().mean())
        if mapping_recovery_acc == 1.0:
            sign_hat = torch.sign(cosmap.gather(1, perm_hat.view(-1, 1)).squeeze(1)).to(dtype)
            h = torch.randn(m_tokens, d_out, dtype=dtype, device=device, generator=g)
            signs = fold.get("signs")
            h_masked = h[:, true_perm] * (signs if signs is not None else 1.0)  # = H·N
            inv = torch.empty_like(perm_hat); inv[perm_hat] = torch.arange(perm_hat.numel(), device=perm_hat.device)
            h_rec = (h_masked * sign_hat)[:, inv]                   # un-mask with recovered (perm,signs)
            hidden_state_unmask_err = float((h_rec - h).abs().max())
    mask_recovered = bool(mapping_recovery_acc == 1.0)

    return {
        "variant": variant,
        "attack_surface": "anchor_weight_alignment",
        "requires_plaintext_weight_anchor": True,
        "config": {"d_in": d_in, "d_out": d_out, "lambda": lam, "seed": seed,
                   "dtype": str(dtype).replace("torch.", "")},
        **gram,
        **vma,
        "arrow_match_acc": arrow_acc,
        "weight_mapping_recovery_acc": mapping_recovery_acc,
        "weight_mapping_fully_recovered": mask_recovered,
        "hidden_state_unmask_max_abs_err": hidden_state_unmask_err,
        "note": ("permutation folds (signed_perm) are recovered by Gram/VMA/Arrow;"
                 "dense congruences (orthogonal or two-sided non-orthogonal) are not. "
                 "Token recovery only attempted where a mapping was actually recovered."),
    }


__all__ = ["build_variant_fold", "gram_alignment", "vma_matching", "run_anchor_audit"]
