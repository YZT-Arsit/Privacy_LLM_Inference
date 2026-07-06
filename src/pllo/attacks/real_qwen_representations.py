"""Real-Qwen attack representations for the defense comparison (single load).

Captures the plaintext residual state ``H = embed_tokens(ids)`` of a REAL Qwen2
checkpoint once, then applies each defense's *actual* cloud-visible transform to
the SAME ``H`` (and to a real weight matrix, for the weight-leakage worst case)
so every attack compares like-for-like on the real model.

Columns (explicit — no more ambiguous ``ours_amulet_style``):

  plaintext_gpu                 P = H                        (no protection)
  obfuscatune_qwen_orthogonal   P = H @ R      (ONE fixed orthogonal R; the
                                               ObfuscaTune weights are obfuscated
                                               once, so R is static per deployment)
  stip_qwen                     P = H[:, pi]   (STIP feature permutation, static)
  ours_amulet_style_signed_perm P = H @ N_res  (current mainline: ONE static
                                               signed-perm residual mask; cf.
                                               tee/runtime_api.derive_residual_mask)
  ours_amulet_style_fresh_pad   P_i = H_i @ Q_i (candidate: a FRESH orthogonal
                                               mask per token — no global mask)
  ours_non_isometric_variant    P_i = H_i @ M_i (design candidate: FRESH
                                               non-orthogonal well-conditioned
                                               mask; NOT norm-preserving. Uses the
                                               audited matrix_with_condition_number.
                                               is_design_candidate=True — needs a
                                               folded N_i^{-1} correctness proof.)

Every method carries known_plaintext_pairs (X, X*) so KPA *runs* on the fresh
variants and is *measured as failing* (high held-out error) rather than blocked.
Secrets (R, pi, N_res, per-token masks) never enter an attack's view
(secret_metadata_present_but_not_revealed=True); ``true_permutation`` / ``signs``
placed in model_weights / defense_metadata are used ONLY to score recovery.

Kronecker note: the repo's Kronecker construction
(``pllo.ops.amulet_right_mask_islands`` / amulet_secure_R) is a *nonlinear-island
lift* that expands dimensions to hide activations — it is NOT a same-dim
residual-stream mask, so there is no faithful "Kronecker residual column" and
those variants are reported design_needed by the runners, never fabricated.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.baselines.obfuscatune.random_matrices import (matrix_with_condition_number,
                                                        orthogonal_matrix)
from .representations import AttackInputs

# columns this module can actually BUILD (measured); kronecker variants are
# design_needed and handled by the runner, not here.
BUILDABLE_METHODS = [
    "plaintext_gpu", "stip_qwen", "obfuscatune_qwen_orthogonal",
    "ours_amulet_style_signed_perm", "ours_amulet_style_fresh_pad",
    "ours_non_isometric_variant",
]
DESIGN_NEEDED_METHODS = ["ours_amulet_style_kronecker", "ours_fresh_pad_kronecker"]


def _signed_perm(hidden: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed-permutation residual mask N_res — cf.
    ``pllo.tee.runtime_api.derive_residual_mask``."""
    perm = torch.randperm(hidden, generator=g)
    signs = torch.where(torch.rand(hidden, generator=g) < 0.5,
                        torch.tensor(-1.0), torch.tensor(1.0))
    return perm, signs


def capture_plaintext_hidden(model, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (H, table): the plaintext residual entering layer 0 (= embed(ids))
    flattened to (N, hidden), and the embedding table (V, hidden)."""
    with torch.no_grad():
        table = model.model.embed_tokens.weight.detach()
        H = model.model.embed_tokens(ids).detach()
    return H.reshape(-1, H.shape[-1]), table


def build_real_representations(
    model, ids: torch.Tensor, *, seed: int = 0, non_isometric_cond: float = 5.0,
) -> dict[str, AttackInputs]:
    """Build {method: AttackInputs} from one real-Qwen forward.

    Includes the fresh-pad and non-isometric candidates as first-class columns.
    ``non_isometric_cond``: target condition number (>1 ⇒ non-orthogonal ⇒ NOT
    norm-preserving) for the design-candidate variant.
    """
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    H, table = capture_plaintext_hidden(model, ids)          # (N,d), (V,d)
    H = H.to(torch.float32)
    table = table.to(torch.float32)
    hidden = H.shape[-1]
    n = H.shape[0]
    flat_ids = ids.reshape(-1)
    W = model.model.layers[0].self_attn.q_proj.weight.detach().to(torch.float32)  # (out, in=d)
    g = torch.Generator().manual_seed(seed)

    def _pack(method, protected, *, transform_type, norm_preserving, fresh, kron,
              pad, w_obf=None, perm=None, signs=None, design_candidate=False) -> AttackInputs:
        meta: dict[str, Any] = {
            "method": method, "obfuscation": method, "transform_type": transform_type,
            "is_norm_preserving_theoretically": norm_preserving,
            "whether_fresh_per_sample": fresh, "whether_kronecker_enabled": kron,
            "whether_pad_enabled": pad, "is_design_candidate": design_candidate,
            "global_matrix": not fresh, "fresh_pad": fresh,
        }
        mw = None
        if perm is not None:
            meta["true_permutation"] = perm.tolist()
        if signs is not None:
            meta["signs"] = signs.tolist()
        if w_obf is not None:
            mw = {"public": W, "obfuscated": w_obf}
            if perm is not None:
                mw["true_permutation"] = perm.tolist()
            if signs is not None:
                mw["signs"] = signs.tolist()
        # KPA pairs are provided for EVERY method: for fresh masks the single
        # global solve simply fails to generalise (measured, not blocked).
        return AttackInputs(
            token_ids=flat_ids, plaintext_embeddings=H, protected_embeddings=protected,
            observed_intermediate=protected, embedding_table=table, model_weights=mw,
            known_plaintext_pairs=(H, protected), downstream=None, defense_metadata=meta,
            secret_metadata_present_but_not_revealed=True)

    reps: dict[str, AttackInputs] = {}

    # plaintext: no protection (upper-bound leak). No weight obfuscation.
    reps["plaintext_gpu"] = _pack(
        "plaintext_gpu", H.clone(), transform_type="identity", norm_preserving=True,
        fresh=False, kron=False, pad=False)

    # ObfuscaTune: X* = X R, ONE fixed orthogonal R (kappa=1); norm-preserving.
    R, _ = orthogonal_matrix(hidden, seed=seed, dtype=torch.float32)
    reps["obfuscatune_qwen_orthogonal"] = _pack(
        "obfuscatune_qwen_orthogonal", H @ R, transform_type="fixed_orthogonal",
        norm_preserving=True, fresh=False, kron=False, pad=False, w_obf=W @ R)

    # STIP: static feature permutation.
    pi = torch.randperm(hidden, generator=g)
    reps["stip_qwen"] = _pack(
        "stip_qwen", H[:, pi], transform_type="fixed_permutation", norm_preserving=True,
        fresh=False, kron=False, pad=False, w_obf=W[:, pi], perm=pi)

    # ours current mainline: static signed-perm residual mask N_res.
    perm, signs = _signed_perm(hidden, g)
    reps["ours_amulet_style_signed_perm"] = _pack(
        "ours_amulet_style_signed_perm", H[:, perm] * signs,
        transform_type="fixed_signed_permutation", norm_preserving=True, fresh=False,
        kron=False, pad=False, w_obf=W[:, perm] * signs, perm=perm, signs=signs)

    # candidate: FRESH orthogonal mask per token (norm-preserving; no global mask).
    fresh_orth = torch.stack([
        H[i] @ orthogonal_matrix(hidden, seed=seed + 1 + i, dtype=torch.float32)[0]
        for i in range(n)])
    reps["ours_amulet_style_fresh_pad"] = _pack(
        "ours_amulet_style_fresh_pad", fresh_orth, transform_type="fresh_orthogonal_per_token",
        norm_preserving=True, fresh=True, kron=False, pad=False)

    # design candidate: FRESH non-orthogonal well-conditioned mask (NOT norm-preserving).
    fresh_ni = torch.stack([
        H[i] @ matrix_with_condition_number(hidden, cond=non_isometric_cond,
                                            seed=seed + 10_000 + i, dtype=torch.float32)[0]
        for i in range(n)])
    reps["ours_non_isometric_variant"] = _pack(
        "ours_non_isometric_variant", fresh_ni,
        transform_type=f"fresh_non_orthogonal_cond{non_isometric_cond:g}",
        norm_preserving=False, fresh=True, kron=False, pad=False, design_candidate=True)

    return reps


__all__ = ["build_real_representations", "capture_plaintext_hidden",
           "BUILDABLE_METHODS", "DESIGN_NEEDED_METHODS"]
