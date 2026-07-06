"""Real-Qwen attack representations for all four defenses (single model load).

Captures the plaintext residual state ``H = embed_tokens(ids)`` of a REAL Qwen2
checkpoint once, then applies each defense's *actual* cloud-visible transform to
the SAME ``H`` (and to a real weight matrix, for the weight-leakage worst case)
so every attack compares like-for-like on the real model:

  plaintext_gpu                P = H                       (no protection)
  obfuscatune_qwen_orthogonal  P = H @ R      (orthogonal R; ObfuscaTune X*=XR)
  stip_qwen                    P = H[:, pi]   (STIP feature permutation)
  ours_amulet_style            P = H @ N_res  (signed-perm residual mask; the
                                              exact mask tee/runtime_api derives)

Optionally also ``ours_amulet_style_fresh_pad`` — a *fresh* orthogonal mask per
token (the hardened variant), which has no single global mask and so defeats a
global known-plaintext solve.

Secrets (R, pi, N_res) are generated here and never written into an attack's
view of the data (``secret_metadata_present_but_not_revealed=True``). The
``true_permutation``/``signs`` placed in ``model_weights`` / ``defense_metadata``
are used ONLY to *score* recovery, mirroring the toy harness convention.

The residual-mask math is identical to :mod:`pllo.tee.runtime_api`
(``derive_residual_mask`` / ``apply_signed_permutation``); the orthogonal mask
reuses :func:`pllo.baselines.obfuscatune.random_matrices.orthogonal_matrix`.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.baselines.obfuscatune.random_matrices import orthogonal_matrix
from .representations import AttackInputs

METHODS = ["plaintext_gpu", "stip_qwen", "obfuscatune_qwen_orthogonal", "ours_amulet_style"]


def _signed_perm(hidden: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed-permutation residual mask N_res: (perm, signs) — cf.
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
    model, ids: torch.Tensor, *, seed: int = 0, include_fresh_pad: bool = False,
) -> dict[str, AttackInputs]:
    """Build {method: AttackInputs} from one real-Qwen forward.

    ``ids``: (1, S) or (S,) token ids. A real weight matrix (layer-0 q_proj) is
    folded by each secret to give the weight-leakage worst-case inputs.
    """
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    H, table = capture_plaintext_hidden(model, ids)          # (N,d), (V,d)
    H = H.to(torch.float32)
    table = table.to(torch.float32)
    hidden = H.shape[-1]
    flat_ids = ids.reshape(-1)
    W = model.model.layers[0].self_attn.q_proj.weight.detach().to(torch.float32)  # (out, in=d)
    g = torch.Generator().manual_seed(seed)

    def _pack(method, protected, *, w_obf=None, perm=None, signs=None,
              global_mask=True, fresh_pad=False) -> AttackInputs:
        meta: dict[str, Any] = {"method": method, "obfuscation": method,
                                "global_matrix": global_mask, "fresh_pad": fresh_pad}
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
        kp = (H, protected) if global_mask else None
        return AttackInputs(
            token_ids=flat_ids, plaintext_embeddings=H, protected_embeddings=protected,
            observed_intermediate=protected, embedding_table=table, model_weights=mw,
            known_plaintext_pairs=kp, downstream=None, defense_metadata=meta,
            secret_metadata_present_but_not_revealed=True)

    reps: dict[str, AttackInputs] = {}

    # plaintext: no protection (upper-bound leak)
    reps["plaintext_gpu"] = _pack("plaintext_gpu", H.clone())

    # ObfuscaTune: X* = X R, R orthogonal (kappa=1); weight columns mixed by R
    R, _ = orthogonal_matrix(hidden, seed=seed, dtype=torch.float32)
    reps["obfuscatune_qwen_orthogonal"] = _pack(
        "obfuscatune_qwen_orthogonal", H @ R, w_obf=W @ R)

    # STIP: feature permutation (global residual pi is arbitrary at the stream)
    pi = torch.randperm(hidden, generator=g)
    reps["stip_qwen"] = _pack("stip_qwen", H[:, pi], w_obf=W[:, pi], perm=pi)

    # ours (amulet-style): signed-perm residual mask N_res = tee/runtime_api mask
    perm, signs = _signed_perm(hidden, g)
    P_ours = H[:, perm] * signs                              # H @ N_res
    W_ours = W[:, perm] * signs                              # columns signed-permuted
    reps["ours_amulet_style"] = _pack(
        "ours_amulet_style", P_ours, w_obf=W_ours, perm=perm, signs=signs)

    if include_fresh_pad:
        # hardened variant: a fresh orthogonal mask per token -> no global mask
        fresh = torch.stack([H[i] @ orthogonal_matrix(hidden, seed=seed + 1 + i,
                                                       dtype=torch.float32)[0]
                             for i in range(H.shape[0])])
        reps["ours_amulet_style_fresh_pad"] = _pack(
            "ours_amulet_style_fresh_pad", fresh, global_mask=False, fresh_pad=True)

    return reps


__all__ = ["build_real_representations", "capture_plaintext_hidden", "METHODS"]
