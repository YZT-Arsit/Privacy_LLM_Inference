"""Adapt STIP-on-Qwen protected output into the attack representation interface."""

from __future__ import annotations

from typing import Any

import torch

from .permutation import make_stip_permutations
from .qwen_stip import stip_protect_qwen


def stip_attack_representation(model, input_ids: torch.Tensor, *, seed: int = 0,
                              layer: int = 0) -> dict[str, Any]:
    """Return a dict compatible with attacks.representations.AttackInputs fields.

    The cloud-visible (permuted) hidden state is the ``observed_intermediate``;
    the permuted embedding is ``protected_embeddings``; the device-side plaintext
    embedding table is available (public model). Known-plaintext pairs are the
    (plaintext, permuted) embeddings for KPA.
    """
    from pllo.attacks.representations import AttackInputs
    out = stip_protect_qwen(model, input_ids, seed=seed)
    plain_emb = out["plaintext_embedding"][0]         # (S, H)
    prot_emb = out["protected_embedding"][0]
    observed = out.get(f"layer{layer}_hidden_perm")
    observed = observed[0] if observed is not None else prot_emb
    ai = AttackInputs(
        token_ids=out["input_ids"][0],
        plaintext_embeddings=plain_emb,
        protected_embeddings=prot_emb,
        observed_intermediate=observed,
        embedding_table=out["embedding_table"],
        known_plaintext_pairs=(plain_emb, prot_emb),
        defense_metadata={
            "method": "stip_qwen",
            "obfuscation": "feature_permutation",
            "attention_scores_unpermuted": True,
            "per_token_norm_preserved": True,
            "coordinate_multiset_preserved": True,
            "kv_cache_protected": False,
            "global_matrix": True,      # a single global permutation exists (KPA-solvable on π)
        },
    )
    return {"attack_inputs": ai, "stip_output": out}


__all__ = ["stip_attack_representation"]
