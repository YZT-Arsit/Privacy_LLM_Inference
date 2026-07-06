"""STIP baseline (Secure Transformer Inference Protocol, Yuan et al.).

Feature-dimension permutation obfuscation (NO TEE). Adapted to Qwen2 with the
correctness fixes the paper omits (RoPE-safe whole-head permutation, GQA groups,
SwiGLU shared intermediate permutation, QKV biases). See
``docs/attack_paper_audit.md`` and ``docs/baseline_audit_obfuscatune_stip.md``.

Kept strictly separate from our amulet-style / trusted-shortcut code and from the
ObfuscaTune baseline.
"""

from __future__ import annotations

from .permutation import (
    StipPermutations,
    apply_perm_last,
    inverse_perm,
    make_stip_permutations,
    random_permutation,
    whole_head_permutation,
)
from .qwen_stip import (
    build_transformed_qwen,
    stip_parameter_transform_linear,
    stip_protect_qwen,
    stip_recover_logits,
    verify_linear_equivalence,
)

__all__ = [
    "StipPermutations", "apply_perm_last", "inverse_perm", "make_stip_permutations",
    "random_permutation", "whole_head_permutation",
    "stip_parameter_transform_linear", "stip_protect_qwen", "stip_recover_logits",
    "build_transformed_qwen", "verify_linear_equivalence",
]
