"""Per-token norm-leakage audit across the defense variants.

Answers: why is frequency/per-token-norm leakage 1.0 for the current columns,
is Kronecker in the tested path, and what actually breaks norm leakage.

For each method we compare the per-token norm of the plaintext residual ‖H_i‖ to
the per-token norm of the cloud-visible protected state ‖P_i‖:

  * norm_preserved_exactly: max_i |‖P_i‖-‖H_i‖| / ‖H_i‖  (≈0 ⇒ isometry)
  * measured_norm_leakage : Pearson corr(‖H_i‖, ‖P_i‖)   (≈1 ⇒ norm leaks)

Any ISOMETRY (orthogonal / permutation / signed-perm / a FRESH orthogonal mask)
preserves ‖·‖ exactly ⇒ norm leaks (corr=1). Only a NON-isometric transform
(fresh non-orthogonal well-conditioned mask, random per-token scaling, additive
in-TEE norm noise) decorrelates ‖P_i‖ from ‖H_i‖ and breaks the leak.
"""

from __future__ import annotations

from typing import Any

import torch


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64); b = b.to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def audit_norm_leakage(reps: dict[str, Any]) -> dict[str, Any]:
    """reps: {method: AttackInputs} from build_real_representations."""
    per_method = {}
    for method, inp in reps.items():
        H = inp.plaintext_embeddings.to(torch.float32)
        P = inp.protected_embeddings.to(torch.float32)
        hn = H.norm(dim=-1)
        pn = P.norm(dim=-1)
        rel = float(((pn - hn).abs() / hn.clamp_min(1e-9)).max())
        corr = _pearson(hn, pn)
        meta = inp.defense_metadata
        per_method[method] = {
            "method": method,
            "transform_type": meta.get("transform_type"),
            "is_norm_preserving_theoretically": meta.get("is_norm_preserving_theoretically"),
            "whether_kronecker_enabled": meta.get("whether_kronecker_enabled"),
            "whether_pad_enabled": meta.get("whether_pad_enabled"),
            "whether_fresh_per_token": meta.get("whether_fresh_per_sample"),
            "is_design_candidate": meta.get("is_design_candidate", False),
            "observed_tensor_name": "protected_state_at_layer0_input",
            "norm_preserved_exactly_max_rel_err": rel,
            "measured_norm_leakage_pearson": corr,
            "breaks_norm_leakage": bool(corr < 0.9),
        }
    # global recommendation
    breaker = [m for m, r in per_method.items() if r["breaks_norm_leakage"]]
    rec = (
        "Every isometry (orthogonal R, permutation, signed-perm, and even a FRESH "
        "orthogonal per-token mask) preserves per-token norm exactly, so norm leaks "
        "(corr=1.0) — norm=1.0 is CORRECT, not a measurement bug. Kronecker in this "
        "repo is a nonlinear-island lift (amulet_secure_R), NOT a residual mask, so it "
        "is not in this path and would not change residual norm leakage. To BREAK norm "
        "leakage the residual mask must be NON-isometric: a fresh non-orthogonal "
        "well-conditioned mask (measured here as ours_non_isometric_variant), a random "
        "per-token positive-diagonal scaling, or additive in-TEE norm noise. These "
        "require folding N_i^{-1} into the next linear to keep correctness — a design "
        "change, validated numerically before any deployment claim."
    )
    return {"per_method": per_method, "norm_leakage_breakers": breaker,
            "recommendation": rec}


__all__ = ["audit_norm_leakage"]
