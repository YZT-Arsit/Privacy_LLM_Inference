"""Experiment-level *variant* registry for the masked-inference comparison.

This is a **lightweight, experiment-only** registry that names the three
masked-inference variants compared in the Kronecker-lifted study.  It is
deliberately separate from the paper-facing nonlinear-design/attestation
registry (``pllo.experiments.nonlinear_designs``): variants A and B *reference*
existing paper-facing designs, while variant C (``kronecker_lifted_linear``) is
a new **experiment-only** linear/residual-stream transform that is NOT wired into
the production Qwen7B worker or the attestation pipeline.

    A. right_pad_pure_right   -> design A_rightmul       (stable state H N)
    B. right_pad_amulet_gelu  -> design amulet_secure_R  (stable state H N;
                                 transient Kronecker lift inside GELU/SiLU only)
    C. kronecker_lifted_linear-> NEW, experiment-only     (stable state / folded
                                 linear weights live in a lifted Kronecker space)

Every record is JSON-serializable and carries the honest-scope flags requested
for the study.  None of these records is hashed into attestation.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "VARIANTS",
    "get_variant",
    "list_variants",
    "variant_report_fields",
]


VARIANTS: Dict[str, Dict[str, Any]] = {
    "right_pad_pure_right": {
        "variant": "right_pad_pure_right",
        "label": "A",
        "references_design": "A_rightmul",
        "linear_layer": "right_mask_plus_pad",
        "nonlinear": "pure_right_compatible_nonlinear",
        "stable_state_form": "H_tilde = H N",
        "uses_right_mask": True,
        "uses_kronecker_lifted_state": False,
        "lifted_linear_weights": False,
        "lifted_residual_stream": False,
        "attention_kv_lifted": False,
        "public_attack_surface": ["stable_state_HN", "folded_weights_NinvWM"],
        "experiment_only": False,
        "formal_security_claim": False,
        "production_qwen7b_integration": True,
    },
    "right_pad_amulet_gelu": {
        "variant": "right_pad_amulet_gelu",
        "label": "B",
        "references_design": "amulet_secure_R",
        "linear_layer": "right_mask_plus_pad",
        "nonlinear": "right_mask_amulet_kronecker_gelu_silu",
        "stable_state_form": "H_tilde = H N",
        "uses_right_mask": True,
        "uses_kronecker_lifted_state": False,          # only a *transient* lift
        "transient_kronecker_lift_in_nonlinear": True,
        "lifted_linear_weights": False,
        "lifted_residual_stream": False,
        "attention_kv_lifted": False,
        "public_attack_surface": ["stable_state_HN", "folded_weights_NinvWM"],
        "note": (
            "Amulet R_bar lifts only the transient nonlinear tensor Z; the "
            "squeezed stable state is again H N, so stable-state Gram/norm/"
            "weight-alignment attacks are unaffected by R_bar."),
        "experiment_only": False,
        "formal_security_claim": False,
        "production_qwen7b_integration": True,
    },
    "two_sided_nonorthogonal_exact": {
        "variant": "two_sided_nonorthogonal_exact",
        "label": "D",
        "aliases": ["exact_keymat_two_sided", "variant_d"],
        "references_design": None,
        "linear_layer": "two_sided_nonorthogonal_keymat",
        "nonlinear": "none_exact_linear_chain_only",   # P non-orthogonal: no exact nonlinearity pass
        "stable_state_form": "X_tilde = X P_in (P_in non-orthogonal)",
        "folded_weight_form": "W_tilde = Q_in W P_out",
        "uses_two_sided_keymat": True,
        "uses_nonorthogonal_transform": True,
        "uses_noise": False,                           # NOT AloePri: no Gaussian noise
        "uses_rmsnorm_expectation_correction": False,
        "uses_right_mask": False,
        "uses_kronecker_lifted_state": False,
        "lifted_linear_weights": False,
        "lifted_residual_stream": False,
        "attention_kv_lifted": False,
        "exact_scope": "linear_chain_only",
        "passes_elementwise_nonlinearity_exactly": False,
        "passes_rmsnorm_exactly": False,
        "public_attack_surface": ["stable_state_XPin", "folded_weights_QinWPout"],
        "anchor_attack_audit": True,
        "requires_plaintext_weight_anchor": True,
        "is_aloepri_replication": False,
        "experiment_only": True,
        "formal_security_claim": False,
        "production_qwen7b_integration": False,
        "security_note": (
            "Two-sided non-orthogonal fold scrambles BOTH the left row-Gram and the "
            "right column-Gram (unlike a right-only mask), so Gram/VMA/ArrowMatch "
            "anchor attacks recover no mapping — while staying exact for the LINEAR "
            "chain. Cost: non-orthogonal P does not commute with RMSNorm/elementwise "
            "nonlinearity, so end-to-end needs TEE crossings (ObfuscaTune-class); and "
            "it is a STATIC per-deployment key (KPA-broken). See "
            "docs/rmsnorm_exact_norm_impossibility.md."),
    },
    "two_sided_amulet_island": {
        "variant": "two_sided_amulet_island",
        "label": "E",
        "aliases": ["exact_two_sided_nonlinear_switch", "variant_e"],
        "references_design": None,
        "linear_layer": "two_sided_nonorthogonal_keymat_island",
        "nonlinear": "amulet_gelu_or_permutation_swiglu_boundary",
        "stable_state_form": "H_tilde = H N_res (residual rides N_res; dense keymat only inside linear islands)",
        "uses_two_sided_keymat": True,
        "uses_nonorthogonal_transform": True,
        "uses_noise": False,
        "uses_mask_switching_at_nonlinear_boundary": True,
        "nonlinear_mode": "amulet_or_structured",
        "residual_boundary_mask": "N_res_signed_permutation",
        "elementwise_boundary_mask": "pure_permutation",
        "dense_keymat_crosses_rmsnorm": False,
        "dense_keymat_crosses_rope": False,
        "dense_keymat_crosses_hadamard": False,
        "exact_scope": "linear_islands_plus_compatible_nonlinear_boundary",
        "changes_stable_state_norm_leakage": False,     # residual still rides N_res
        "public_attack_surface": ["stable_state_HNres", "island_folded_weights_QinWPout",
                                  "boundary_switch_weights_no_public_anchor"],
        "anchor_attack_audit": True,
        "is_aloepri_replication": False,
        "experiment_only": True,
        "formal_security_claim": False,
        "production_qwen7b_integration": False,
        "security_note": (
            "Local island: two-sided non-orthogonal keymat protects the LINEAR "
            "folded weights (anchor-resistant like Variant D), switching back to a "
            "compatible mask (signed-perm N_res / permutation S) before every "
            "nonlinear/RMSNorm/RoPE/Hadamard/residual boundary. Exact for the GPT-2 "
            "GELU MLP island (Amulet-GELU) and the Qwen SwiGLU FFN island "
            "(permutation Hadamard). Does NOT change stable-state norm leakage (the "
            "residual still rides N_res). See docs/rmsnorm_exact_norm_impossibility.md."),
    },
    "selective_token_mixing_shield": {
        "variant": "selective_token_mixing_shield",
        "label": "F",
        "aliases": ["stms_boundary_shield", "variant_f"],
        "references_design": None,
        "linear_layer": "boundary_token_dim_mixing_shield",
        "nonlinear": "none_boundary_only",
        "stable_state_form": "U = A (H N) at observable boundary; recovered to H N before compute",
        "uses_token_dim_mixing": True,
        "fresh_per_batch_mixing": True,
        "crosses_attention": False,
        "crosses_rmsnorm": False,
        "crosses_rope": False,
        "crosses_kv_cache": False,
        "shield_rows_optional": True,
        "is_boundary_shield_not_full_left_mixing": True,
        "protects_compute_visible_HN": False,   # only the shielded artifact U
        "exact_lossless_claim": "only_if_recovered_before_compute",
        "public_attack_surface": ["shielded_boundary_artifact_U"],
        "is_aloepri_replication": False,
        "experiment_only": True,
        "formal_security_claim": False,
        "production_qwen7b_integration": False,
        "security_note": (
            "Boundary shield only: token-dim fresh mix A shields the observable "
            "U = A(H N) against an observer WITHOUT A that does not see the "
            "post-recovery H N. Orthogonal A leaves the singular-value spectrum of "
            "H INVARIANT (U U^T = A(H H^T)A^T), so a spectrum-matching forward "
            "oracle still ranks candidates; non-orthogonal A distorts the spectrum "
            "but reduces to BSS/ICA hardness. Does NOT protect the compute-visible "
            "H N (the V4 / weight-recovery threat model) — A never crosses "
            "attention/RMSNorm/RoPE/KV. See results/attacks/stms_boundary_shield/."),
    },
    "kronecker_lifted_linear": {
        "variant": "kronecker_lifted_linear",
        "label": "C",
        "aliases": ["lifted_linear", "lifted_residual", "kron_lifted_linear"],
        "references_design": None,
        "linear_layer": "kronecker_lifted_weight",
        "nonlinear": "squeeze_around_nonlinear_or_transient_lifted",
        "stable_state_form": "H_hat = Pi (H (x) R) Omega",
        "uses_right_mask": "partial",
        "uses_kronecker_lifted_state": True,
        "lift_factor": None,                            # filled per-run
        "lifted_linear_weights": True,
        "lifted_residual_stream": "experimental",       # C2 only
        "attention_kv_lifted": False,
        "public_attack_surface": [
            "lifted_stable_state_Pi_HkronR_Omega",
            "lifted_folded_weights_W_hat",
        ],
        "stages": {
            "C0": "lifted linear layer + lifted folded weights (exact)",
            "C1": ("lifted up/gate/down weights, squeeze around nonlinearity; "
                   "protects folded MLP weights only, NOT the stable residual "
                   "stream"),
            "C2": ("lifted residual stream across the MLP block; nonlinearity "
                   "transiently squeezes internally; attention/KV NOT lifted "
                   "(experimental)"),
        },
        "experiment_only": True,
        "formal_security_claim": False,
        "production_qwen7b_integration": False,
        "security_note": (
            "Kronecker expansion does NOT information-theoretically remove "
            "norm/Gram (||X kron R|| = ||X|| ||R||; (X kron R)(X kron R)^T = "
            "(X X^T) kron (R R^T)). Security relies on hidden block structure + "
            "permutation/mixing; recovering the block structure re-exposes "
            "Gram/norm."),
    },
}


def _key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


_ALIAS: Dict[str, str] = {}
for _canon, _rec in VARIANTS.items():
    _ALIAS[_key(_canon)] = _canon
    for _a in _rec.get("aliases", []):
        _ALIAS[_key(_a)] = _canon


def list_variants() -> list[str]:
    return list(VARIANTS)


def get_variant(name: str) -> Dict[str, Any]:
    canon = _ALIAS.get(_key(name))
    if canon is None:
        raise KeyError(
            f"unknown variant {name!r}; known: {sorted(VARIANTS)}"
        )
    return dict(VARIANTS[canon])


def variant_report_fields(name: str, *, lift_factor: int | None = None) -> Dict[str, Any]:
    """Return the JSON-safe report record for a variant, stamping ``lift_factor``."""
    rec = get_variant(name)
    if lift_factor is not None and rec["variant"] == "kronecker_lifted_linear":
        rec["lift_factor"] = int(lift_factor)
    return rec
