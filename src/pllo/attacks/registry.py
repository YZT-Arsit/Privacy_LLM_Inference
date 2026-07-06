"""Registry of attack probes: id -> family, priority, implementation status."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackSpec:
    attack_id: str
    attack_family: str
    priority: str                 # "P0" | "P1" | "P2"
    implementation_level: str     # full | best_effort | partial | blocked
    requires_gpu_for_scale: bool
    paper: str
    reproduction_scope: str


ATTACKS: dict[str, AttackSpec] = {
    "nn_embedding_inversion": AttackSpec(
        "nn_embedding_inversion", "structural", "P0", "full", False,
        "EDNN (Lin et al. EMNLP'24) / embedding inversion",
        "NN + EDNN differential inversion. Full."),
    "eia_optimization": AttackSpec(
        "eia_optimization", "optimization", "P0", "best_effort", True,
        "EIA (Song & Raghunathan CCS'20 family)",
        "Simplified/modern optimization inversion. Not a verbatim reproduction."),
    "bre_bisr_forward": AttackSpec(
        "bre_bisr_forward", "optimization", "P0", "best_effort", True,
        "BRE/BiSR (Chen et al. CCS'24)",
        "Forward smashed-data matching (Eq.7-8) only; NOT full BiSR."),
    "bre_bisr_backward": AttackSpec(
        "bre_bisr_backward", "optimization", "P0", "blocked", True,
        "BRE/BiSR (Chen et al. CCS'24)",
        "Backward gradient matching (Eq.9): needs split-FT training gradients. Blocked."),
    "kpa_known_plaintext": AttackSpec(
        "kpa_known_plaintext", "cryptanalysis", "P0", "full", False,
        "KPA on linear/permutation obfuscation (ObfuscaTune/STIP)",
        "Linear + permutation known-plaintext recovery. Full."),
    "multiset_permutation_leakage": AttackSpec(
        "multiset_permutation_leakage", "structural", "P1", "full", True,
        "Permutation attack (Thomas et al. 2505.18332)",
        "Structural leakage probe + Alg.3 sorted-L1 autoregressive decode. Full."),
    "arrowmatch_weight_alignment": AttackSpec(
        "arrowmatch_weight_alignment", "alignment", "P1", "full", False,
        "Game of Arrows / ArrowMatch (Wang et al. USENIX Sec'25)",
        "S1 cosine argmin + S2 length ratio. Full. Weight-leakage worst case only."),
    "pia_prompt_inversion": AttackSpec(
        "pia_prompt_inversion", "optimization", "P2", "best_effort", True,
        "PIA (Qu et al. IEEE S&P'25)",
        "Phase1 constrained opt + Phase2 discretization; oracle-LLM S_s omitted."),
    "frequency_distribution": AttackSpec(
        "frequency_distribution", "statistical", "P2", "full", False,
        "Frequency/distribution analysis (supplementary)",
        "Norm/frequency rank-correlation leakage probe. Full (supplementary)."),
}


def get_attack(attack_id: str) -> AttackSpec:
    if attack_id not in ATTACKS:
        raise KeyError(f"unknown attack {attack_id!r}; known: {sorted(ATTACKS)}")
    return ATTACKS[attack_id]


__all__ = ["AttackSpec", "ATTACKS", "get_attack"]
