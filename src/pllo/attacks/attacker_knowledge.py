"""Threat models + attacker-knowledge presets (Kerckhoffs setting).

Kerckhoffs: the attacker KNOWS the defense algorithm but NOT the TEE secrets
(orthogonal matrices, padding, Kronecker factors, fresh masks). The public Qwen
model used in experiments is a stand-in; in real deployment the attacker does
NOT hold the private model weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackerKnowledge:
    """A named bundle of attacker-capability flags for AttackResult fields."""

    threat_model: str
    attacker_has_weights: bool
    attacker_has_embedding_table: bool
    attacker_has_obfuscated_pairs: bool
    attacker_has_intermediate_activations: bool
    attacker_has_defense_algorithm: bool = True   # Kerckhoffs: always True
    attacker_has_secret_keys: bool = False        # TEE secrets: never

    def as_fields(self) -> dict:
        return {
            "threat_model": self.threat_model,
            "attacker_has_weights": self.attacker_has_weights,
            "attacker_has_embedding_table": self.attacker_has_embedding_table,
            "attacker_has_obfuscated_pairs": self.attacker_has_obfuscated_pairs,
            "attacker_has_intermediate_activations": self.attacker_has_intermediate_activations,
            "attacker_has_defense_algorithm": self.attacker_has_defense_algorithm,
            "attacker_has_secret_keys": self.attacker_has_secret_keys,
        }


# Default deployment: closed model, no weight access, sees obfuscated activations.
CLOSED_MODEL = AttackerKnowledge(
    threat_model="closed_model_no_weight_access",
    attacker_has_weights=False,
    attacker_has_embedding_table=False,
    attacker_has_obfuscated_pairs=False,
    attacker_has_intermediate_activations=True,
)

# Public-model sanity setting: attacker has the (public) embedding table.
PUBLIC_MODEL = AttackerKnowledge(
    threat_model="public_model",
    attacker_has_weights=True,
    attacker_has_embedding_table=True,
    attacker_has_obfuscated_pairs=False,
    attacker_has_intermediate_activations=True,
)

# Known-plaintext (KPA): attacker holds (plaintext, obfuscated) pairs.
KNOWN_PLAINTEXT = AttackerKnowledge(
    threat_model="known_plaintext",
    attacker_has_weights=False,
    attacker_has_embedding_table=False,
    attacker_has_obfuscated_pairs=True,
    attacker_has_intermediate_activations=True,
)

# Worst case: model weights leaked (used ONLY for consequence analysis).
WEIGHT_LEAKAGE_WORST_CASE = AttackerKnowledge(
    threat_model="weight_leakage_worst_case",
    attacker_has_weights=True,
    attacker_has_embedding_table=True,
    attacker_has_obfuscated_pairs=True,
    attacker_has_intermediate_activations=True,
)

PRESETS = {
    "closed_model": CLOSED_MODEL,
    "public_model": PUBLIC_MODEL,
    "known_plaintext": KNOWN_PLAINTEXT,
    "weight_leakage_worst_case": WEIGHT_LEAKAGE_WORST_CASE,
}


def get_preset(name: str) -> AttackerKnowledge:
    if name not in PRESETS:
        raise KeyError(f"unknown attacker-knowledge preset {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]


__all__ = [
    "AttackerKnowledge", "CLOSED_MODEL", "PUBLIC_MODEL", "KNOWN_PLAINTEXT",
    "WEIGHT_LEAKAGE_WORST_CASE", "PRESETS", "get_preset",
]
