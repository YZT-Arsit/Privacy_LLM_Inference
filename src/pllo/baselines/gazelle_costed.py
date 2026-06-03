"""Stage 7.5c - Gazelle / Delphi / SecureML / MiniONN cost-model-only baselines.

These are cryptographic-MPC / hybrid-HE inference systems whose runtime
is dominated by network rounds and ciphertext sizes that we cannot honest-
ly reproduce without a real implementation. We refuse to fabricate
runtime numbers. Instead this module records:

* Whether the system is implemented (always ``False`` here);
* Whether a real cryptographic library would be required (always ``True``);
* The protocol's cost dimensions as cited in the original paper
  (rounds, ciphertext modulus bit-width, online vs.\\ offline cost);
* The threat model (semi-honest / malicious / static-corruption).

``directly_comparable_on_runtime`` is hard-coded to ``False`` for every
row produced here. These rows belong in Related Work and in a
cost-model comparison; they are not measured baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pllo.baselines.baseline_protocol import (
    BaselineProtocol,
    BaselineSelfDeclaration,
    UnsupportedResult,
)


def _decl(name: str, paper: str, notes: str) -> BaselineSelfDeclaration:
    return BaselineSelfDeclaration(
        name=name,
        paper=paper,
        exact_primitive_implemented=False,
        full_system_reproduced=False,
        requires_crypto_library=True,
        supports_static_forward=False,
        supports_decoder_generation=False,
        supports_kv_cache_append=False,
        supports_lora_training=False,
        arithmetic_skeleton_only=False,
        cost_model_only=True,
        notes=notes,
    )


@dataclass
class CostModelEntry:
    """Cost dimensions from a paper -- NOT a measured local result."""

    protocol_rounds: str
    ciphertext_modulus_bits: str
    online_offline_split: str
    threat_model: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_rounds": self.protocol_rounds,
            "ciphertext_modulus_bits": self.ciphertext_modulus_bits,
            "online_offline_split": self.online_offline_split,
            "threat_model": self.threat_model,
            "notes": self.notes,
        }


class CostModelBaseline(BaselineProtocol):
    """Common base for cost-model-only baselines."""

    directly_comparable_on_runtime: bool = False

    def __init__(self, cost: CostModelEntry) -> None:
        self.cost = cost

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "directly_comparable_on_runtime": False,
            "reason": (
                "no executable cryptographic protocol implemented; cost"
                " dimensions only"
            ),
            "cost_model": self.cost.to_dict(),
        }

    def decode_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="cost-model baseline cannot execute decode",
            implementation_scope_reason="no cryptographic protocol implementation in this artifact",
        )

    def train_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="cost-model baseline cannot execute training",
            implementation_scope_reason="no cryptographic protocol implementation in this artifact",
        )


class GazelleCostModel(CostModelBaseline):
    declare = _decl(
        "gazelle_cost_model_only",
        "GAZELLE (USENIX Security 2018)",
        "Hybrid HE+GC inference; cost-model only -- no implementation here.",
    )

    def __init__(self) -> None:
        super().__init__(
            CostModelEntry(
                protocol_rounds=(
                    "multi-round HE convolution + garbled-circuit ReLU;"
                    " several rounds per nonlinear layer"
                ),
                ciphertext_modulus_bits="60-180 (BFV; depends on tile and depth)",
                online_offline_split=(
                    "online: HE matvec + GC ReLU per layer; offline:"
                    " key generation + GC table prep"
                ),
                threat_model="semi-honest; single-server inference",
                notes=(
                    "Real GAZELLE runtimes depend on SEAL/HElib parameters"
                    " and a real network; not measured here."
                ),
            )
        )


class DelphiCostModel(CostModelBaseline):
    declare = _decl(
        "delphi_cost_model_only",
        "Delphi (USENIX Security 2020)",
        "Hybrid neural-cryptographic inference; cost-model only.",
    )

    def __init__(self) -> None:
        super().__init__(
            CostModelEntry(
                protocol_rounds=(
                    "linear layers via HE; nonlinear approximated by"
                    " quadratic + secret sharing; multiple rounds"
                ),
                ciphertext_modulus_bits="120-200",
                online_offline_split=(
                    "online dominated by share computation; offline by"
                    " preprocessing"
                ),
                threat_model="semi-honest; single-server inference",
            )
        )


class SecureMLCostModel(CostModelBaseline):
    declare = _decl(
        "secureml_cost_model_only",
        "SecureML (IEEE S&P 2017)",
        "Two-party secret-sharing inference and training; cost-model only.",
    )

    def __init__(self) -> None:
        super().__init__(
            CostModelEntry(
                protocol_rounds=(
                    "two-party additive sharing; multiple online rounds"
                    " per layer for multiplication triples"
                ),
                ciphertext_modulus_bits="ring Z_{2^64} (no HE; secret sharing)",
                online_offline_split=(
                    "online dominated by share comm.; offline by Beaver"
                    " triple generation"
                ),
                threat_model=(
                    "semi-honest; two non-colluding servers"
                ),
            )
        )


class MiniONNCostModel(CostModelBaseline):
    declare = _decl(
        "minionn_cost_model_only",
        "MiniONN (CCS 2017)",
        "Oblivious inference via HE+GC; cost-model only.",
    )

    def __init__(self) -> None:
        super().__init__(
            CostModelEntry(
                protocol_rounds=(
                    "HE-based linear + GC-based ReLU; multiple rounds"
                    " per nonlinear layer"
                ),
                ciphertext_modulus_bits="60-180 (BFV)",
                online_offline_split=(
                    "online matvec + GC eval; offline keygen + GC prep"
                ),
                threat_model="semi-honest; client + server",
            )
        )


__all__ = [
    "CostModelEntry",
    "CostModelBaseline",
    "GazelleCostModel",
    "DelphiCostModel",
    "SecureMLCostModel",
    "MiniONNCostModel",
]
