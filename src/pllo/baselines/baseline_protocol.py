"""Stage 7.5c - common base class for direct prior-work primitive implementations.

Every baseline in :mod:`pllo.baselines` subclasses :class:`BaselineProtocol`
and self-declares: which primitives it actually implements from a known
paper formula, which it explicitly does NOT implement, and which features
(decoder generation, KV cache append, LoRA training) are out of its scope
under its threat model. ``UnsupportedResult`` is a first-class result, not
an error; the paper relies on it to keep the comparison honest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UnsupportedResult:
    """A baseline returning this is *the* experimental result for that op."""

    reason: str
    mathematical_reason: str = ""
    paper_scope_reason: str = ""
    implementation_scope_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineSelfDeclaration:
    name: str
    paper: str
    exact_primitive_implemented: bool
    full_system_reproduced: bool
    requires_crypto_library: bool
    supports_static_forward: bool
    supports_decoder_generation: bool
    supports_kv_cache_append: bool
    supports_lora_training: bool
    arithmetic_skeleton_only: bool = False
    cost_model_only: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineProtocol:
    """Base class. Concrete baselines override the relevant ``*_forward`` /
    ``*_step`` methods and the ``declare()`` class attribute.
    """

    declare: BaselineSelfDeclaration  # set on subclasses

    # ------------------------------------------------------------------
    # Default implementations: every op is unsupported until overridden.
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="static forward is not implemented for this baseline",
            implementation_scope_reason="default BaselineProtocol stub",
        )

    def prefill(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="prefill is not implemented for this baseline",
            implementation_scope_reason="default BaselineProtocol stub",
        )

    def decode_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="decode_step is not implemented for this baseline",
            implementation_scope_reason="default BaselineProtocol stub",
        )

    def train_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="LoRA training is not implemented for this baseline",
            implementation_scope_reason="default BaselineProtocol stub",
        )

    def explain_unsupported(self, op: str) -> UnsupportedResult:
        return UnsupportedResult(
            reason=f"{op} is not implemented for this baseline",
            implementation_scope_reason=(
                "default BaselineProtocol.explain_unsupported"
            ),
        )

    def threat_model_summary(self) -> dict[str, Any]:
        d = self.declare.to_dict()
        d["threat_model"] = (
            "as stated in the baseline paper; the implementation here only"
            " reproduces the primitive listed under exact_primitive_implemented."
        )
        return d

    def transcript_summary(self) -> dict[str, Any]:
        return {
            "baseline": self.declare.name,
            "paper": self.declare.paper,
            "transcript_published": "primitive-level only; not a full-system trace",
        }


__all__ = [
    "BaselineProtocol",
    "BaselineSelfDeclaration",
    "UnsupportedResult",
]
