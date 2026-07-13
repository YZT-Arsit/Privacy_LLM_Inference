"""Fail-closed data contracts for the authorized confidentiality evaluation.

The attacker-facing code receives :class:`ViewRecord` objects, never an oracle,
model, checkpoint path, or the unfiltered source dictionary.  A record is
validated against an exact per-view allowlist at construction and access time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class View(str, Enum):
    V0 = "V0"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"


COMMON = frozenset({"sample_id", "run_id", "task", "split"})
ALLOWLISTS: dict[View, frozenset[str]] = {
    View.V0: COMMON | frozenset({"final_label", "output_text", "output_token_ids"}),
    View.V1: COMMON | frozenset({
        "final_label", "output_text", "output_token_ids", "api_confidence",
        "api_token_probabilities",
    }),
    View.V2: COMMON | frozenset({
        "transformed_base", "transformed_adapter", "transformed_hidden",
        "masked_q", "masked_k", "masked_v", "attention_scores",
        "masked_kv_cache", "masked_logits", "masked_dlogits",
        "transformed_gradients", "transformed_optimizer_state",
        "tensor_shapes", "package_metadata", "protocol_transcript",
    }),
    View.V3: COMMON | frozenset({
        "final_label", "output_text", "output_token_ids", "input_token_ids",
        "input_text", "reference", "labels", "logits", "hidden_states",
        "attention_scores", "plaintext_weights", "plaintext_adapter",
        "plaintext_gradients", "plaintext_lora_state", "loss",
    }),
}

# These evaluator/TDX-only names are forbidden even if a future allowlist is
# edited accidentally.  V3 is the sole exception because it is the trusted
# positive-control view.
HARD_FORBIDDEN = frozenset({
    "plaintext_w", "plaintext_weights", "plaintext_a", "plaintext_b",
    "plaintext_adapter", "plaintext_deltaw", "mask", "masks", "inverse",
    "inverses", "gamma", "rank_secret", "tdx_secret", "tdx_checkpoint",
    "reference", "labels", "input_text", "model", "oracle",
})


class ViewAccessError(PermissionError):
    pass


def _contains_runtime_object(value: Any) -> bool:
    """Reject live model/oracle-like objects without importing torch."""
    if isinstance(value, (str, bytes, int, float, bool, type(None), list, tuple, dict)):
        if isinstance(value, dict):
            return any(_contains_runtime_object(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(_contains_runtime_object(v) for v in value)
        return False
    module = type(value).__module__
    name = type(value).__name__.lower()
    return module.startswith("torch.nn") or "model" in name or "oracle" in name


@dataclass(frozen=True)
class ViewRecord:
    view: View
    _data: Mapping[str, Any]

    @classmethod
    def build(cls, view: str | View, data: Mapping[str, Any]) -> "ViewRecord":
        resolved = View(view)
        keys = frozenset(data)
        unexpected = keys - ALLOWLISTS[resolved]
        if unexpected:
            raise ViewAccessError(
                f"{resolved.value} rejects fields outside its allowlist: {sorted(unexpected)}"
            )
        if resolved is not View.V3:
            forbidden = keys & HARD_FORBIDDEN
            if forbidden:
                raise ViewAccessError(
                    f"{resolved.value} rejects evaluator/TDX-only fields: {sorted(forbidden)}"
                )
        if any(_contains_runtime_object(v) for v in data.values()):
            raise ViewAccessError(f"{resolved.value} rejects live model/oracle objects")
        return cls(resolved, MappingProxyType(dict(data)))

    def get(self, field: str) -> Any:
        if field not in ALLOWLISTS[self.view]:
            raise ViewAccessError(f"{field!r} is not accessible in {self.view.value}")
        if field not in self._data:
            raise KeyError(field)
        return self._data[field]

    def export(self) -> dict[str, Any]:
        return dict(self._data)


def registry_json() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "default_policy": "deny",
        "views": {view.value: sorted(fields) for view, fields in ALLOWLISTS.items()},
        "hard_forbidden_outside_v3": sorted(HARD_FORBIDDEN),
    }
