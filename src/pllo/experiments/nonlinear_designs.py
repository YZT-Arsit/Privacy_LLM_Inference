"""First-class registry of nonlinear-layer *designs* for the paper matrix.

The folded Qwen pipeline can handle transformer nonlinear "islands"
(GELU/SiLU MLP, Softmax, LayerNorm/RMSNorm, and the trusted-softmax shortcut)
under more than one design. The advisor requires that the *same* full
experiment suite run under BOTH designs so either can be chosen later with
complete results. This module makes the nonlinear design a first-class
experimental dimension: a stable registry, name normalization (with CLI
aliases so the schemes can be renamed cleanly later), a per-design metadata
hash that binds into the attestation runtime hash, and the report fields every
paper-facing report must carry.

Two designs:

* ``current`` (design A / baseline) -- the nonlinearity is evaluated inside the
  *trusted boundary* (trusted island / trusted shortcut). This is the design the
  repo has validated end-to-end (no-LoRA TDX-lite/attested decode).
* ``trusted_shortcut`` (design B / alternative) -- the bulk of the nonlinearity
  is *migrated* off the trusted boundary onto the untrusted accelerator via an
  Amulet-style lifted/transformed view, keeping only a small trusted reduction
  shortcut. Its security is **not formally claimed** (under discussion with the
  advisor); correctness is exact.

This *design* registry sits above the lower-level op-backend registry in
:mod:`pllo.nonlinear.registry` (keys ``current`` / ``amulet_migrated``). Use
:func:`op_backend_for_design` to map a design name to the op-backend key.

stdlib only (json / hashlib). No torch import here -- safe to import from any
script, including trusted-boundary code that is measured for attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, List, Optional

__all__ = [
    "NONLINEAR_DESIGN_REGISTRY_VERSION",
    "NONLINEAR_DESIGNS",
    "DEFAULT_NONLINEAR_BACKEND",
    "UnknownNonlinearBackend",
    "normalize_nonlinear_backend",
    "list_nonlinear_backends",
    "nonlinear_backend_metadata",
    "assert_supported_nonlinear_backend",
    "op_backend_for_design",
    "nonlinear_design_metadata_hash",
    "nonlinear_design_report_fields",
    "add_nonlinear_backend_arg",
    "add_nonlinear_backends_arg",
    "parse_nonlinear_backends",
]

# Bump when the *meaning* of a design record changes in a way that should change
# the metadata hash / attestation binding intentionally.
NONLINEAR_DESIGN_REGISTRY_VERSION = "1.0"

DEFAULT_NONLINEAR_BACKEND = "current"


class UnknownNonlinearBackend(ValueError):
    """Raised when a nonlinear backend/design name is not recognized."""


# Each record is the public design contract. Keep it JSON-serializable and
# stable: the canonical form is hashed into the attestation runtime hash so a
# design change cannot silently reuse another design's attestation evidence.
NONLINEAR_DESIGNS: Dict[str, Dict[str, Any]] = {
    "current": {
        "name": "current",
        "design_label": "design_A",
        "version": "1.0",
        # op-backend key in pllo.nonlinear.registry
        "op_backend": "current",
        "aliases": [
            "current", "design_a", "baseline", "baseline_nonlinear",
            "trusted_island", "trusted_boundary_nonlinear", "a",
        ],
        "description": (
            "Baseline design: nonlinear islands (GELU/SiLU MLP, Softmax, "
            "LayerNorm/RMSNorm) are evaluated inside the trusted boundary; the "
            "untrusted accelerator only sees masked/folded linear payloads."),
        "trusted_boundary_role": (
            "evaluates every nonlinear island (unmask -> nonlinearity -> "
            "remask); holds the trusted-softmax shortcut reduction."),
        "gpu_worker_role": (
            "performs masked/folded linear matmuls only; never evaluates the "
            "model nonlinearity in plaintext."),
        "supports_gelu": True,
        "supports_silu": True,
        "supports_mlp": True,
        "correctness_expectation": "exact_vs_float64_reference",
        "security_status": "trusted_boundary",
        "security_claim_status": "established",
        "security_notes": (
            "Nonlinearity never leaves the trusted boundary in plaintext; this "
            "is the design validated end-to-end (no-LoRA TDX-lite/attested)."),
        "expected_extra_boundary_calls": 0,
        "expected_extra_trusted_bytes": 0,
        "limitations": [
            "more trusted-boundary compute per nonlinear island than design B",
            "trusted boundary must host the activation functions",
        ],
    },
    "trusted_shortcut": {
        "name": "trusted_shortcut",
        "design_label": "design_B",
        "version": "1.0",
        # bridges to the existing Amulet-migrated op backend
        "op_backend": "amulet_migrated",
        "aliases": [
            "trusted_shortcut", "design_b", "alternative", "amulet_migrated",
            "amulet", "tee_shortcut_nonlinear", "tee_shortcut", "b",
        ],
        "description": (
            "Alternative design: the bulk of the nonlinearity is migrated onto "
            "the untrusted accelerator via an Amulet-style lifted view; only a "
            "small trusted reduction shortcut (softmax/norm denominators) stays "
            "in the trusted boundary."),
        "trusted_boundary_role": (
            "holds only a small reduction shortcut (e.g. softmax/norm "
            "denominators); the activation evaluation is lifted to the GPU."),
        "gpu_worker_role": (
            "evaluates the lifted nonlinear view over masked activations; "
            "receives extra lifted payload (selector-lift) per nonlinear op."),
        "supports_gelu": True,
        "supports_silu": True,
        "supports_mlp": True,
        "correctness_expectation": "exact_vs_float64_reference",
        "security_status": "not_formally_claimed",
        "security_claim_status": "under_discussion",
        "security_notes": (
            "Amulet migration security is NOT formally claimed (selector-leak "
            "caveat, under discussion with advisor). Correctness is exact; the "
            "trade is WHERE the nonlinear work runs + latency."),
        "expected_extra_boundary_calls": 0,
        "expected_extra_trusted_bytes": 0,
        "limitations": [
            "security boundary not formally proven (selector-leakage caveat)",
            "extra GPU payload per nonlinear island (lifted view)",
            "must be re-attested separately; cannot reuse design A evidence",
        ],
    },
}

# alias -> canonical name (built once; case/space/hyphen-insensitive lookups go
# through _key()).
_ALIAS_TO_NAME: Dict[str, str] = {}
for _canon, _rec in NONLINEAR_DESIGNS.items():
    for _a in _rec["aliases"]:
        _ALIAS_TO_NAME[_a] = _canon
    _ALIAS_TO_NAME[_canon] = _canon


def _key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_nonlinear_backend(name: str) -> str:
    """Map any alias/case/hyphenation to a canonical design name.

    ``amulet_migrated`` / ``tee_shortcut_nonlinear`` / ``design_b`` -> the
    canonical ``trusted_shortcut``; ``baseline_nonlinear`` / ``design_a`` ->
    ``current``. Raises :class:`UnknownNonlinearBackend` otherwise."""
    if name is None:
        raise UnknownNonlinearBackend("nonlinear backend is None")
    canon = _ALIAS_TO_NAME.get(_key(name))
    if canon is None:
        raise UnknownNonlinearBackend(
            "unknown nonlinear backend %r; expected one of %s (aliases: %s)"
            % (name, list(NONLINEAR_DESIGNS), sorted(_ALIAS_TO_NAME)))
    return canon


def list_nonlinear_backends() -> List[str]:
    """Canonical design names, registry order."""
    return list(NONLINEAR_DESIGNS)


def nonlinear_backend_metadata(name: str) -> Dict[str, Any]:
    """Return a deep-ish copy of the design record (normalizing the name)."""
    rec = NONLINEAR_DESIGNS[normalize_nonlinear_backend(name)]
    return json.loads(json.dumps(rec))            # cheap deep copy, JSON-safe


def assert_supported_nonlinear_backend(name: str) -> str:
    """Validate + return the canonical name (raise UnknownNonlinearBackend)."""
    return normalize_nonlinear_backend(name)


def op_backend_for_design(name: str) -> str:
    """Map a design name to the lower-level op-backend key
    (:mod:`pllo.nonlinear.registry`)."""
    return NONLINEAR_DESIGNS[normalize_nonlinear_backend(name)]["op_backend"]


def _canonical_metadata_bytes(name: str) -> bytes:
    rec = NONLINEAR_DESIGNS[normalize_nonlinear_backend(name)]
    payload = {"registry_version": NONLINEAR_DESIGN_REGISTRY_VERSION,
               "design": rec}
    return json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def nonlinear_design_metadata_hash(name: str) -> str:
    """SHA-256 over the canonical design record (+ registry version).

    Binds a specific nonlinear design into manifests and the attestation runtime
    hash, so design A evidence cannot be reused for design B."""
    return hashlib.sha256(_canonical_metadata_bytes(name)).hexdigest()


def nonlinear_design_report_fields(name: str) -> Dict[str, Any]:
    """The fields EVERY paper-facing report must carry for the nonlinear design.

    Use ``report.update(nonlinear_design_report_fields(args.nonlinear_backend))``
    after normalizing. Returns a flat, JSON-safe dict (no torch)."""
    canon = normalize_nonlinear_backend(name)
    rec = NONLINEAR_DESIGNS[canon]
    return {
        "nonlinear_backend": canon,
        "nonlinear_design_name": canon,
        "nonlinear_design_label": rec["design_label"],
        "nonlinear_design_version": rec["version"],
        "nonlinear_design_registry_version": NONLINEAR_DESIGN_REGISTRY_VERSION,
        "nonlinear_design_metadata_hash": nonlinear_design_metadata_hash(canon),
        "nonlinear_design_metadata_summary": {
            "op_backend": rec["op_backend"],
            "security_status": rec["security_status"],
            "security_claim_status": rec["security_claim_status"],
            "trusted_boundary_role": rec["trusted_boundary_role"],
            "gpu_worker_role": rec["gpu_worker_role"],
        },
        "nonlinear_design_limitations": list(rec["limitations"]),
    }


# ---------------------------------------------------------------------------
# argparse helpers (so every script wires the flag the same way)
# ---------------------------------------------------------------------------


def add_nonlinear_backend_arg(parser: argparse.ArgumentParser,
                              default: str = DEFAULT_NONLINEAR_BACKEND,
                              required: bool = False) -> None:
    """Add ``--nonlinear-backend`` to a parser (single design)."""
    parser.add_argument(
        "--nonlinear-backend", default=default, required=required,
        help=("nonlinear-layer design: %s (aliases accepted, e.g. "
              "amulet_migrated/tee_shortcut_nonlinear)"
              % ", ".join(list_nonlinear_backends())))


def add_nonlinear_backends_arg(parser: argparse.ArgumentParser,
                               default: str = "current,trusted_shortcut",
                               flag: str = "--nonlinear-backends") -> None:
    """Add a comma-separated ``--nonlinear-backends`` matrix flag."""
    parser.add_argument(
        flag, default=default,
        help=("comma-separated nonlinear designs for the matrix; default %r "
              "(canonical: %s)" % (default, ", ".join(list_nonlinear_backends()))))


def parse_nonlinear_backends(value: str) -> List[str]:
    """Parse a comma-separated backend list -> ordered, de-duplicated canonical
    names. Raises :class:`UnknownNonlinearBackend` on any unknown entry."""
    out: List[str] = []
    for tok in str(value).split(","):
        tok = tok.strip()
        if not tok:
            continue
        canon = normalize_nonlinear_backend(tok)
        if canon not in out:
            out.append(canon)
    if not out:
        raise UnknownNonlinearBackend("no nonlinear backends parsed from %r"
                                      % value)
    return out
