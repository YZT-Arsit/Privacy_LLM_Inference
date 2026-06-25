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
    "NonlinearDesignNotWired",
    "real_path_executes",
    "real_path_execution_status",
    "assert_real_path_execution",
    "report_has_amulet_execution",
    "trusted_shortcut_tag_only",
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


# ---------------------------------------------------------------------------
# Real-path execution status (HONESTY GUARD)
# ---------------------------------------------------------------------------
#
# Whether the *real* Qwen folded-package / worker / probe / E3 / E9 path actually
# EXECUTES this design's nonlinear handling -- as opposed to merely tagging the
# design into report metadata + the attestation runtime hash.
#
# IMPORTANT: this mapping is intentionally kept OUT of the ``NONLINEAR_DESIGNS``
# records so it does NOT feed ``nonlinear_design_metadata_hash`` -- editing it
# must never invalidate already-built folded packages (their stored design hash
# stays stable). It is a runtime/honesty annotation, not part of the design
# identity.
#
#   * "current"          -> "trusted_boundary_inline": the folded worker + the
#       trusted boundary run the current trusted-island nonlinearity; this IS the
#       real, executed path (pllo.ops.nonlinear_islands).
#   * "trusted_shortcut" -> "lifted_on_accelerator": the Amulet-style lifted
#       backend is WIRED into the real folded worker (pllo.deployment.folded_nonlinear
#       -> pllo.nonlinear.amulet_backend). The MLP activation (SiLU/SwiGLU) is
#       lifted onto the untrusted accelerator and softmax/RMSNorm are migrated with
#       a trusted reduction shortcut; the worker stamps measured execution evidence
#       (amulet_lift_executed / lifted_nonlinear_ops_count / lift_k /
#       lifted_gpu_bytes) from NonlinearOpResult counters.
#
# NOTE: ``real_path_executes`` only states the design is CAPABLE of executing in
# the real path; whether a SPECIFIC report actually ran the lift is decided by
# ``report_has_amulet_execution`` (measured counters). A non-execution report (a
# folded-package BUILD, which only folds weights and never runs a nonlinearity)
# is therefore NOT treated as tag-only -- see ``_report_is_execution_bearing``.
_REAL_PATH_EXECUTION: Dict[str, str] = {
    "current": "trusted_boundary_inline",
    "trusted_shortcut": "lifted_on_accelerator",
}

# Report stages / signals that DO execute the model nonlinearity (and therefore
# must carry Amulet-lift evidence when tagged trusted_shortcut). A build / setup /
# estimate report is intentionally NOT here: it folds weights but never runs an
# activation, so it cannot and need not carry lift counters.
_EXECUTION_BEARING_STAGES = frozenset({
    "qwen7b_folded_package_prefill_probe",
    "qwen7b_folded_package_decode_probe",
    "qwen7b_folded_package_onestep_logits_probe",
    "qwen7b_folded_remote_package_decode",
    "remote_package_decode_scaling",
    "e3_remote_decode_scaling",
    "tee_gpu_protocol_demo",
    "e9_task_utility_benchmark",
    "e9_pairwise_utility_preservation",
    "e9_aggregate_utility_preservation",
    "e10_lora_utility_benchmark",
})


def _report_is_execution_bearing(report: Dict[str, Any]) -> bool:
    """True iff a report reflects an actual nonlinear-EXECUTION run (decode /
    prefill / utility), as opposed to a build/setup/estimate report."""
    if not isinstance(report, dict):
        return False
    if (report.get("stage") or "") in _EXECUTION_BEARING_STAGES:
        return True
    return (report.get("package_backed_decode") is True
            or report.get("package_backed_prefill") is True
            or report.get("tokens_exact_match") is not None
            or report.get("utility_preserved") is not None)


class NonlinearDesignNotWired(RuntimeError):
    """A paper-facing run selected a design whose nonlinear handling is not yet
    executed in the real Qwen path (tag-only)."""


def real_path_execution_status(name: str) -> str:
    """The real-path execution status string for a design (see above)."""
    return _REAL_PATH_EXECUTION.get(normalize_nonlinear_backend(name),
                                    "prototype_only")


def real_path_executes(name: str) -> bool:
    """True iff the real Qwen folded path actually executes this design's
    nonlinear handling (False for tag-only prototypes like trusted_shortcut)."""
    return real_path_execution_status(name) != "prototype_only"


def assert_real_path_execution(name: str, *, dry_run: bool = False,
                               allow_unwired: bool = False) -> str:
    """Guard for paper-facing runs: raise :class:`NonlinearDesignNotWired` if a
    real (non-dry-run) run selects a design not executed in the real path.

    ``dry_run`` runs are always allowed (they are clearly labeled prototypes);
    ``allow_unwired=True`` is an explicit opt-in that lets a PROTOTYPE run proceed
    (it can never produce paper-facing evidence -- the claim validator / gate
    independently reject tag-only trusted_shortcut)."""
    canon = normalize_nonlinear_backend(name)
    if dry_run or allow_unwired or real_path_executes(canon):
        return canon
    raise NonlinearDesignNotWired(
        "nonlinear design %r is a correctness PROTOTYPE (status=%s) that is NOT "
        "wired into the real Qwen folded-package/worker path -- it would only be "
        "TAGGED, not executed (the worker runs the 'current' trusted-island "
        "nonlinearity). A paper-facing run must not select it. Wire the "
        "amulet_migrated op backend into the real path (op_backend_for_design + "
        "make_nonlinear_backend, stamping amulet_lift_executed / "
        "lifted_nonlinear_ops_count / lift_k / lifted_gpu_bytes), or pass "
        "--allow-unwired-nonlinear for a clearly non-paper-facing prototype run, "
        "or use --dry-run." % (canon, real_path_execution_status(canon)))


def report_has_amulet_execution(report: Dict[str, Any]) -> bool:
    """True iff a report carries genuine runtime evidence that the Amulet-style
    lifted nonlinear backend actually executed (not just a design tag)."""
    if not isinstance(report, dict):
        return False
    return (report.get("nonlinear_op_backend") == "amulet_migrated"
            and (report.get("amulet_lift_executed") is True
                 or report.get("amulet_backend_used") is True)
            and (report.get("lifted_nonlinear_ops_count") or 0) > 0
            and (report.get("lift_k") or 0) >= 2
            and (report.get("lifted_gpu_bytes") or 0) > 0)


def trusted_shortcut_tag_only(report: Dict[str, Any]) -> bool:
    """True iff an EXECUTION-bearing report is TAGGED trusted_shortcut but lacks
    real Amulet-lift execution evidence (i.e. tag-only -- it ran the 'current'
    path under a trusted_shortcut tag, or the lift was never wired in).

    A build / setup / estimate report (which never runs a nonlinearity) is NOT
    flagged: it legitimately carries no lift counters."""
    if not isinstance(report, dict):
        return False
    nb = report.get("nonlinear_backend") or report.get("nonlinear_design_name")
    if not nb:
        return False
    try:
        canon = normalize_nonlinear_backend(nb)
    except UnknownNonlinearBackend:
        return False
    if canon != "trusted_shortcut":
        return False
    if not _report_is_execution_bearing(report):
        return False
    return not report_has_amulet_execution(report)


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
    executed = real_path_executes(canon)
    return {
        "nonlinear_backend": canon,
        "nonlinear_design_name": canon,
        "nonlinear_design_label": rec["design_label"],
        "nonlinear_design_version": rec["version"],
        "nonlinear_design_registry_version": NONLINEAR_DESIGN_REGISTRY_VERSION,
        "nonlinear_design_metadata_hash": nonlinear_design_metadata_hash(canon),
        "nonlinear_op_backend": rec["op_backend"],
        # HONEST execution annotation: whether the real Qwen path actually runs
        # this design's nonlinearity, vs. only tagging it. A wired real path that
        # truly ran the lift must OVERRIDE amulet_lift_executed (and stamp
        # lifted_nonlinear_ops_count / lift_k / lifted_gpu_bytes) AFTER this.
        "nonlinear_real_path_execution": real_path_execution_status(canon),
        "nonlinear_real_path_executed": executed,
        "amulet_lift_executed": False,
        "nonlinear_execution_status": (
            # capability-level default stamp; an execution-bearing run OVERRIDES
            # this with measured counters (folded_nonlinear runner). For design B
            # the design fields alone are NOT execution evidence -- the worker
            # must stamp amulet_lift_executed / lifted_* afterward.
            real_path_execution_status(canon) if executed
            else "tag_only_prototype_not_wired"),
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
