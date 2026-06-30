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
    "report_has_right_multiply_execution",
    "report_has_secure_right_multiply_execution",
    "report_has_real_nonlinear_execution",
    "report_nonlinear_trusted_calls_clean",
    "nonlinear_tag_only",
    "trusted_shortcut_tag_only",
    "PAPER_FACING_DESIGNS",
    "PAPER_FACING_DEFAULT_DESIGNS",
    "NON_PAPER_FACING_DESIGNS",
    "NonPaperFacingDesign",
    "is_paper_facing_design",
    "assert_paper_facing_design",
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
    "A_rightmul": {
        "name": "A_rightmul",
        "design_label": "design_A_rightmul",
        "version": "1.0",
        # op-backend key in pllo.nonlinear.registry
        "op_backend": "compatible_right_multiply",
        "aliases": [
            "a_rightmul", "rightmul", "right_multiply", "right_mul",
            "compatible_right_multiply", "compatible_nonlinear_islands",
            "compatible_right_multiply_islands", "ours_compatible_nonlinear_islands",
        ],
        "description": (
            "A_rightmul design: every transformer nonlinear island (SiLU/SwiGLU "
            "MLP, attention softmax, RMSNorm/LayerNorm core) is evaluated "
            "directly on the untrusted accelerator over the compatible "
            "right-multiply / permutation-masked state, with NO trusted-boundary "
            "crossing for the nonlinearity. The TEE is entered once (input mask) "
            "and exited once (logits recovery); no nonlinear op runs in the TEE."),
        "trusted_boundary_role": (
            "input embedding + mask (once) and final logits recovery + sampling "
            "(once); holds NO nonlinear reduction shortcut -- zero nonlinear "
            "crossings."),
        "gpu_worker_role": (
            "evaluates every nonlinear island in place on the masked state using "
            "compatible right-multiply / permutation masks (signed-permutation "
            "residual mask, per-head Q/K/V masks with Q~K~^T=QK^T, SwiGLU channel "
            "permutation); output stays in the masked basis."),
        "supports_gelu": True,
        "supports_silu": True,
        "supports_mlp": True,
        "correctness_expectation": "exact_vs_float64_reference",
        "security_status": "claimed_under_compatible_mask_assumption",
        "security_claim_status": "claimed_under_assumption",
        "security_notes": (
            "Compatible right-multiply security is CLAIMED UNDER the compatible-"
            "mask assumption: the residual/RMSNorm/LayerNorm mask is a signed "
            "permutation (orthogonal monomial), attention uses Q/K masks with "
            "Q~K~^T == QK^T, and GELU/SiLU/SwiGLU use a shared channel "
            "permutation. These conditions are CHECKED (raise on violation); the "
            "claim holds under the assumption that the adversary only observes "
            "the masked state. NOT a completed formal proof; no arbitrary dense "
            "mask is claimed to commute with the nonlinear cores."),
        "expected_extra_boundary_calls": 0,
        "expected_extra_trusted_bytes": 0,
        "nonlinear_masking_mode": "compatible_right_multiply_or_permutation",
        "limitations": [
            "claim holds only under the compatible-mask assumption (checked)",
            "requires compatible masks (signed-permutation residual, per-head "
            "Q/K/V, SwiGLU channel permutation) -- arbitrary dense masks do not "
            "commute with the nonlinear cores (raise on violation)",
        ],
    },
    "amulet_secure_R": {
        "name": "amulet_secure_R",
        "design_label": "design_amulet_secure_R",
        "version": "1.0",
        "op_backend": "amulet_secure_R",
        "aliases": [
            "amulet_secure_r", "secure_r", "secure_rightmul",
            "secure_right_multiply", "amulet_secure", "amulet_secure_r_nonlinear",
            "secure_amulet", "design_b_secure",
        ],
        "description": (
            "Amulet-like secure-R design: GELU/SiLU are evaluated on the "
            "untrusted accelerator via a dense single-one Kronecker lift "
            "(R_bar = R1 R2 R3, exactly one secret unit entry, no zero decoys, "
            "no visible one-hot selector) with secret shuffles; softmax / "
            "RMSNorm / LayerNorm run directly over the masked state with NO "
            "trusted reduction shortcut. Single TEE entry/exit; zero online "
            "trusted nonlinear crossings."),
        "trusted_boundary_role": (
            "input embedding + mask (once) and final logits recovery + sampling "
            "(once); holds NO nonlinear reduction shortcut (unlike the legacy "
            "trusted_shortcut design) -- zero online nonlinear crossings."),
        "gpu_worker_role": (
            "evaluates GELU/SiLU on a shuffled dense secure-R lift (the only "
            "GPU-visible activation artifact) and softmax/RMSNorm/LayerNorm "
            "directly over the masked state; the squeeze is folded with secret "
            "permutations so the valid channel is not directly observable."),
        "supports_gelu": True,
        "supports_silu": True,
        "supports_mlp": True,
        "correctness_expectation": "exact_vs_float64_reference",
        "security_status": "claimed_under_secure_R_assumption",
        "security_claim_status": "claimed_under_assumption",
        "security_notes": (
            "Secure-R security is CLAIMED UNDER the secure-R assumption: the "
            "secret coordinate (a,b) and shuffles are not recoverable from the "
            "GPU-visible dense lift; checkable conditions (no zero decoys, dense "
            "single-one R, no raw one-hot/selector tensor, secret coordinate "
            "never reported, trusted_calls == 0) are ASSERTED. NOT a completed "
            "formal proof."),
        "expected_extra_boundary_calls": 0,
        "expected_extra_trusted_bytes": 0,
        "nonlinear_masking_mode": "amulet_secure_right_multiply",
        "limitations": [
            "claim holds only under the secure-R assumption (checked conditions)",
            "dense Kronecker lift costs ~k^2 x activation bytes on the "
            "accelerator (no trusted shortcut, no zero decoys)",
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
#   * "A_rightmul"        -> "right_multiply_on_accelerator": every nonlinear
#       island runs in place on the masked state on the untrusted accelerator
#       (compatible right-multiply / permutation masks); the folded worker stamps
#       measured evidence (right_multiply_nonlinear_executed /
#       right_multiply_nonlinear_ops_count, trusted_nonlinear_ops_count == 0).
_REAL_PATH_EXECUTION: Dict[str, str] = {
    "current": "trusted_boundary_inline",
    "trusted_shortcut": "lifted_on_accelerator",
    "A_rightmul": "right_multiply_on_accelerator",
    "amulet_secure_R": "secure_right_multiply_on_accelerator",
}

# Paper-facing nonlinear designs. The legacy ``current`` (trusted-island) and
# ``trusted_shortcut`` (per-op trusted reduction shortcut) designs are kept only
# as debug/local baselines and are REJECTED by paper-facing runs / the gate /
# the claim validator (see assert_paper_facing_design).
PAPER_FACING_DESIGNS: tuple = ("A_rightmul", "amulet_secure_R")
PAPER_FACING_DEFAULT_DESIGNS = "A_rightmul,amulet_secure_R"
# Designs that may NEVER back a paper-facing claim (single TEE entry/exit is not
# met: current evaluates nonlinear in the trusted island; trusted_shortcut keeps
# a per-op trusted reduction shortcut).
NON_PAPER_FACING_DESIGNS: tuple = ("current", "trusted_shortcut")


class NonPaperFacingDesign(RuntimeError):
    """A paper-facing run selected a legacy design (current / trusted_shortcut)
    that does not meet the single-TEE-entry / zero-trusted-nonlinear contract."""


def is_paper_facing_design(name: str) -> bool:
    try:
        return normalize_nonlinear_backend(name) in PAPER_FACING_DESIGNS
    except UnknownNonlinearBackend:
        return False


def assert_paper_facing_design(name: str) -> str:
    """Return the canonical name iff it is a paper-facing design; else raise.

    Use in any script under ``--paper-facing`` / ``--require-real``."""
    canon = normalize_nonlinear_backend(name)
    if canon not in PAPER_FACING_DESIGNS:
        raise NonPaperFacingDesign(
            "nonlinear design %r is NOT paper-facing (legacy debug/local "
            "baseline). Paper-facing runs must use one of %s. 'current' "
            "evaluates the nonlinearity in the trusted island and "
            "'trusted_shortcut' keeps a per-op trusted reduction shortcut -- "
            "both violate the single-TEE-entry / zero-trusted-nonlinear "
            "contract." % (canon, list(PAPER_FACING_DESIGNS)))
    return canon

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


def report_has_right_multiply_execution(report: Dict[str, Any]) -> bool:
    """True iff a report carries genuine runtime evidence that the A_rightmul
    compatible right-multiply nonlinear backend actually executed on the
    accelerator (not just a design tag): the op backend is
    ``compatible_right_multiply``, the right-multiply path executed at least one
    op, and NO nonlinear work crossed the trusted boundary."""
    if not isinstance(report, dict):
        return False
    return (report.get("nonlinear_op_backend") == "compatible_right_multiply"
            and report.get("right_multiply_nonlinear_executed") is True
            and (report.get("right_multiply_nonlinear_ops_count") or 0) > 0
            and (report.get("trusted_nonlinear_ops_count") or 0) == 0
            and (report.get("nonlinear_trusted_calls") or 0) == 0)


def report_has_secure_right_multiply_execution(report: Dict[str, Any]) -> bool:
    """True iff a report carries genuine runtime evidence that ``amulet_secure_R``
    actually executed on the accelerator: op backend ``amulet_secure_R``, the
    secure-R activation executed at least once, NO trusted nonlinear crossing,
    and the secure conditions (no zero decoys, selector not visible) hold."""
    if not isinstance(report, dict):
        return False
    return (report.get("nonlinear_op_backend") == "amulet_secure_R"
            and report.get("secure_right_multiply_executed") is True
            and (report.get("secure_right_multiply_ops_count") or 0) > 0
            and (report.get("trusted_nonlinear_ops_count") or 0) == 0
            and (report.get("nonlinear_trusted_calls") or 0) == 0
            and report.get("secure_R_enabled") is True
            and report.get("zero_decoys") is False
            and report.get("selector_visible_to_gpu") is False)


def report_has_real_nonlinear_execution(report: Dict[str, Any]) -> bool:
    """True iff an execution-bearing report carries genuine measured evidence
    that its tagged nonlinear design actually executed (design-agnostic).

    ``current`` always executes (trusted-island inline). ``trusted_shortcut``
    must show Amulet-lift evidence; ``A_rightmul`` must show right-multiply
    evidence; ``amulet_secure_R`` must show secure right-multiply evidence. Used
    by the non-tag-only validation."""
    if not isinstance(report, dict):
        return False
    nb = report.get("nonlinear_backend") or report.get("nonlinear_design_name")
    try:
        canon = normalize_nonlinear_backend(nb) if nb else None
    except UnknownNonlinearBackend:
        return False
    if canon == "trusted_shortcut":
        return report_has_amulet_execution(report)
    if canon == "A_rightmul":
        return report_has_right_multiply_execution(report)
    if canon == "amulet_secure_R":
        return report_has_secure_right_multiply_execution(report)
    if canon == "current":
        return True
    return False


def report_nonlinear_trusted_calls_clean(report: Dict[str, Any]) -> bool:
    """True iff an execution-bearing report has NO trusted nonlinear crossings.

    Paper-facing designs require zero trusted nonlinear ops (single TEE
    entry/exit). A report with ``nonlinear_trusted_calls > 0`` or
    ``trusted_nonlinear_ops_count > 0`` is rejected."""
    if not isinstance(report, dict):
        return True
    if not _report_is_execution_bearing(report):
        return True
    return ((report.get("nonlinear_trusted_calls") or 0) == 0
            and (report.get("trusted_nonlinear_ops_count") or 0) == 0)


def nonlinear_tag_only(report: Dict[str, Any]) -> bool:
    """True iff an EXECUTION-bearing report is TAGGED with a migrated design
    (``trusted_shortcut`` or ``A_rightmul``) but lacks the corresponding measured
    execution evidence (i.e. tag-only). Build/setup/estimate reports are never
    flagged (they run no nonlinearity)."""
    if not isinstance(report, dict):
        return False
    nb = report.get("nonlinear_backend") or report.get("nonlinear_design_name")
    if not nb:
        return False
    try:
        canon = normalize_nonlinear_backend(nb)
    except UnknownNonlinearBackend:
        return False
    if canon not in ("trusted_shortcut", "A_rightmul", "amulet_secure_R"):
        return False
    if not _report_is_execution_bearing(report):
        return False
    return not report_has_real_nonlinear_execution(report)


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
    fields = {
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
    if canon == "A_rightmul":
        # A_rightmul capability stamp. ``right_multiply_nonlinear_executed`` stays
        # False here (a BUILD folds weights and runs no nonlinearity); an
        # execution-bearing run (probe/decode/demo/ifeval) OVERRIDES it with the
        # measured runner evidence (trusted_nonlinear_ops_count == 0).
        fields.update({
            "right_multiply_nonlinear_executed": False,
            "right_multiply_nonlinear_ops_count": 0,
            "trusted_nonlinear_ops_count": 0,
            "nonlinear_trusted_calls": 0,
            "nonlinear_masking_mode": "compatible_right_multiply_or_permutation",
            "nonlinear_single_tee_entry_exit": True,
            "linear_boundary_pad": True,
        })
    if canon == "amulet_secure_R":
        # amulet_secure_R capability stamp; execution-bearing runs OVERRIDE with
        # measured runner evidence (secure_right_multiply_executed=True etc.).
        fields.update({
            "secure_right_multiply_executed": False,
            "secure_right_multiply_ops_count": 0,
            "trusted_nonlinear_ops_count": 0,
            "nonlinear_trusted_calls": 0,
            "secure_R_enabled": True,
            "zero_decoys": False,
            "selector_visible_to_gpu": False,
            "valid_channel_observable": False,
            "nonlinear_masking_mode": "amulet_secure_right_multiply",
            "nonlinear_single_tee_entry_exit": True,
            "linear_boundary_pad": True,
        })
    return fields


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
                               default: str = PAPER_FACING_DEFAULT_DESIGNS,
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
