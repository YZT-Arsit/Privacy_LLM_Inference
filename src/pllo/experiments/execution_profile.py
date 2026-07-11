"""Explicit *execution profiles* that eliminate the default-nonlinear footgun.

The repo's raw default nonlinear design is ``current`` (the nonlinearity is
evaluated inside the trusted boundary; with no real TEE on the inference
accelerator this materialises the plaintext activation at every island). That is
fine as a debug baseline but MUST NOT silently back a paper-facing security
claim. Rather than only flipping ``DEFAULT_NONLINEAR_MODE``, this module makes the
*execution profile* a first-class, explicit, fail-closed object.

Three profiles:

* ``paper_safe`` -- A_rightmul, masked residual transport + masked inter-block
  handoff required, zero trusted nonlinear crossings, zero plaintext-hidden
  materialisations, ``current``/``trusted_shortcut`` forbidden, no silent
  fallback. ``paper_facing = True``; ``security_claim_allowed`` iff every gate
  passes. This is the DEFAULT (a missing profile resolves to ``paper_safe``).
* ``legacy_debug`` -- ``current`` allowed, ``paper_facing = False``,
  ``security_claim_allowed = False``, plaintext nonlinear materialisation is
  explicitly RECORDED (never hidden), requires explicit opt-in.
* ``experimental`` -- explicit opt-in, no security claim by default.

Design rules enforced here:

* AAAI runner + public runtime entrypoint default to ``paper_safe``.
* ``current`` / ``trusted_shortcut`` never pass the paper-facing validator.
* ``profile`` + ``nonlinear_backend`` + inter-block mode are written into the
  results manifest.
* A mode mismatch is fail-closed (raises), never a silent runtime downgrade.

stdlib only (no torch) so it is safe to import from trusted-boundary /
attestation code that is measured for the runtime hash. Backend normalization is
reused from :mod:`pllo.experiments.nonlinear_designs` (also stdlib-only).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Optional, Tuple

from pllo.experiments.nonlinear_designs import (
    NON_PAPER_FACING_DESIGNS,
    PAPER_FACING_DESIGNS,
    UnknownNonlinearBackend,
    normalize_nonlinear_backend,
)

__all__ = [
    "EXECUTION_PROFILE_SCHEMA_VERSION",
    "ExecutionProfile",
    "ExecutionProfileViolation",
    "UnknownExecutionProfile",
    "PROFILE_REGISTRY",
    "DEFAULT_PROFILE",
    "get_profile",
    "resolve_profile",
    "validate_run",
    "security_claim_allowed",
    "profile_manifest_fields",
    "execution_profile_metadata_hash",
]

# Bump when the *meaning* of a profile changes in a way that should change the
# attestation-bound metadata hash intentionally.
EXECUTION_PROFILE_SCHEMA_VERSION = "1.0"

DEFAULT_PROFILE = "paper_safe"


class UnknownExecutionProfile(ValueError):
    """A profile name that is not in the registry."""


class ExecutionProfileViolation(RuntimeError):
    """A run's observed execution facts violate its declared profile.

    Raised (fail-closed) instead of silently downgrading. Carries the list of
    concrete violations so the caller can log/serialise them."""

    def __init__(self, profile: str, violations: Tuple[str, ...]):
        self.profile = profile
        self.violations = tuple(violations)
        super().__init__(
            "execution profile %r violated (fail-closed): %s"
            % (profile, "; ".join(violations)))


@dataclass(frozen=True)
class ExecutionProfile:
    """A frozen, explicit description of how a run is allowed to execute.

    ``*_required`` gates assert a POSITIVE property must be observed true;
    ``max_*`` gates cap a counter (0 == "must never happen"); ``allowed_backends``
    / ``forbidden_backends`` gate the nonlinear design.
    """

    name: str
    default_nonlinear_backend: str
    allowed_backends: Tuple[str, ...]
    forbidden_backends: Tuple[str, ...]
    masked_residual_transport_required: bool
    masked_inter_block_handoff_required: bool
    max_nonlinear_trusted_calls: int
    max_trusted_nonlinear_ops: int
    max_plaintext_hidden_materializations: int
    silent_fallback_forbidden: bool
    paper_facing: bool
    # Whether this profile is even *eligible* to back a security claim (still
    # requires all gates to pass at validate time -- see security_claim_allowed).
    security_claim_eligible: bool
    requires_explicit_opt_in: bool
    # legacy_debug records (does not forbid) plaintext nonlinear materialisation.
    record_plaintext_materialization: bool
    note: str = ""

    # -- backend gating -----------------------------------------------------
    def backend_allowed(self, backend: str) -> bool:
        try:
            canon = normalize_nonlinear_backend(backend)
        except UnknownNonlinearBackend:
            return False
        if canon in self.forbidden_backends:
            return False
        return canon in self.allowed_backends

    def canonical_backend(self, backend: Optional[str]) -> str:
        """Normalize + gate a requested backend; fall back to the profile
        default only when nothing was requested. A REQUESTED-but-forbidden
        backend is fail-closed (never silently replaced by the default)."""
        if backend is None or backend == "":
            return self.default_nonlinear_backend
        canon = normalize_nonlinear_backend(backend)
        if not self.backend_allowed(canon):
            raise ExecutionProfileViolation(
                self.name,
                ("nonlinear_backend %r is forbidden under profile %r "
                 "(allowed=%s, forbidden=%s)"
                 % (canon, self.name, list(self.allowed_backends),
                    list(self.forbidden_backends)),))
        return canon


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROFILE_REGISTRY: Dict[str, ExecutionProfile] = {
    "paper_safe": ExecutionProfile(
        name="paper_safe",
        default_nonlinear_backend="A_rightmul",
        allowed_backends=PAPER_FACING_DESIGNS,          # A_rightmul, amulet_secure_R
        forbidden_backends=NON_PAPER_FACING_DESIGNS,    # current, trusted_shortcut
        masked_residual_transport_required=True,
        masked_inter_block_handoff_required=True,
        max_nonlinear_trusted_calls=0,
        max_trusted_nonlinear_ops=0,
        max_plaintext_hidden_materializations=0,
        silent_fallback_forbidden=True,
        paper_facing=True,
        security_claim_eligible=True,
        requires_explicit_opt_in=False,                 # this is the safe default
        record_plaintext_materialization=False,
        note=("A_rightmul only; masked inter-block domain; zero trusted "
              "nonlinear crossings; zero plaintext-hidden materialisations; "
              "no silent fallback. Security claim allowed iff all gates pass."),
    ),
    "legacy_debug": ExecutionProfile(
        name="legacy_debug",
        default_nonlinear_backend="current",
        allowed_backends=("current", "trusted_shortcut") + PAPER_FACING_DESIGNS,
        forbidden_backends=(),
        masked_residual_transport_required=False,
        masked_inter_block_handoff_required=False,
        max_nonlinear_trusted_calls=10 ** 12,           # effectively unbounded
        max_trusted_nonlinear_ops=10 ** 12,
        max_plaintext_hidden_materializations=10 ** 12,
        silent_fallback_forbidden=False,
        paper_facing=False,
        security_claim_eligible=False,                  # never backs a claim
        requires_explicit_opt_in=True,
        record_plaintext_materialization=True,          # explicitly recorded
        note=("Legacy/local baseline. 'current' allowed; NOT paper-facing; "
              "security_claim_allowed is always False; plaintext nonlinear "
              "materialisation is explicitly recorded. Requires explicit "
              "opt-in."),
    ),
    "experimental": ExecutionProfile(
        name="experimental",
        default_nonlinear_backend="A_rightmul",
        allowed_backends=("current", "trusted_shortcut") + PAPER_FACING_DESIGNS,
        forbidden_backends=(),
        masked_residual_transport_required=False,
        masked_inter_block_handoff_required=False,
        max_nonlinear_trusted_calls=10 ** 12,
        max_trusted_nonlinear_ops=10 ** 12,
        max_plaintext_hidden_materializations=10 ** 12,
        silent_fallback_forbidden=True,
        paper_facing=False,
        security_claim_eligible=False,                  # no claim by default
        requires_explicit_opt_in=True,
        record_plaintext_materialization=True,
        note=("Explicit opt-in research profile; no security claim by default."),
    ),
}


def get_profile(name: str) -> ExecutionProfile:
    if name not in PROFILE_REGISTRY:
        raise UnknownExecutionProfile(
            "unknown execution_profile %r; known: %s"
            % (name, sorted(PROFILE_REGISTRY)))
    return PROFILE_REGISTRY[name]


def resolve_profile(name: Optional[str]) -> ExecutionProfile:
    """A missing/blank profile resolves to the safe default (``paper_safe``).

    A *present but unknown* profile is fail-closed (raises) -- we never silently
    treat a typo'd profile as the default."""
    if name is None or name == "":
        return PROFILE_REGISTRY[DEFAULT_PROFILE]
    return get_profile(name)


# ---------------------------------------------------------------------------
# Validation (fail-closed) + claim gating
# ---------------------------------------------------------------------------

_OBSERVED_KEYS = (
    "nonlinear_backend",
    "nonlinear_trusted_calls",
    "trusted_nonlinear_ops_count",
    "plaintext_hidden_materializations",
    "masked_residual_transport",
    "masked_inter_block_handoff",
    "silent_fallback_occurred",
)


def _collect_violations(profile: ExecutionProfile,
                        observed: Mapping[str, Any]) -> Tuple[str, ...]:
    v = []
    backend = observed.get("nonlinear_backend")
    if backend is None:
        v.append("nonlinear_backend missing from observed run facts")
    else:
        try:
            canon = normalize_nonlinear_backend(backend)
        except UnknownNonlinearBackend:
            canon = None
            v.append("nonlinear_backend %r is not a known design" % (backend,))
        if canon is not None and not profile.backend_allowed(canon):
            v.append("nonlinear_backend %r forbidden under %r (allowed=%s)"
                     % (canon, profile.name, list(profile.allowed_backends)))

    def _cap(key: str, cap: int):
        val = observed.get(key)
        if val is None:
            # For a zero-cap gate, a missing counter is itself a violation: we
            # cannot certify "0" if the run did not report it.
            if cap == 0:
                v.append("%s missing (a zero-cap gate cannot be certified)" % key)
            return
        try:
            iv = int(val)
        except (TypeError, ValueError):
            v.append("%s=%r is not an integer" % (key, val))
            return
        if iv > cap:
            v.append("%s=%d exceeds cap %d" % (key, iv, cap))

    _cap("nonlinear_trusted_calls", profile.max_nonlinear_trusted_calls)
    _cap("trusted_nonlinear_ops_count", profile.max_trusted_nonlinear_ops)
    _cap("plaintext_hidden_materializations",
         profile.max_plaintext_hidden_materializations)

    if profile.masked_residual_transport_required and \
            observed.get("masked_residual_transport") is not True:
        v.append("masked_residual_transport required True but observed %r"
                 % (observed.get("masked_residual_transport"),))
    if profile.masked_inter_block_handoff_required and \
            observed.get("masked_inter_block_handoff") is not True:
        v.append("masked_inter_block_handoff required True but observed %r"
                 % (observed.get("masked_inter_block_handoff"),))

    if profile.silent_fallback_forbidden and \
            observed.get("silent_fallback_occurred") is True:
        v.append("silent_fallback_occurred True but forbidden under %r"
                 % (profile.name,))
    return tuple(v)


def validate_run(profile_name: str, observed: Mapping[str, Any],
                 *, raise_on_violation: bool = True) -> Tuple[bool, Tuple[str, ...]]:
    """Check a run's observed facts against its declared profile.

    Returns ``(ok, violations)``. With ``raise_on_violation`` (default) a
    non-empty violation set raises :class:`ExecutionProfileViolation` -- the
    fail-closed path used at the security boundary."""
    profile = resolve_profile(profile_name)
    violations = _collect_violations(profile, observed)
    ok = not violations
    if not ok and raise_on_violation:
        raise ExecutionProfileViolation(profile.name, violations)
    return ok, violations


def security_claim_allowed(profile_name: str,
                           observed: Mapping[str, Any]) -> bool:
    """True iff the profile is claim-eligible AND every gate passes.

    Never raises -- returns False for an ineligible profile or any violation."""
    profile = resolve_profile(profile_name)
    if not profile.security_claim_eligible:
        return False
    ok, _ = validate_run(profile.name, observed, raise_on_violation=False)
    return ok


# ---------------------------------------------------------------------------
# Manifest + attestation binding
# ---------------------------------------------------------------------------


def profile_manifest_fields(profile_name: str,
                            observed: Optional[Mapping[str, Any]] = None
                            ) -> Dict[str, Any]:
    """The profile/backend/inter-block fields that MUST be written into every
    results manifest, plus (if given) the pass/fail + claim decision."""
    profile = resolve_profile(profile_name)
    out: Dict[str, Any] = {
        "execution_profile": profile.name,
        "execution_profile_schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "nonlinear_backend_default": profile.default_nonlinear_backend,
        "masked_inter_block": profile.masked_inter_block_handoff_required,
        "masked_residual_transport_required":
            profile.masked_residual_transport_required,
        "paper_facing": profile.paper_facing,
        "security_claim_eligible": profile.security_claim_eligible,
        "forbidden_backends": list(profile.forbidden_backends),
    }
    if observed is not None:
        ok, violations = validate_run(profile.name, observed,
                                      raise_on_violation=False)
        out["nonlinear_backend"] = observed.get("nonlinear_backend")
        out["profile_gates_passed"] = ok
        out["profile_violations"] = list(violations)
        out["security_claim_allowed"] = security_claim_allowed(
            profile.name, observed)
    return out


def execution_profile_metadata_hash(profile_name: str) -> str:
    """Stable digest of a profile record, to bind into config_digest /
    runtime hash / attestation report_data. Order-independent (sorted keys)."""
    profile = resolve_profile(profile_name)
    payload = dict(asdict(profile))
    payload["schema_version"] = EXECUTION_PROFILE_SCHEMA_VERSION
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
