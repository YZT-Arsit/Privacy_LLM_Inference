"""Gate 3.5 -- config schema for a real unified masked-training TDX run.

Prepares (but does NOT start) the config for the unified masked-training real run.
It freezes the fields, produces a ``config_digest``, and exposes the exact field
sets that must be folded into the attestation ``report_data`` / runtime hash and
written into the results manifest + validator. No quote / nonce / ECDH key /
session is created here -- per the Gate 3.5 rule, the old Gate-2/Gate-3 quote,
nonce and session must NOT be reused; a fresh quote is taken only when the real
unified run actually starts (``requires_fresh_attestation``).

Every field is gated through the ``paper_safe`` execution profile, so a config
that names a forbidden backend / non-paper profile is fail-closed at construction.

stdlib only (hashlib/json/dataclasses); imports the stdlib-only
:mod:`pllo.experiments.execution_profile` and ``nonlinear_designs``. Safe to
import from trusted-boundary code measured for the runtime hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from pllo.experiments.execution_profile import (
    ExecutionProfileViolation,
    execution_profile_metadata_hash,
    profile_manifest_fields,
    resolve_profile,
    security_claim_allowed,
    validate_run,
)
from pllo.experiments.nonlinear_designs import (
    nonlinear_design_metadata_hash,
    normalize_nonlinear_backend,
)

__all__ = [
    "UNIFIED_TRAINING_CONFIG_SCHEMA_VERSION",
    "LOGITS_MASK_FAMILIES",
    "UnifiedTrainingConfig",
    "UnifiedConfigError",
]

UNIFIED_TRAINING_CONFIG_SCHEMA_VERSION = "1.0"

# Frozen choice for the first unified run: monomial (hides the logit value
# multiset) with permutation kept as an explicit baseline. Dense vocab masks are
# not offered (see monomial_logit_mask -- ~92 GB at V=151936, unnecessary).
LOGITS_MASK_FAMILIES: Tuple[str, ...] = ("monomial", "permutation")


class UnifiedConfigError(ValueError):
    """A unified-training config that violates the schema / profile (fail-closed)."""


@dataclass(frozen=True)
class UnifiedTrainingConfig:
    model_id: str = "Qwen2.5-0.5B"
    # exact model weight hash -- filled from the real checkpoint at run time.
    model_sha256: str = "PENDING_REAL_CHECKPOINT_SHA256"
    execution_profile: str = "paper_safe"
    optimizer_mode: str = "gpu_masked_sgd"
    nonlinear_backend: str = "A_rightmul"
    logits_mask_family: str = "monomial"      # frozen; "permutation" = baseline
    gradient_convention: str = "nout_dual"
    masked_inter_block: bool = True
    plaintext_hidden_materializations_expected: int = 0
    # a fresh quote/nonce/ECDH/session is mandatory at real-run start (no reuse
    # of Gate-2/Gate-3 attestation material).
    requires_fresh_attestation: bool = True
    schema_version: str = UNIFIED_TRAINING_CONFIG_SCHEMA_VERSION

    def __post_init__(self):
        prof = resolve_profile(self.execution_profile)
        # backend must be allowed under the profile (fail-closed, never silent)
        try:
            canon = prof.canonical_backend(self.nonlinear_backend)
        except ExecutionProfileViolation as e:
            raise UnifiedConfigError(str(e)) from e
        object.__setattr__(self, "nonlinear_backend", canon)
        if self.logits_mask_family not in LOGITS_MASK_FAMILIES:
            raise UnifiedConfigError(
                "logits_mask_family %r not in %s"
                % (self.logits_mask_family, list(LOGITS_MASK_FAMILIES)))
        if self.optimizer_mode != "gpu_masked_sgd":
            raise UnifiedConfigError(
                "unified masked training requires optimizer_mode="
                "'gpu_masked_sgd' (no trusted optimizer / packed update); got %r"
                % (self.optimizer_mode,))
        if prof.masked_inter_block_handoff_required and not self.masked_inter_block:
            raise UnifiedConfigError(
                "profile %r requires masked_inter_block=True" % prof.name)
        if prof.max_plaintext_hidden_materializations == 0 and \
                self.plaintext_hidden_materializations_expected != 0:
            raise UnifiedConfigError(
                "profile %r forbids plaintext hidden materialisations; expected "
                "must be 0" % prof.name)

    # -- binding surfaces ---------------------------------------------------
    def _canonical_payload(self) -> Dict[str, Any]:
        p = dict(asdict(self))
        p["execution_profile_metadata_hash"] = execution_profile_metadata_hash(
            self.execution_profile)
        p["nonlinear_design_metadata_hash"] = nonlinear_design_metadata_hash(
            self.nonlinear_backend)
        return p

    def config_digest(self) -> str:
        blob = json.dumps(self._canonical_payload(), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def runtime_hash_fields(self) -> Dict[str, Any]:
        """Fields that must be folded into the service runtime hash (bind the
        design + profile into attestation so a design/profile change re-quotes)."""
        return {
            "execution_profile": self.execution_profile,
            "execution_profile_metadata_hash":
                execution_profile_metadata_hash(self.execution_profile),
            "nonlinear_backend": self.nonlinear_backend,
            "nonlinear_design_metadata_hash":
                nonlinear_design_metadata_hash(self.nonlinear_backend),
            "optimizer_mode": self.optimizer_mode,
            "logits_mask_family": self.logits_mask_family,
            "gradient_convention": self.gradient_convention,
            "masked_inter_block": self.masked_inter_block,
        }

    def report_data_fields(self) -> Dict[str, Any]:
        """Fields to bind into the attestation report_data (over and above the
        runtime hash): the config digest + model identity + freshness marker."""
        return {
            "config_digest": self.config_digest(),
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "requires_fresh_attestation": self.requires_fresh_attestation,
        }

    def manifest_fields(self,
                        observed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fields for the results manifest (profile + backend + inter-block +
        digest + claim decision)."""
        out: Dict[str, Any] = {
            "unified_training_schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "optimizer_mode": self.optimizer_mode,
            "logits_mask_family": self.logits_mask_family,
            "gradient_convention": self.gradient_convention,
            "config_digest": self.config_digest(),
            "requires_fresh_attestation": self.requires_fresh_attestation,
        }
        out.update(profile_manifest_fields(self.execution_profile, observed))
        return out

    # -- validator ----------------------------------------------------------
    def validate_observed(self, observed: Dict[str, Any],
                          *, raise_on_violation: bool = True
                          ) -> Tuple[bool, Tuple[str, ...]]:
        """Validate a real run's observed facts against this config + profile.

        Also checks the observed nonlinear backend / mask family match the frozen
        config (a mismatch is fail-closed, not a silent downgrade)."""
        extra = []
        obs_backend = observed.get("nonlinear_backend")
        if obs_backend is not None and \
                normalize_nonlinear_backend(obs_backend) != self.nonlinear_backend:
            extra.append("observed nonlinear_backend %r != config %r"
                         % (obs_backend, self.nonlinear_backend))
        fam = observed.get("logits_mask_family")
        if fam is not None and fam != self.logits_mask_family:
            extra.append("observed logits_mask_family %r != config %r"
                         % (fam, self.logits_mask_family))
        ok, violations = validate_run(self.execution_profile, observed,
                                      raise_on_violation=False)
        all_v = tuple(extra) + tuple(violations)
        ok = not all_v
        if not ok and raise_on_violation:
            raise UnifiedConfigError(
                "unified config validation failed (fail-closed): %s"
                % "; ".join(all_v))
        return ok, all_v

    def security_claim_allowed(self, observed: Dict[str, Any]) -> bool:
        ok, _ = self.validate_observed(observed, raise_on_violation=False)
        return ok and security_claim_allowed(self.execution_profile, observed)
