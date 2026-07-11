"""Tests for the explicit execution-profile gate (paper_safe / legacy_debug /
experimental). Enforces the fail-closed matrix from Gate 3.5 spec B."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.execution_profile import (  # noqa: E402
    DEFAULT_PROFILE,
    ExecutionProfileViolation,
    UnknownExecutionProfile,
    execution_profile_metadata_hash,
    get_profile,
    profile_manifest_fields,
    resolve_profile,
    security_claim_allowed,
    validate_run,
)


def _clean_paper_safe_observed(backend="A_rightmul"):
    """A run that satisfies every paper_safe gate."""
    return {
        "nonlinear_backend": backend,
        "nonlinear_trusted_calls": 0,
        "trusted_nonlinear_ops_count": 0,
        "plaintext_hidden_materializations": 0,
        "masked_residual_transport": True,
        "masked_inter_block_handoff": True,
        "silent_fallback_occurred": False,
    }


# -- default resolution -----------------------------------------------------

def test_missing_profile_defaults_to_paper_safe():
    assert resolve_profile(None).name == "paper_safe"
    assert resolve_profile("").name == "paper_safe"
    assert DEFAULT_PROFILE == "paper_safe"


def test_present_but_unknown_profile_is_fail_closed():
    with pytest.raises(UnknownExecutionProfile):
        get_profile("totally_made_up")
    with pytest.raises(UnknownExecutionProfile):
        resolve_profile("totally_made_up")   # typo != silent default


# -- paper_safe rejects legacy designs (spec B matrix) ----------------------

def test_paper_safe_plus_current_rejected():
    obs = _clean_paper_safe_observed(backend="current")
    with pytest.raises(ExecutionProfileViolation):
        validate_run("paper_safe", obs)
    assert security_claim_allowed("paper_safe", obs) is False


def test_paper_safe_plus_trusted_shortcut_rejected():
    obs = _clean_paper_safe_observed(backend="trusted_shortcut")
    with pytest.raises(ExecutionProfileViolation):
        validate_run("paper_safe", obs)
    assert security_claim_allowed("paper_safe", obs) is False


def test_paper_safe_plus_plaintext_materialization_rejected():
    obs = _clean_paper_safe_observed()
    obs["plaintext_hidden_materializations"] = 1
    with pytest.raises(ExecutionProfileViolation) as e:
        validate_run("paper_safe", obs)
    assert "plaintext_hidden_materializations" in str(e.value)
    assert security_claim_allowed("paper_safe", obs) is False


def test_paper_safe_plus_trusted_nonlinear_call_rejected():
    obs = _clean_paper_safe_observed()
    obs["nonlinear_trusted_calls"] = 3
    with pytest.raises(ExecutionProfileViolation):
        validate_run("paper_safe", obs)


def test_paper_safe_requires_masked_inter_block_and_transport():
    for key in ("masked_residual_transport", "masked_inter_block_handoff"):
        obs = _clean_paper_safe_observed()
        obs[key] = False
        with pytest.raises(ExecutionProfileViolation) as e:
            validate_run("paper_safe", obs)
        assert key in str(e.value)


def test_paper_safe_silent_fallback_forbidden():
    obs = _clean_paper_safe_observed()
    obs["silent_fallback_occurred"] = True
    with pytest.raises(ExecutionProfileViolation):
        validate_run("paper_safe", obs)


def test_paper_safe_missing_zero_cap_counter_is_violation():
    # a zero-cap gate cannot be certified if the run never reported the counter
    obs = _clean_paper_safe_observed()
    del obs["nonlinear_trusted_calls"]
    ok, violations = validate_run("paper_safe", obs, raise_on_violation=False)
    assert not ok
    assert any("nonlinear_trusted_calls missing" in v for v in violations)


def test_paper_safe_clean_run_passes_and_allows_claim():
    obs = _clean_paper_safe_observed()
    ok, violations = validate_run("paper_safe", obs)
    assert ok and not violations
    assert security_claim_allowed("paper_safe", obs) is True


def test_paper_safe_amulet_secure_r_allowed():
    obs = _clean_paper_safe_observed(backend="amulet_secure_R")
    ok, _ = validate_run("paper_safe", obs)
    assert ok


# -- requested-but-forbidden backend is never silently replaced -------------

def test_canonical_backend_fail_closed_on_forbidden():
    p = get_profile("paper_safe")
    assert p.canonical_backend(None) == "A_rightmul"     # default when unset
    assert p.canonical_backend("A_rightmul") == "A_rightmul"
    with pytest.raises(ExecutionProfileViolation):
        p.canonical_backend("current")                   # not silently -> default


# -- legacy_debug: runnable but never claim-eligible ------------------------

def test_legacy_debug_runs_current_but_no_claim():
    obs = _clean_paper_safe_observed(backend="current")
    obs["plaintext_hidden_materializations"] = 42        # recorded, not fatal
    ok, violations = validate_run("legacy_debug", obs)
    assert ok and not violations                         # legacy tolerates it
    assert security_claim_allowed("legacy_debug", obs) is False
    assert get_profile("legacy_debug").record_plaintext_materialization is True
    assert get_profile("legacy_debug").requires_explicit_opt_in is True


def test_experimental_no_claim_by_default():
    obs = _clean_paper_safe_observed()
    assert security_claim_allowed("experimental", obs) is False
    assert get_profile("experimental").requires_explicit_opt_in is True


# -- manifest fields (spec B: profile/backend/inter-block into manifest) -----

def test_manifest_fields_present():
    obs = _clean_paper_safe_observed()
    m = profile_manifest_fields("paper_safe", obs)
    for k in ("execution_profile", "nonlinear_backend", "masked_inter_block",
              "paper_facing", "profile_gates_passed", "security_claim_allowed",
              "forbidden_backends"):
        assert k in m, k
    assert m["execution_profile"] == "paper_safe"
    assert m["profile_gates_passed"] is True
    assert m["security_claim_allowed"] is True
    assert "current" in m["forbidden_backends"]


def test_metadata_hash_stable_and_distinct():
    a = execution_profile_metadata_hash("paper_safe")
    b = execution_profile_metadata_hash("paper_safe")
    c = execution_profile_metadata_hash("legacy_debug")
    assert a == b and a != c and len(a) == 64
