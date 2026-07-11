"""Tests for the unified masked-training TDX config schema (Gate 3.5 spec F)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.unified_training_config import (  # noqa: E402
    UnifiedConfigError,
    UnifiedTrainingConfig,
)


def _clean_observed():
    return {
        "nonlinear_backend": "A_rightmul",
        "logits_mask_family": "monomial",
        "nonlinear_trusted_calls": 0,
        "trusted_nonlinear_ops_count": 0,
        "plaintext_hidden_materializations": 0,
        "masked_residual_transport": True,
        "masked_inter_block_handoff": True,
        "silent_fallback_occurred": False,
    }


def test_default_config_is_paper_safe_and_valid():
    cfg = UnifiedTrainingConfig()
    assert cfg.execution_profile == "paper_safe"
    assert cfg.nonlinear_backend == "A_rightmul"
    assert cfg.optimizer_mode == "gpu_masked_sgd"
    assert cfg.masked_inter_block is True
    assert cfg.requires_fresh_attestation is True
    assert len(cfg.config_digest()) == 64


def test_config_digest_deterministic_and_binds_fields():
    a = UnifiedTrainingConfig().config_digest()
    b = UnifiedTrainingConfig().config_digest()
    assert a == b
    # a different model hash changes the digest
    c = UnifiedTrainingConfig(model_sha256="deadbeef").config_digest()
    assert c != a
    # a different mask family changes the digest
    d = UnifiedTrainingConfig(logits_mask_family="permutation").config_digest()
    assert d != a


def test_forbidden_backend_fail_closed():
    with pytest.raises(UnifiedConfigError):
        UnifiedTrainingConfig(nonlinear_backend="current")
    with pytest.raises(UnifiedConfigError):
        UnifiedTrainingConfig(nonlinear_backend="trusted_shortcut")


def test_bad_optimizer_mode_fail_closed():
    with pytest.raises(UnifiedConfigError):
        UnifiedTrainingConfig(optimizer_mode="trusted_adamw")


def test_bad_mask_family_fail_closed():
    with pytest.raises(UnifiedConfigError):
        UnifiedTrainingConfig(logits_mask_family="dense")


def test_paper_safe_forbids_expected_materializations():
    with pytest.raises(UnifiedConfigError):
        UnifiedTrainingConfig(plaintext_hidden_materializations_expected=1)


def test_runtime_and_report_binding_fields():
    cfg = UnifiedTrainingConfig()
    rh = cfg.runtime_hash_fields()
    assert rh["nonlinear_backend"] == "A_rightmul"
    assert "execution_profile_metadata_hash" in rh
    assert "nonlinear_design_metadata_hash" in rh
    rd = cfg.report_data_fields()
    assert rd["config_digest"] == cfg.config_digest()
    assert rd["requires_fresh_attestation"] is True


def test_manifest_fields_include_profile_and_claim():
    cfg = UnifiedTrainingConfig()
    m = cfg.manifest_fields(_clean_observed())
    assert m["execution_profile"] == "paper_safe"
    assert m["config_digest"] == cfg.config_digest()
    assert m["security_claim_allowed"] is True
    assert m["logits_mask_family"] == "monomial"


def test_validate_observed_pass_and_claim_allowed():
    cfg = UnifiedTrainingConfig()
    obs = _clean_observed()
    ok, violations = cfg.validate_observed(obs)
    assert ok and not violations
    assert cfg.security_claim_allowed(obs) is True


def test_validate_observed_backend_mismatch_fail_closed():
    cfg = UnifiedTrainingConfig()
    obs = _clean_observed()
    obs["nonlinear_backend"] = "amulet_secure_R"   # allowed by profile, != config
    with pytest.raises(UnifiedConfigError):
        cfg.validate_observed(obs)
    assert cfg.security_claim_allowed(obs) is False


def test_validate_observed_mask_family_mismatch_fail_closed():
    cfg = UnifiedTrainingConfig()               # monomial
    obs = _clean_observed()
    obs["logits_mask_family"] = "permutation"
    ok, violations = cfg.validate_observed(obs, raise_on_violation=False)
    assert not ok
    assert any("logits_mask_family" in v for v in violations)


def test_permutation_baseline_config_valid():
    cfg = UnifiedTrainingConfig(logits_mask_family="permutation")
    obs = _clean_observed()
    obs["logits_mask_family"] = "permutation"
    ok, _ = cfg.validate_observed(obs)
    assert ok
