"""Stage 5.6 — tests for the inter-block residual masking gap probe."""

from __future__ import annotations

import json

import pytest

from pllo.experiments.inter_block_masking_probe import (
    INTER_BLOCK_PLAIN_TENSORS,
    InterBlockMaskingProbeConfig,
    run_inter_block_masking_probe,
)


def _cfg(**overrides):
    cfg = dict(
        seed=2026,
        stage_5_5b_artifact="outputs/real_token_activation_attacks.json",
        inter_block_mask_mode="plain_boundary",
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        batch_size=2,
        seq_len=6,
    )
    cfg.update(overrides)
    return InterBlockMaskingProbeConfig(**cfg)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_inter_block_masking_probe(_cfg())


def test_gap_accounting_flags_boundary_input_and_final(report) -> None:
    assert report["current_plain_boundary_detected"] is True
    assert "boundary_input" in report["affected_tensors"]
    assert "final" in report["affected_tensors"]
    assert tuple(report["affected_tensors"]) == INTER_BLOCK_PLAIN_TENSORS


def test_accounting_risk_high_under_default(report) -> None:
    assert report["accounting_risk_level"] in ("high", "medium")


def test_single_transition_probe_allclose(report) -> None:
    probe = report["single_transition_probe"]
    assert probe["rmsnorm_invariant_allclose"] is True
    assert probe["q_projection_path_allclose"] is True
    assert probe["residual_recovery_allclose"] is True
    assert report["single_transition_probe_status"] == "single_transition_probe_passed"


def test_masked_boundary_experimental_default_off(report) -> None:
    assert report["masked_boundary_experimental_default"] == "off"


def test_masked_boundary_experimental_explicit_request_is_not_implemented() -> None:
    """When the user explicitly requests masked_boundary_experimental, the
    probe must return ``not_implemented_in_stage_5_6`` rather than silently
    passing."""
    r = run_inter_block_masking_probe(
        _cfg(inter_block_mask_mode="masked_boundary_experimental")
    )
    assert r["masked_boundary_experimental_status"] == "not_implemented_in_stage_5_6"


def test_no_secret_mask_in_output(report) -> None:
    text = json.dumps(report, default=str)
    assert "tensor(" not in text
    # No "N_inter": " followed by a numeric tensor dump.
    assert "N_inter=" not in text
