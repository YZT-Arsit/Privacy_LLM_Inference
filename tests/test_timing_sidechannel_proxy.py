"""Stage 5.6 — tests for the timing / boundary-call side-channel proxy."""

from __future__ import annotations

import json

import pytest

from pllo.experiments.timing_sidechannel_proxy import (
    TimingSidechannelConfig,
    run_timing_sidechannel_proxy,
)


def _small_cfg(**overrides):
    cfg = dict(
        seed=2026,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        head_dim=4,
        layers=2,
        vocab_size=32,
        batch_size=1,
        prompt_lengths=(4, 8),
        decode_steps=(0, 1, 2),
        samples_per_bin=8,
        timing_noise_std=0.05,
    )
    cfg.update(overrides)
    return TimingSidechannelConfig(**cfg)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_timing_sidechannel_proxy(_small_cfg())


def test_latency_simulator_runs(report) -> None:
    assert "prompt_length_leakage" in report
    assert "decode_step_leakage" in report
    assert "method_distinguishability" in report
    assert "mitigation_distinguishability" in report


def test_prompt_length_leakage_emits_correlation(report) -> None:
    d = report["prompt_length_leakage"]
    assert "correlation_latency_length" in d
    assert "length_bucket_accuracy" in d
    assert "risk_level" in d


def test_method_distinguishability_emits_accuracy(report) -> None:
    d = report["method_distinguishability"]
    assert "method_accuracy" in d
    assert "random_chance_baseline" in d
    assert "risk_level" in d


def test_cost_model_note_marks_proxy(report) -> None:
    assert "NOT a real TEE" in report["cost_model_note"]


def test_no_wall_time_claim(report) -> None:
    # The cost_model_note must explicitly disclaim a wall_time measurement
    # and reaffirm `wall_time_source = projected_from_op_counts`. Any
    # appearance of `wall_time_source` in the report MUST be paired with
    # the projected-from-op-counts disclaimer.
    text = json.dumps(report, default=str).lower()
    if "wall_time_source" in text:
        assert "projected_from_op_counts" in text, (
            "wall_time_source mentioned but not paired with the"
            " projected-from-op-counts disclaimer"
        )
    note = report["cost_model_note"].lower()
    assert "not a real tee" in note or "not real tee" in note


def test_boundary_call_pattern_emits_per_method(report) -> None:
    rows = report["boundary_call_pattern"]
    methods = {r["method"] for r in rows}
    assert {
        "ours_current",
        "ours_compatible_nonlinear_islands",
        "tslp_trusted_nonlinear_baseline",
    } <= methods


def test_limitations_mentions_not_real_tee(report) -> None:
    text = " ".join(report["limitations"]).lower()
    assert "not real tee" in text or "not a real tee" in text or "not real" in text


def test_json_safe(report) -> None:
    text = json.dumps(report, default=str)
    assert "tensor(" not in text
