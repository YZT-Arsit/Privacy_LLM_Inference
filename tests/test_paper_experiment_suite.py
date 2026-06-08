"""Tests for the paper experiment suite aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.paper_experiment_suite import (
    PaperExperimentSuiteConfig,
    render_markdown,
    run_paper_experiment_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_paper_experiment_suite(
        cfg=PaperExperimentSuiteConfig(outputs_dir=REPO_ROOT / "outputs")
    )


def test_environment_flags(report: dict) -> None:
    env = report["environment"]
    assert env["device"] == "cpu"
    assert env["real_gpu"] is False
    assert env["real_tee"] is False
    assert env["network_required"] is False


def test_all_stages_present(report: dict) -> None:
    expected = {
        "7.6e_modern_decoder_generation_correctness",
        "7.6f_modern_decoder_low_interaction_correctness",
        "7.6g_modern_decoder_rope_safe_low_interaction",
        "7.6h_norm_granularity_low_interaction",
        "7.6i_attention_privacy_modes",
        "7.7a_lm_head_scalability",
        "7.7b_lora_integration",
        "7.7c_paged_kv_abstraction",
        "7.7d_multi_session_batching",
        "7.7e_integrity_spotcheck",
        "7.7f_complexity_model",
        "7.7g_paper_claims_audit_v2",
        "7.8a_sliding_window_attention",
        "7.8b_precision_quantization_stability",
        "7.8c_generation_processor_coverage",
        "7.8d_decoder_component_coverage_audit",
    }
    assert set(report["stages"].keys()) == expected


def test_decoder_component_coverage_section(report: dict) -> None:
    dc = report["decoder_component_coverage"]
    assert "RMSNorm" in dc["covered_components"]
    assert "SwiGLU" in dc["covered_components"]
    assert "GQA / MQA" in dc["covered_components"]
    assert "M-RoPE / multimodal positional encoding" in dc["unsupported_components"]
    assert "MoE router / expert dispatch" in dc["unsupported_components"]
    assert "speculative decoding" in dc["unsupported_components"]


def test_render_includes_decoder_coverage_section(report: dict) -> None:
    md = render_markdown(report)
    assert "Decoder-only Component Coverage" in md
    assert "Covered Components" in md
    assert "Unsupported Components" in md


def test_paper_claims_table_populated(report: dict) -> None:
    # 15 baseline + 11 Stage 7.8 addendum = 26.
    assert len(report["paper_claims_table"]) == 26
    assert len(report["supported_claims"]) >= 14
    assert len(report["unsupported_claims"]) >= 9


def test_unsupported_claims_listed(report: dict) -> None:
    must_be_unsupported = {
        "no_real_gpu_or_tee_wall_clock",
        "no_formal_cryptographic_security",
        "no_full_qwen_or_llama_deployment_unless_real_wrapper",
        "no_hardware_side_channel_evaluation",
    }
    assert set(report["unsupported_claims"]) >= must_be_unsupported


def test_limitations_state_cpu_only(report: dict) -> None:
    text = "\n".join(report["limitations"]).lower()
    assert "cpu local emulation" in text
    assert "no real tee" in text
    assert "no formal cryptographic" in text
    assert "no hardware side-channel" in text


def test_unsafe_wording_list_contains_canonical_items(report: dict) -> None:
    unsafe = report["unsafe_wording_to_avoid"]
    expected_phrases = [
        "real TEE/GPU performance",
        "formal cryptographic security",
        "attention maps hidden in exact low-interaction mode",
        "dense vocab mask is scalable",
        "active malicious accelerator fully handled",
        "hardware side channels evaluated",
    ]
    for ph in expected_phrases:
        assert any(ph.lower() in u.lower() for u in unsafe), ph


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    for header in (
        "Paper-Ready Experiment Suite",
        "Experiment Matrix",
        "Paper Claims Summary",
        "Mode Comparison",
        "Remaining Blockers",
        "Recommended Paper Wording",
        "Unsafe Wording to Avoid",
    ):
        assert header in md, header


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "paper_experiment_suite.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7"
