"""Stage 7.7g tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.paper_claims_audit_v2 import (
    CLAIMS,
    render_markdown,
    run_paper_claims_audit_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_paper_claims_audit_v2(outputs_dir=REPO_ROOT / "outputs")


def test_all_claims_present(report: dict) -> None:
    # 15 baseline + 11 Stage 7.8 addendum = 26 claims.
    assert len(report["claims"]) == 26
    ids = {c["id"] for c in report["claims"]}
    baseline = {
        "padded_full_generation_correctness",
        "one_round_low_interaction_exact_mode",
        "rope_transient_plain_qk_eliminated",
        "norm_full_gram_reduced_by_token_chunk_masks",
        "attention_maps_hidden_only_in_trusted_softmax_mode",
        "attention_maps_visible_in_exact_low_interaction_mode",
        "scalable_lm_head_dense_mask_not_feasible",
        "lora_integration_supported_for_specified_sites",
        "paged_kv_invariant_supported_in_synthetic_abstraction",
        "multi_session_mask_isolation_supported_in_cpu_simulation",
        "integrity_only_probabilistic_spot_check",
        "no_real_gpu_or_tee_wall_clock",
        "no_formal_cryptographic_security",
        "no_full_qwen_or_llama_deployment_unless_real_wrapper",
        "no_hardware_side_channel_evaluation",
    }
    stage_7_8_addendum = {
        "sliding_window_attention_supported_in_cpu_synthetic_abstraction",
        "rolling_kv_window_invariant_supported",
        "fp16_bf16_int8_int4_simulated_only_not_real_kernels",
        "well_conditioned_masks_recommended_for_low_precision",
        "generation_processors_safe_only_inside_trusted_side",
        "output_length_side_channel_not_hidden_unless_separately_padded",
        "standard_1d_rope_scaling_covered_only_under_same_plane_rotation",
        "m_rope_multimodal_unsupported",
        "moe_unsupported",
        "speculative_decoding_unsupported",
        "quantized_real_model_deployment_unsupported_without_real_backend",
    }
    assert ids == baseline | stage_7_8_addendum


def test_unsupported_marked_unsupported(report: dict) -> None:
    must_be_unsupported = {
        "no_real_gpu_or_tee_wall_clock",
        "no_formal_cryptographic_security",
        "no_full_qwen_or_llama_deployment_unless_real_wrapper",
        "no_hardware_side_channel_evaluation",
        # Stage 7.8 addendum unsupported claims:
        "fp16_bf16_int8_int4_simulated_only_not_real_kernels",
        "output_length_side_channel_not_hidden_unless_separately_padded",
        "m_rope_multimodal_unsupported",
        "moe_unsupported",
        "speculative_decoding_unsupported",
        "quantized_real_model_deployment_unsupported_without_real_backend",
    }
    for c in report["claims"]:
        if c["id"] in must_be_unsupported:
            assert c["status"] == "unsupported", c["id"]


def test_stage_7_8_addendum_supported_claims(report: dict) -> None:
    must_be_supported = {
        "sliding_window_attention_supported_in_cpu_synthetic_abstraction",
        "rolling_kv_window_invariant_supported",
        "well_conditioned_masks_recommended_for_low_precision",
        "generation_processors_safe_only_inside_trusted_side",
        "standard_1d_rope_scaling_covered_only_under_same_plane_rotation",
    }
    for c in report["claims"]:
        if c["id"] in must_be_supported:
            assert c["status"] == "supported", c["id"]


def test_integrity_marked_proxy(report: dict) -> None:
    for c in report["claims"]:
        if c["id"] == "integrity_only_probabilistic_spot_check":
            assert c["status"] == "proxy_supported"


def test_attention_maps_classification(report: dict) -> None:
    ids = {c["id"]: c["status"] for c in report["claims"]}
    assert ids["attention_maps_hidden_only_in_trusted_softmax_mode"] == "supported"
    assert ids["attention_maps_visible_in_exact_low_interaction_mode"] == "supported"


def test_supported_claims_have_artifacts(report: dict) -> None:
    for c in report["claims"]:
        if c["status"] == "supported" and \
                c["evidence_artifact"].startswith("outputs/"):
            assert c["evidence_artifact_exists"] is True, c["id"]


def test_no_unsafe_wording_promoted(report: dict) -> None:
    # No claim's safe_wording should contain unsafe phrases.
    forbidden = [
        "cryptographic security",
        "hides attention maps in exact",
        "side channels evaluated",
        "wall-clock benchmark",
    ]
    for c in report["claims"]:
        safe = c["safe_wording"].lower()
        for ph in forbidden:
            assert ph.lower() not in safe, f"claim {c['id']} safe wording contains '{ph}'"


def test_summary_counts_match(report: dict) -> None:
    s = report["summary"]
    counts = {"supported": 0, "proxy_supported": 0,
              "cost_model_only": 0, "unsupported": 0}
    for c in report["claims"]:
        counts[c["status"]] += 1
    assert s == counts


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Paper Claims Audit v2" in md
    assert "Claims Table" in md
    assert "Safe Wording Per Claim" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "paper_claims_audit_v2.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7g"
