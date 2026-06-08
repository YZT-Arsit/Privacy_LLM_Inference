"""Stage 7.8d tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.decoder_component_coverage_audit import (
    render_markdown,
    run_decoder_component_coverage_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_decoder_component_coverage_audit(
        outputs_dir=REPO_ROOT / "outputs"
    )


def _by_name(entries, name):
    for e in entries:
        if e["component"] == name:
            return e
    raise KeyError(name)


def test_required_components_present(report: dict) -> None:
    covered = {e["component"] for e in report["covered_in_main_protocol"]}
    partial = {e["component"] for e in report["partially_covered_or_extension"]}
    unsupp = {e["component"] for e in report["not_covered_future_work"]}
    must_be_covered = {
        "RMSNorm", "SwiGLU", "standard 1D RoPE", "GQA / MQA",
        "causal attention", "KV cache", "paged KV abstraction",
        "LM head", "LoRA inference",
        "generation processors inside TEE",
    }
    assert must_be_covered <= covered
    must_be_partial_or_covered = {
        "sliding window attention",
        "quantization (fp16 / bf16 / int8 / int4)",
    }
    assert must_be_partial_or_covered <= (covered | partial)
    must_be_unsupported = {
        "M-RoPE / multimodal positional encoding",
        "MoE router / expert dispatch",
        "speculative decoding",
        "real vLLM / FlashAttention backend",
        "real GPU / TEE hardware side channels",
        "full active malicious security",
        "LoRA training (backward)",
        "full Qwen / LLaMA deployment",
    }
    assert must_be_unsupported <= unsupp


def test_no_unsupported_component_marked_supported(report: dict) -> None:
    for e in report["not_covered_future_work"]:
        assert e["status"] == "unsupported", e["component"]


def test_rope_scaling_supported_only_under_same_plane_rotation(report: dict) -> None:
    e = _by_name(report["covered_in_main_protocol"], "standard 1D RoPE")
    assert "block-diagonal" in e["required_invariant"]
    assert "RoPE pair" in e["required_invariant"] or "RoPE-plane" in e["reason"]


def test_m_rope_unsupported(report: dict) -> None:
    e = _by_name(
        report["not_covered_future_work"],
        "M-RoPE / multimodal positional encoding",
    )
    assert e["status"] == "unsupported"


def test_moe_unsupported(report: dict) -> None:
    e = _by_name(
        report["not_covered_future_work"], "MoE router / expert dispatch"
    )
    assert e["status"] == "unsupported"


def test_speculative_decoding_unsupported(report: dict) -> None:
    e = _by_name(report["not_covered_future_work"], "speculative decoding")
    assert e["status"] == "unsupported"


def test_quantization_status_reflects_7_8b(report: dict) -> None:
    e = _by_name(
        report["partially_covered_or_extension"],
        "quantization (fp16 / bf16 / int8 / int4)",
    )
    artifact_path = (
        REPO_ROOT / "outputs" / "precision_quantization_stability.json"
    )
    if artifact_path.exists():
        assert e["status"] == "partially_supported"
    else:
        assert e["status"] == "audit_only"


def test_sliding_window_status_reflects_7_8a(report: dict) -> None:
    e = _by_name(
        report["partially_covered_or_extension"], "sliding window attention"
    )
    artifact_path = REPO_ROOT / "outputs" / "sliding_window_attention.json"
    if artifact_path.exists():
        assert e["status"] == "supported"
    else:
        assert e["status"] == "audit_only"


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Decoder-only Component Coverage Audit" in md
    assert "Covered in Main Protocol" in md
    assert "Partially Covered" in md
    assert "Not Covered" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "decoder_component_coverage_audit.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.8d"
