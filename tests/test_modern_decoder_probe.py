"""Stage 6.4 — Modern decoder-only orchestrator + script tests.

The default test config runs in synthetic-only mode so the test suite
never depends on HuggingFace network downloads. ``attempt_real_model_load``
remains ``False`` unless an explicit opt-in test sets it.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.modern_decoder_probe import (
    ModernDecoderProbeConfig,
    run_modern_decoder_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_modern_decoder_probe.py"
WORKLOAD_PROFILE_JSON = PROJECT_ROOT / "outputs" / "workload_profile.json"
WORKLOAD_PROFILE_MD = PROJECT_ROOT / "outputs" / "workload_profile.md"
CROSS_ARCH_MD = PROJECT_ROOT / "outputs" / "cross_architecture_summary.md"
CROSS_ARCH_JSON = PROJECT_ROOT / "outputs" / "cross_architecture_summary.json"


# ---------------------------------------------------------------------------
# Synthetic-only orchestrator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_report() -> dict:
    return run_modern_decoder_probe(
        ModernDecoderProbeConfig(
            batch_size=2,
            seq_len=8,
            hidden_size=64,
            intermediate_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=16,
            attempt_real_model_load=False,
        )
    )


def test_synthetic_mode_does_not_require_network(synthetic_report) -> None:
    ml = synthetic_report["model_loading"]
    assert ml["status"] == "synthetic_only"
    assert ml["model_id"] is None


def test_architecture_spec_documents_rmsnorm_swiglu_rotary(synthetic_report) -> None:
    spec = synthetic_report["architecture_spec"]
    assert spec["norm_type"] == "rmsnorm"
    assert spec["activation_type"] == "swiglu"
    assert spec["position_encoding_type"] == "rotary"
    assert spec["attention_variant"] in {"mha", "gqa", "mqa"}


def test_rmsnorm_probe_allclose(synthetic_report) -> None:
    rms = synthetic_report["rmsnorm_probe"]
    assert rms["allclose"] is True
    for r in rms["per_use_pad"].values():
        assert r["allclose"] is True
        assert r["online_extra_matmul_count"] == 0


def test_swiglu_probe_allclose(synthetic_report) -> None:
    swi = synthetic_report["swiglu_probe"]
    assert swi["allclose"] is True
    for r in swi["per_use_pad"].values():
        assert r["allclose"] is True
        assert r["online_extra_matmul_count"] == 0
        assert r["shared_permutation_for_up_gate"] is True
        assert r["permutation_dim"] == r["intermediate_size"]


def test_rope_probe_post_mask_allclose(synthetic_report) -> None:
    rope = synthetic_report["rope_probe"]
    assert rope["status"] == "ok"
    a = rope["probe_a_post_rope_masking_invariant"]
    assert a["allclose"] is True


def test_gqa_probe_allclose(synthetic_report) -> None:
    gqa = synthetic_report["gqa_probe"]
    assert gqa["status"] == "ok"
    assert gqa["allclose"] is True


def test_pad_placement_linear_boundary_only_when_padded(synthetic_report) -> None:
    swi = synthetic_report["swiglu_probe"]["per_use_pad"]
    assert swi["True"]["pad_placement"] == "linear_boundary_only"
    assert swi["False"]["pad_placement"] == "n/a"


def test_global_summary_records_integration_level(synthetic_report) -> None:
    g = synthetic_report["global_summary"]
    assert g["integration_level"] == "probe_level"
    assert g["all_required_probes_allclose"] is True
    assert g["online_extra_matmul_count"] == 0
    assert g["default_nonlinear_mode"] == "trusted"
    assert g["norm_type"] == "rmsnorm"
    assert g["activation_type"] == "swiglu"
    assert g["position_encoding_type"] == "rotary"


def test_limitations_inherit_stage_5_4_caveats(synthetic_report) -> None:
    text = " ".join(synthetic_report["limitations"]).lower()
    assert "stage 5.4" in text
    assert "not formal security" in text
    assert "not a real tee" in text
    assert "fresh permutation" in text
    assert "dense sandwich" in text


# ---------------------------------------------------------------------------
# Script end-to-end
# ---------------------------------------------------------------------------


def test_script_generates_json_csv_markdown(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--hidden-size",
            "64",
            "--intermediate-size",
            "128",
            "--num-query-heads",
            "4",
            "--num-kv-heads",
            "2",
            "--head-dim",
            "16",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert (tmp_path / "modern_decoder_probe.json").exists()
    assert (tmp_path / "modern_decoder_probe.csv").exists()
    md = (tmp_path / "modern_decoder_probe.md").read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Model Loading Status",
        "Modern Decoder Architecture Spec",
        "RMSNorm Orthogonal Island Probe",
        "SwiGLU Paired-Permutation Island Probe",
        "RoPE-Aware Attention Probe",
        "GQA / MQA KV Shape Probe",
        "Workload / Integration Status",
        "Security Caveats from Stage 5.4",
        "Limitations",
        "Next Stage Plan",
        "fresh permutation",
        "dense sandwich",
        "pad at Linear boundaries",
        "not a real TEE measurement",
        "not formal security",
        "probe-level migration",
        "RMSNorm",
        "SwiGLU",
        "RoPE",
        "GQA",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"
    # CSV has long format with the expected sections.
    with (tmp_path / "modern_decoder_probe.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert {
        "model_loading",
        "architecture_spec",
        "rmsnorm_probe",
        "swiglu_probe",
        "rope_probe",
        "gqa_probe",
    } <= sections


# ---------------------------------------------------------------------------
# Workload profiler + cross-architecture summary integration
# ---------------------------------------------------------------------------


def test_workload_profile_records_modern_decoder_integration() -> None:
    if not WORKLOAD_PROFILE_JSON.exists():
        pytest.skip(
            "outputs/workload_profile.json missing — run "
            "`python scripts/run_workload_profile.py` first."
        )
    payload = json.loads(WORKLOAD_PROFILE_JSON.read_text(encoding="utf-8"))
    method = payload["methods"]["ours_compatible_nonlinear_islands"]
    status = method["wrapper_integration_status"]
    assert status["qwen_or_modern_decoder"] == "implemented_probe_level"
    assert status["modern_decoder_probe"] == "implemented"
    assert (
        method["measured_integration_scope"]
        == "cross_architecture_plus_modern_decoder_probe_level"
    )
    # implemented must still be False — probe-level only.
    assert method["implemented"] is False
    top = payload["wrapper_integration_status"][
        "ours_compatible_nonlinear_islands"
    ]
    assert top["qwen_or_modern_decoder"] == "implemented_probe_level"
    assert top["modern_decoder_probe"] == "implemented"


def test_workload_markdown_mentions_modern_decoder() -> None:
    if not WORKLOAD_PROFILE_MD.exists():
        pytest.skip("outputs/workload_profile.md missing")
    md = WORKLOAD_PROFILE_MD.read_text(encoding="utf-8")
    for phrase in (
        "probe-level migration",
        "RMSNorm",
        "SwiGLU",
        "RoPE",
        "GQA",
        "modern_decoder_probe",
        "not full",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"


def test_cross_architecture_summary_records_modern_decoder_row() -> None:
    if not CROSS_ARCH_JSON.exists():
        pytest.skip("outputs/cross_architecture_summary.json missing")
    payload = json.loads(CROSS_ARCH_JSON.read_text(encoding="utf-8"))
    integ = payload["compatible_island_integration_status"]
    rows = {entry["architecture_type"]: entry for entry in integ["per_architecture"]}
    assert "modern_decoder_only" in rows
    mod = rows["modern_decoder_only"]
    assert mod["integration_level"] == "probe_level"
    assert mod["nonlinear_mode_available"] == ["trusted", "compatible_islands"]
    assert mod["online_extra_matmul_count"] == 0


def test_cross_architecture_summary_markdown_includes_modern_decoder() -> None:
    if not CROSS_ARCH_MD.exists():
        pytest.skip("outputs/cross_architecture_summary.md missing")
    md = CROSS_ARCH_MD.read_text(encoding="utf-8")
    assert "modern_decoder_only" in md
