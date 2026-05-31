"""Stage 5.3c — workload profiler cross-architecture integration status tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments import (
    CrossArchitectureSummaryConfig,
    run_cross_architecture_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_PROFILE_JSON = PROJECT_ROOT / "outputs" / "workload_profile.json"
WORKLOAD_PROFILE_MD = PROJECT_ROOT / "outputs" / "workload_profile.md"
CROSS_SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "run_cross_architecture_summary.py"

METHOD_NAME = "ours_compatible_nonlinear_islands"


# ---------------------------------------------------------------------------
# Workload profile JSON
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def profile_payload() -> dict:
    if not WORKLOAD_PROFILE_JSON.exists():
        pytest.skip(
            "outputs/workload_profile.json missing — run "
            "`python scripts/run_workload_profile.py` first."
        )
    return json.loads(WORKLOAD_PROFILE_JSON.read_text(encoding="utf-8"))


def test_workload_bert_status_probe_level(profile_payload) -> None:
    status = profile_payload["methods"][METHOD_NAME]["wrapper_integration_status"]
    assert status["bert"] == "implemented_probe_level"


def test_workload_t5_status_probe_level(profile_payload) -> None:
    status = profile_payload["methods"][METHOD_NAME]["wrapper_integration_status"]
    assert status["t5"] == "implemented_probe_level"


def test_workload_measured_integration_scope_cross_architecture(
    profile_payload,
) -> None:
    m = profile_payload["methods"][METHOD_NAME]
    assert m["measured_integration_scope"] == "cross_architecture_probe_level"


def test_workload_all_architecture_probe_level_implemented(profile_payload) -> None:
    m = profile_payload["methods"][METHOD_NAME]
    assert m["all_architecture_probe_level_implemented"] is True


def test_workload_full_runtime_integrated_false(profile_payload) -> None:
    m = profile_payload["methods"][METHOD_NAME]
    assert m["full_runtime_integrated"] is False


def test_workload_implemented_remains_false(profile_payload) -> None:
    """``implemented=True`` would imply full runtime integration; must remain False."""
    m = profile_payload["methods"][METHOD_NAME]
    assert m["implemented"] is False
    assert m["wall_time_source"] == "projected_from_op_counts"


def test_workload_top_level_status_mirrors(profile_payload) -> None:
    top = profile_payload.get("wrapper_integration_status", {}).get(METHOD_NAME)
    assert top is not None
    assert top["bert"] == "implemented_probe_level"
    assert top["t5"] == "implemented_probe_level"
    assert top["measured_integration_scope"] == "cross_architecture_probe_level"
    assert top["full_runtime_integrated"] is False
    assert top["all_architecture_probe_level_implemented"] is True


def test_workload_markdown_contains_required_phrases() -> None:
    if not WORKLOAD_PROFILE_MD.exists():
        pytest.skip("outputs/workload_profile.md missing — run the script first.")
    md = WORKLOAD_PROFILE_MD.read_text(encoding="utf-8")
    for phrase in (
        "GPT-2 model-level integration is available",
        "BERT/T5 are probe-level integrations, not full wrappers",
        "measured_integration_scope = \"cross_architecture_probe_level\"",
        "full_runtime_integrated = False",
        "all_architecture_probe_level_implemented = True",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# Cross-architecture summary
# ---------------------------------------------------------------------------


def _write_workload_with_5_3c(path: Path) -> None:
    payload = {
        "config": {"layers": 2},
        "calibration": {"flops_per_op": 1.0},
        "methods": {
            "ours_current": {
                "implemented": True,
                "online_boundary_calls": 36,
                "boundary_calls_formula": "4L + 1 = 9 per forward",
                "online_trusted_compute_ops": 1_116_310,
                "online_gpu_ops": 4_429_848,
                "preprocessing_trusted_ops": 200_000,
                "measured_wall_time_ms": 6.5,
                "wall_time_source": "measured",
                "online_extra_matmul_count": 0,
                "uses_compatible_nonlinear_islands": False,
                "security_profile": "n/a",
            },
            METHOD_NAME: {
                "implemented": False,
                "online_boundary_calls": 16,
                "boundary_calls_formula": "L + 2 = 4 per forward (projected)",
                "online_trusted_compute_ops": 1_105_830,
                "online_gpu_ops": 4_434_424,
                "preprocessing_trusted_ops": 320_000,
                "measured_wall_time_ms": None,
                "wall_time_source": "projected_from_op_counts",
                "online_extra_matmul_count": 0,
                "uses_compatible_nonlinear_islands": True,
                "security_profile": "proxy-evaluated, not formal",
                "partial_implementation": True,
                "measured_integration_scope": "cross_architecture_probe_level",
                "all_architecture_probe_level_implemented": True,
                "full_runtime_integrated": False,
                "wrapper_integration_status": {
                    "gpt2_single_block": "implemented",
                    "gpt2_model_level": "implemented",
                    "bert": "implemented_probe_level",
                    "t5": "implemented_probe_level",
                },
            },
        },
        "paper_metrics": {
            METHOD_NAME: {
                "boundary_call_reduction_vs_ours_current": 0.555,
                "trusted_compute_reduction_vs_ours_current": 0.009,
                "preprocessing_cost_increase_vs_ours_current": 0.6,
                "online_extra_matmul_count": 0,
                "gpu_offload_ratio": 0.8,
                "projected_not_measured": True,
                "security_proxy_available": True,
                "security_proxy_caveats": [
                    "Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.",
                ],
            }
        },
        "wrapper_integration_status": {
            METHOD_NAME: {
                "gpt2_single_block": "implemented",
                "gpt2_model_level": "implemented",
                "bert": "implemented_probe_level",
                "t5": "implemented_probe_level",
                "measured_integration_scope": "cross_architecture_probe_level",
                "all_architecture_probe_level_implemented": True,
                "full_runtime_integrated": False,
                "note": "Stage 5.3c cross-architecture probe-level integration.",
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_coverage(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "coverage": [
                    {
                        "architecture_key": key,
                        "status": "loaded",
                        "model_id": model_id,
                        "spec": {
                            "model_class": model_class,
                            "has_encoder": key != "decoder_only",
                            "has_decoder": key != "encoder_only",
                            "has_cross_attention": key == "encoder_decoder",
                            "supports_past_key_values": True,
                            "vocab_size": 100,
                            "hidden_size": 8,
                            "num_layers": 2,
                            "num_heads": 2,
                        },
                        "module_paths": {},
                    }
                    for key, model_id, model_class in (
                        ("decoder_only", "sshleifer/tiny-gpt2", "GPT2LMHeadModel"),
                        ("encoder_only", "hf-internal-testing/tiny-bert", "BertForMaskedLM"),
                        (
                            "encoder_decoder",
                            "hf-internal-testing/tiny-random-t5",
                            "T5ForConditionalGeneration",
                        ),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )


def test_summary_exposes_compatible_island_integration_status(tmp_path) -> None:
    _write_workload_with_5_3c(tmp_path / "workload_profile.json")
    _write_minimal_coverage(tmp_path / "architecture_coverage.json")
    summary = run_cross_architecture_summary(
        CrossArchitectureSummaryConfig(output_dir=str(tmp_path))
    )
    integ = summary["compatible_island_integration_status"]
    assert integ["status"] == "available"
    assert integ["measured_integration_scope"] == "cross_architecture_probe_level"
    assert integ["full_runtime_integrated"] is False
    assert integ["all_architecture_probe_level_implemented"] is True
    rows = {entry["architecture_type"]: entry for entry in integ["per_architecture"]}
    assert rows["decoder_only"]["integration_level"] == "model_level"
    assert rows["encoder_only"]["integration_level"] == "probe_level"
    assert rows["encoder_decoder"]["integration_level"] == "probe_level"
    for entry in integ["per_architecture"]:
        assert entry["nonlinear_mode_available"] == ["trusted", "compatible_islands"]
        assert entry["use_pad_supported"] is True
        assert entry["online_extra_matmul_count"] == 0
        assert entry["security_proxy_status"] == "proxy-evaluated, not formal"
    summary_globals = summary["global_summary"]
    assert summary_globals["compatible_island_integration_status_available"] is True
    assert summary_globals["compatible_island_full_runtime_integrated"] is False
    assert (
        summary_globals["compatible_island_all_architecture_probe_level_implemented"]
        is True
    )


def test_script_markdown_contains_compatible_island_integration_status(
    tmp_path,
) -> None:
    _write_workload_with_5_3c(tmp_path / "workload_profile.json")
    _write_minimal_coverage(tmp_path / "architecture_coverage.json")
    subprocess.run(
        [sys.executable, str(CROSS_SUMMARY_SCRIPT), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    md = (tmp_path / "cross_architecture_summary.md").read_text(encoding="utf-8")
    assert "Compatible Island Integration Status" in md
    assert "GPT-2 model-level integration is available" in md
    assert "BERT/T5 are probe-level integrations, not full wrappers" in md
    assert "default mode remains `trusted`" in md
    assert (
        "LayerNorm remains trusted unless explicitly stated otherwise" in md
    )
    assert "no generation changes for BERT/T5" in md
    assert "security follows Stage 5.2b caveats" in md
    assert "not a real TEE measurement" in md
    assert "not full BERT/T5 wrapper integration" in md
    for arch in ("decoder_only", "encoder_only", "encoder_decoder"):
        assert arch in md
    assert "`measured_integration_scope = \"cross_architecture_probe_level\"`" in md
    assert "`full_runtime_integrated = False`" in md
    assert "`all_architecture_probe_level_implemented = True`" in md
