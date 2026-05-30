"""Stage 5.2c — cross-architecture summary tests for the compatible-island projection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pllo.experiments import (
    CrossArchitectureSummaryConfig,
    run_cross_architecture_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_cross_architecture_summary.py"
METHOD_NAME = "ours_compatible_nonlinear_islands"


# ---------------------------------------------------------------------------
# Synthetic upstream fixtures (no HF download)
# ---------------------------------------------------------------------------


def _write_workload_with_compatible_islands(path: Path) -> None:
    """Write a synthetic workload_profile.json containing the new method."""
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


# ---------------------------------------------------------------------------
# Projection structure
# ---------------------------------------------------------------------------


def test_summary_exposes_compatible_island_projection(tmp_path) -> None:
    _write_workload_with_compatible_islands(tmp_path / "workload_profile.json")
    _write_minimal_coverage(tmp_path / "architecture_coverage.json")
    summary = run_cross_architecture_summary(
        CrossArchitectureSummaryConfig(output_dir=str(tmp_path))
    )
    proj = summary["compatible_island_projection"]
    assert proj["status"] == "available"
    assert summary["global_summary"][
        "compatible_island_projection_available"
    ] is True
    per_arch = proj["per_architecture"]
    assert {entry["architecture_type"] for entry in per_arch} == {
        "decoder_only",
        "encoder_only",
        "encoder_decoder",
    }
    for entry in per_arch:
        assert entry["compatible_island_method"] == METHOD_NAME
        assert entry["online_extra_matmul_count"] == 0
        assert entry["status"] == "projected_from_probe"
        assert "Compatible mask families are weaker" in " ".join(
            entry["limitations"]
        )


def test_summary_projection_carries_paper_metrics(tmp_path) -> None:
    _write_workload_with_compatible_islands(tmp_path / "workload_profile.json")
    _write_minimal_coverage(tmp_path / "architecture_coverage.json")
    summary = run_cross_architecture_summary(
        CrossArchitectureSummaryConfig(output_dir=str(tmp_path))
    )
    pm = summary["compatible_island_projection"]["paper_metrics"]
    assert pm["projected_not_measured"] is True
    assert pm["security_proxy_available"] is True
    assert pm["online_extra_matmul_count"] == 0


def test_summary_handles_workload_without_compatible_islands(tmp_path) -> None:
    """If workload_profile.json lacks the new method, projection records absent."""
    (tmp_path / "workload_profile.json").write_text(
        json.dumps(
            {
                "config": {"layers": 2},
                "calibration": {},
                "methods": {
                    "ours_current": {
                        "implemented": True,
                        "online_boundary_calls": 36,
                        "boundary_calls_formula": "4L + 1",
                        "online_trusted_compute_ops": 1,
                        "online_gpu_ops": 1,
                        "preprocessing_trusted_ops": 1,
                        "measured_wall_time_ms": 1.0,
                        "wall_time_source": "measured",
                    }
                },
                "paper_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    _write_minimal_coverage(tmp_path / "architecture_coverage.json")
    summary = run_cross_architecture_summary(
        CrossArchitectureSummaryConfig(output_dir=str(tmp_path))
    )
    proj = summary["compatible_island_projection"]
    assert proj["status"] == "workload_missing_or_method_absent"
    assert summary["global_summary"][
        "compatible_island_projection_available"
    ] is False


# ---------------------------------------------------------------------------
# Markdown emitter end-to-end
# ---------------------------------------------------------------------------


def test_script_markdown_contains_compatible_island_projection(tmp_path) -> None:
    _write_workload_with_compatible_islands(tmp_path / "workload_profile.json")
    _write_minimal_coverage(tmp_path / "architecture_coverage.json")
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    md = (tmp_path / "cross_architecture_summary.md").read_text(encoding="utf-8")
    assert "Compatible Nonlinear Island Workload Projection" in md
    assert "projected_from_probe" in md
    assert "Compatible mask families are weaker" in md
    assert "Fresh permutation, dense sandwiching, and pad at Linear boundaries" in md
    # Each architecture must show up in the projection table.
    for arch in ("decoder_only", "encoder_only", "encoder_decoder"):
        assert arch in md
