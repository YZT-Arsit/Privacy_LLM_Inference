"""Tests for the Stage 6.3 cross-architecture summary aggregator."""

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


# ---------------------------------------------------------------------------
# Synthetic upstream JSON fixtures
# ---------------------------------------------------------------------------


def _write_decoder_attention(path: Path) -> None:
    payload = {
        "results": [
            {
                "config": {
                    "model_id": "sshleifer/tiny-gpt2",
                    "batch_size": 1,
                    "seq_len": 4,
                    "decode_steps": 1,
                    "use_pad": True,
                },
                "full_attention": {
                    "output_metrics": {"max_abs_error": 1e-8, "allclose": True},
                    "score_metrics": {"max_abs_error": 1e-10, "allclose": True},
                    "prob_metrics": {"max_abs_error": 0.0, "allclose": True},
                    "v_aggr_metrics": {"max_abs_error": 2e-9, "allclose": True},
                    "qk_constraint_error": 1e-7,
                    "allclose": True,
                },
                "prefill_attention": {
                    "output_metrics": {"max_abs_error": 1e-8, "allclose": True},
                    "cache_key_metrics": {"max_abs_error": 1e-9, "allclose": True},
                    "cache_value_metrics": {"max_abs_error": 2e-9, "allclose": True},
                    "cache_invariant_allclose": True,
                },
                "decode_attention": {
                    "per_step": [{"output_metrics": {"max_abs_error": 1e-9, "allclose": True}}],
                    "decode_output_max_abs_error_max": 1e-9,
                    "cache_append_key_metrics": {"max_abs_error": 3e-9, "allclose": True},
                    "cache_append_value_metrics": {"max_abs_error": 4e-9, "allclose": True},
                    "cache_append_invariant_allclose": True,
                },
                "mask_structure": {},
                "pad_report": {"use_pad": True},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_encoder_attention(path: Path) -> None:
    payload = {
        "results": [
            {
                "config": {
                    "model_id": "hf-internal-testing/tiny-bert",
                    "batch_size": 1,
                    "seq_len": 4,
                    "use_pad": False,
                },
                "model_loading": {
                    "status": "loaded",
                    "model_id": "hf-internal-testing/tiny-bert",
                    "model_class": "BertForMaskedLM",
                    "hidden_size": 128,
                    "num_attention_heads": 2,
                    "head_dim": 64,
                },
                "qkv_invariants": {
                    "qk_constraint_error": 1e-7,
                    "qkv_allclose": True,
                },
                "results_per_mask": {
                    "all_ones": {
                        "score_metrics": {"max_abs_error": 1e-6, "allclose": True},
                        "prob_metrics": {"max_abs_error": 1e-7, "allclose": True},
                        "v_aggr_metrics": {"max_abs_error": 2e-6, "allclose": True},
                        "output_metrics": {"max_abs_error": 3e-6, "allclose": True},
                        "allclose": True,
                    },
                    "padding": {
                        "score_metrics": {"max_abs_error": 1e-6, "allclose": True},
                        "prob_metrics": {"max_abs_error": 1e-7, "allclose": True},
                        "v_aggr_metrics": {"max_abs_error": 2e-6, "allclose": True},
                        "output_metrics": {"max_abs_error": 4e-6, "allclose": True},
                        "allclose": True,
                    },
                },
                "pad_report": {"use_pad": False, "per_mask": {}},
                "mask_structure": {},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cross_attention(path: Path) -> None:
    payload = {
        "results": [
            {
                "config": {
                    "model_id": "hf-internal-testing/tiny-random-t5",
                    "batch_size": 1,
                    "dec_seq_len": 4,
                    "enc_seq_len": 8,
                    "use_pad": True,
                },
                "model_loading": {
                    "status": "loaded",
                    "model_id": "hf-internal-testing/tiny-random-t5",
                    "model_class": "T5ForConditionalGeneration",
                    "family": "t5",
                    "hidden_size": 32,
                    "num_attention_heads": 4,
                    "head_dim": 8,
                    "inner_dim": 32,
                    "bias_present": {"q": False, "k": False, "v": False, "o": False},
                    "cross_attention_has_relative_bias": False,
                },
                "qkv_invariants": {
                    "qk_constraint_error": 1e-7,
                    "qkv_allclose": True,
                },
                "encoder_memory_cache": {
                    "key_metrics": {"max_abs_error": 5e-7, "allclose": True},
                    "value_metrics": {"max_abs_error": 6e-7, "allclose": True},
                    "encoder_seq_len": 8,
                    "batch_size": 1,
                    "allclose": True,
                },
                "results_per_mask": {
                    "all_ones": {
                        "score_metrics": {"max_abs_error": 1e-6, "allclose": True},
                        "prob_metrics": {"max_abs_error": 1e-7, "allclose": True},
                        "v_aggr_metrics": {"max_abs_error": 2e-6, "allclose": True},
                        "output_metrics": {"max_abs_error": 3e-6, "allclose": True},
                        "allclose": True,
                    },
                    "padding": {
                        "score_metrics": {"max_abs_error": 1e-6, "allclose": True},
                        "prob_metrics": {"max_abs_error": 1e-7, "allclose": True},
                        "v_aggr_metrics": {"max_abs_error": 2e-6, "allclose": True},
                        "output_metrics": {"max_abs_error": 4e-6, "allclose": True},
                        "allclose": True,
                    },
                },
                "pad_report": {"use_pad": True, "per_mask": {}},
                "mask_structure": {},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_coverage(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "coverage": [
                    {
                        "architecture_key": "decoder_only",
                        "status": "loaded",
                        "model_id": "sshleifer/tiny-gpt2",
                        "spec": {
                            "model_class": "GPT2LMHeadModel",
                            "has_encoder": False,
                            "has_decoder": True,
                            "has_cross_attention": False,
                            "supports_past_key_values": True,
                            "vocab_size": 50257,
                            "hidden_size": 2,
                            "num_layers": 2,
                            "num_heads": 2,
                        },
                        "module_paths": {"self_attention": "transformer.h.0.attn"},
                    },
                    {
                        "architecture_key": "encoder_only",
                        "status": "loaded",
                        "model_id": "hf-internal-testing/tiny-bert",
                        "spec": {
                            "model_class": "BertForMaskedLM",
                            "has_encoder": True,
                            "has_decoder": False,
                            "has_cross_attention": False,
                            "supports_past_key_values": False,
                            "vocab_size": 100,
                            "hidden_size": 128,
                            "num_layers": 2,
                            "num_heads": 2,
                        },
                        "module_paths": {},
                    },
                    {
                        "architecture_key": "encoder_decoder",
                        "status": "loaded",
                        "model_id": "hf-internal-testing/tiny-random-t5",
                        "spec": {
                            "model_class": "T5ForConditionalGeneration",
                            "has_encoder": True,
                            "has_decoder": True,
                            "has_cross_attention": True,
                            "supports_past_key_values": True,
                            "vocab_size": 32100,
                            "hidden_size": 32,
                            "num_layers": 5,
                            "num_heads": 4,
                        },
                        "module_paths": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_workload(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "config": {"layers": 2},
                "calibration": {"flops_per_op": 1.0},
                "methods": {
                    "plain_hf_gpu": {
                        "implemented": True,
                        "online_boundary_calls": 0,
                        "boundary_calls_formula": "0 (no boundary)",
                        "online_trusted_compute_ops": 0,
                        "online_gpu_ops": 1000,
                        "measured_wall_time_ms": 3.8,
                        "wall_time_source": "measured",
                    },
                    "ours_current": {
                        "implemented": True,
                        "online_boundary_calls": 9,
                        "boundary_calls_formula": "4L+1 with L=2",
                        "online_trusted_compute_ops": 50,
                        "online_gpu_ops": 800,
                        "measured_wall_time_ms": 5.1,
                        "wall_time_source": "measured",
                    },
                },
                "paper_metrics": {"gpu_offload_ratio": 0.8},
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_summary_handles_all_missing_outputs(tmp_path) -> None:
    config = CrossArchitectureSummaryConfig(
        output_dir=str(tmp_path), require_existing_outputs=False
    )
    result = run_cross_architecture_summary(config)
    statuses = [a["status"] for a in result["architectures"]]
    assert statuses == ["missing", "missing", "missing"]
    assert result["workload"]["status"] == "missing"
    assert result["global_summary"]["num_missing"] == 3
    assert result["global_summary"]["all_architectures_allclose"] is None


def test_summary_raises_when_require_existing_and_all_missing(tmp_path) -> None:
    import pytest

    config = CrossArchitectureSummaryConfig(
        output_dir=str(tmp_path), require_existing_outputs=True
    )
    with pytest.raises(FileNotFoundError):
        run_cross_architecture_summary(config)


def test_summary_aggregates_synthetic_three_architectures(tmp_path) -> None:
    _write_decoder_attention(tmp_path / "attention_experiments.json")
    _write_encoder_attention(tmp_path / "encoder_attention_experiments.json")
    _write_cross_attention(tmp_path / "cross_attention_experiments.json")
    _write_coverage(tmp_path / "architecture_coverage.json")
    _write_workload(tmp_path / "workload_profile.json")

    config = CrossArchitectureSummaryConfig(output_dir=str(tmp_path))
    result = run_cross_architecture_summary(config)

    by_type = {a["architecture_type"]: a for a in result["architectures"]}
    assert by_type["decoder_only"]["status"] == "aggregated"
    assert by_type["encoder_only"]["status"] == "aggregated"
    assert by_type["encoder_decoder"]["status"] == "aggregated"
    assert by_type["decoder_only"]["num_cells"] == 1
    assert by_type["encoder_only"]["num_rows"] == 2  # 2 mask kinds
    assert by_type["encoder_decoder"]["num_rows"] == 2
    assert by_type["decoder_only"]["max_cache_error"] is not None
    assert by_type["encoder_only"]["max_cache_error"] is None
    assert by_type["encoder_decoder"]["max_cache_error"] is not None
    assert by_type["encoder_decoder"]["padding_mask_supported"] is True
    assert by_type["decoder_only"]["padding_mask_supported"] is False
    assert by_type["encoder_decoder"]["bias_present"] == {
        "q": False,
        "k": False,
        "v": False,
        "o": False,
    }
    assert result["workload"]["status"] == "loaded"
    assert result["workload"]["methods"][0]["method"] == "plain_hf_gpu"
    assert result["global_summary"]["num_aggregated"] == 3
    assert result["global_summary"]["all_architectures_allclose"] is True


def test_script_emits_all_three_artifacts(tmp_path) -> None:
    """End-to-end script smoke against the real ``outputs/`` directory."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    # No upstream JSONs in tmp_path → all architectures missing.
    for filename in (
        "cross_architecture_summary.json",
        "cross_architecture_summary.csv",
        "cross_architecture_summary.md",
    ):
        assert (tmp_path / filename).exists(), filename

    md = (tmp_path / "cross_architecture_summary.md").read_text(encoding="utf-8")
    assert "Cross-Architecture Summary" in md
    assert "Cross-architecture coverage table" in md
    assert "Limitations" in md
    assert "Next stage plan" in md
    assert "architectures=" in result.stdout

    # Sanity-check the CSV header includes the expected columns.
    csv_text = (tmp_path / "cross_architecture_summary.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    for col in ("architecture_type", "max_output_error", "trusted_shortcuts"):
        assert col in header


def test_script_works_with_real_outputs_dir() -> None:
    """The script must be runnable against the real outputs/ dir without crashing."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert "architectures=" in result.stdout
