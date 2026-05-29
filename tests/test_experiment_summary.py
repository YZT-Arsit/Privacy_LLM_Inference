"""Tests for the Stage 4.10 experiment summary aggregator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_experiment_summary.py"


@pytest.fixture(scope="module")
def aggregator():
    spec = importlib.util.spec_from_file_location("experiment_summary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_summary"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_metrics_from_static_payload(aggregator) -> None:
    stage = next(s for s in aggregator.STAGES if s.stage_id == "1")
    payload = {
        "config": {"use_pad": True},
        "metrics": {
            "max_abs_error": 1e-14,
            "mean_abs_error": 3e-15,
            "relative_l2_error": 8e-16,
            "cosine_similarity": 1.0,
            "allclose": True,
        },
    }
    metrics = aggregator.extract_metrics(stage, payload)
    assert metrics["max_abs_error"] == pytest.approx(1e-14)
    assert metrics["allclose"] is True
    assert "top1_match_rate" not in metrics


def test_extract_metrics_from_cache_payload(aggregator) -> None:
    stage = next(s for s in aggregator.STAGES if s.stage_id == "4.8")
    payload = {
        "prefill_logits_metrics": {
            "max_abs_error": 7e-8,
            "allclose": True,
            "top1_match_rate": 1.0,
        },
        "decode_logits_metrics": {
            "max_abs_error_max": 5e-8,
            "allclose_all": True,
            "top1_match_rate_min": 1.0,
        },
        "cache_invariant_metrics": {
            "max_key_error": 7e-9,
            "max_value_error": 1e-8,
            "allclose": True,
        },
    }
    metrics = aggregator.extract_metrics(stage, payload)
    assert metrics["prefill_max_abs_error"] == pytest.approx(7e-8)
    assert metrics["decode_max_abs_error_max"] == pytest.approx(5e-8)
    assert metrics["decode_allclose_all"] is True
    assert metrics["cache_allclose"] is True


def test_extract_metrics_from_generation_payload(aggregator) -> None:
    stage = next(s for s in aggregator.STAGES if s.stage_id == "4.9")
    payload = {
        "generation_metrics": {
            "token_match_rate": 1.0,
            "sequence_exact_match": 1.0,
            "top1_match_rate": 1.0,
        },
        "logits_metrics": {
            "max_abs_error_max": 4e-7,
            "allclose_all": True,
            "top1_match_rate_min": 1.0,
            "per_step": [],
        },
        "cache_invariant_metrics": {
            "max_key_error": 6e-9,
            "max_value_error": 1e-8,
            "allclose": True,
        },
    }
    metrics = aggregator.extract_metrics(stage, payload)
    assert metrics["token_match_rate"] == 1.0
    assert metrics["sequence_exact_match"] == 1.0
    assert metrics["logits_allclose_all"] is True
    assert metrics["cache_allclose"] is True


def test_collect_stage_handles_missing_snapshot(aggregator, tmp_path) -> None:
    stage = next(s for s in aggregator.STAGES if s.stage_id == "1")
    # Build a stage whose snapshot paths point at nonexistent files.
    sentinel_stage = aggregator.StageSpec(
        stage_id="test",
        title="missing",
        summary="",
        trusted_shortcuts=(),
        script=None,
        has_pad_variants=True,
        snapshot_true=tmp_path / "missing_true.json",
        snapshot_false=tmp_path / "missing_false.json",
        pad_flag_style="boolean",
    )
    record = aggregator.collect_stage(sentinel_stage, use_rerun=False)
    assert record["stage"] == "test"
    for variant in ("use_pad=true", "use_pad=false"):
        assert record["variants"][variant]["payload_present"] is False


def test_to_markdown_and_csv_render(aggregator) -> None:
    records = [
        {
            "stage": "1",
            "title": "Static",
            "summary": "static linear",
            "trusted_shortcuts": ["mask"],
            "script": "scripts/run_static_correctness.py",
            "has_pad_variants": True,
            "variants": {
                "use_pad=true": {
                    "source": "outputs/static_correctness.json",
                    "payload_present": True,
                    "metrics": {"max_abs_error": 1e-14, "allclose": True},
                },
                "use_pad=false": {
                    "source": "outputs/static_correctness_no_pad_float32.json",
                    "payload_present": True,
                    "metrics": {"max_abs_error": 2e-14, "allclose": True},
                },
            },
        }
    ]
    md = aggregator.to_markdown(records, rerun=False)
    assert "Stage 1 — Static" in md
    assert "use_pad=true" in md and "use_pad=false" in md
    assert "max_abs_error" in md
    rows = aggregator.to_csv_rows(records)
    assert len(rows) == 2
    variants = {row["variant"] for row in rows}
    assert variants == {"use_pad=true", "use_pad=false"}
    for row in rows:
        assert row["allclose"] is True


def test_summary_outputs_exist_after_snapshot_run(aggregator, tmp_path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    assert (tmp_path / "experiment_summary.json").exists()
    assert (tmp_path / "experiment_summary.csv").exists()
    assert (tmp_path / "experiment_summary.md").exists()
    # Markdown must include the section header.
    md = (tmp_path / "experiment_summary.md").read_text(encoding="utf-8")
    assert "Stage Coverage" in md
    assert "Trusted-side Engineering Shortcuts" in md
    assert "Reproducibility" in md
    assert "stages_recorded" in result.stdout
