"""Stage 7.5 — tests for paper artifact consolidation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.paper_artifact_consolidation import (
    PaperArtifactConsolidationConfig,
    run_paper_artifact_consolidation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_paper_artifact_consolidation.py"


# ---------------------------------------------------------------------------
# 1. runs on existing outputs
# ---------------------------------------------------------------------------


def test_consolidation_runs(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    report = run_paper_artifact_consolidation(cfg)
    assert report["paper_artifact_consolidation_status"] == "implemented"
    # Stage 7.5 had 22 artifacts; Stage 7.5b adds 5 (CPU-only paper experiments).
    assert len(report["artifact_inventory"]) >= 27


def test_cpu_paper_summary_tables_emitted(tmp_path: Path) -> None:
    """Stage 7.5b -- the 5 CPU paper summary tables must be written."""
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    run_paper_artifact_consolidation(cfg)
    base = tmp_path / "paper_results"
    for slug in (
        "toy_task_summary",
        "baseline_comparison_summary",
        "ablation_summary",
        "stability_summary",
        "cpu_runtime_completion",
    ):
        assert (base / "csv" / f"{slug}.csv").exists(), f"missing csv/{slug}"
        assert (base / "markdown" / f"{slug}.md").exists(), f"missing markdown/{slug}"
        assert (base / "latex" / f"{slug}.tex").exists(), f"missing latex/{slug}"


# ---------------------------------------------------------------------------
# 2. artifact_inventory covers both inference and LoRA artifacts
# ---------------------------------------------------------------------------


def test_inventory_covers_both_slots(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    report = run_paper_artifact_consolidation(cfg)
    slots = {r["slot"] for r in report["artifact_inventory"]}
    # Stage 7.5b added the ``cpu_paper`` slot alongside the original
    # inference / lora slots.
    assert {"inference", "lora"}.issubset(slots)
    assert "cpu_paper" in slots


# ---------------------------------------------------------------------------
# 3-5. correctness / security / limitations summaries generated
# ---------------------------------------------------------------------------


def test_correctness_summary_present(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    report = run_paper_artifact_consolidation(cfg)
    assert report["correctness_summary"]
    # Stage 7.x components covered.
    stages = {r["stage"] for r in report["correctness_summary"]}
    for needle in ("7.0", "7.1", "7.2", "7.3", "7.4"):
        assert needle in stages, f"missing stage {needle}"


def test_security_proxy_summary_present(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    report = run_paper_artifact_consolidation(cfg)
    assert report["security_proxy_summary"]


def test_limitations_summary_present(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    report = run_paper_artifact_consolidation(cfg)
    assert report["limitations_summary"]


# ---------------------------------------------------------------------------
# 6. LaTeX table files generated
# ---------------------------------------------------------------------------


def test_latex_files_generated(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    run_paper_artifact_consolidation(cfg)
    paper_dir = tmp_path / "paper_results"
    for slug in (
        "artifact_inventory", "correctness_summary",
        "security_proxy_summary", "workload_summary",
        "lora_training_summary", "limitations_summary",
    ):
        tex_path = paper_dir / "latex" / f"{slug}.tex"
        md_path = paper_dir / "markdown" / f"{slug}.md"
        csv_path = paper_dir / "csv" / f"{slug}.csv"
        assert tex_path.exists(), tex_path
        assert md_path.exists(), md_path
        assert csv_path.exists(), csv_path
        assert "\\begin{tabular}" in tex_path.read_text()


# ---------------------------------------------------------------------------
# 7. strict=False does not fail when an artifact is missing
# ---------------------------------------------------------------------------


def test_missing_artifact_does_not_fail_when_not_strict(tmp_path: Path) -> None:
    fake_outputs = tmp_path / "outputs_empty"
    fake_outputs.mkdir()
    # Place only one of the expected artifacts.
    (fake_outputs / "lora_training_experiments.json").write_text(
        "{\"config\": {}, \"training_correctness\": {\"max_loss_diff\": 0.0}}",
        encoding="utf-8",
    )
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(fake_outputs),
        paper_results_dir=str(tmp_path / "paper_results"),
        strict=False,
    )
    report = run_paper_artifact_consolidation(cfg)
    assert report["missing_artifacts"], "expected missing artifacts list"
    statuses = {r["status"] for r in report["artifact_inventory"]}
    assert "missing" in statuses


def test_missing_artifact_fails_in_strict(tmp_path: Path) -> None:
    fake_outputs = tmp_path / "outputs_empty"
    fake_outputs.mkdir()
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(fake_outputs),
        paper_results_dir=str(tmp_path / "paper_results"),
        strict=True,
    )
    with pytest.raises(FileNotFoundError):
        run_paper_artifact_consolidation(cfg)


# ---------------------------------------------------------------------------
# 8. no raw tensor / mask / adapter / private data in outputs
# ---------------------------------------------------------------------------


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(PROJECT_ROOT / "outputs"),
        paper_results_dir=str(tmp_path / "paper_results"),
    )
    run_paper_artifact_consolidation(cfg)
    paper_dir = tmp_path / "paper_results"
    for sub in ("markdown", "csv", "latex", "json"):
        for path in (paper_dir / sub).glob("*"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "tensor(" not in text, path


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_runner_script_executes(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--outputs-dir", str(PROJECT_ROOT / "outputs"),
        "--paper-results-dir", str(tmp_path / "paper_results"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "paper_results" / "json" / "artifact_inventory.json").exists()
