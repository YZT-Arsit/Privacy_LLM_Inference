"""Tests for the Stage 5.2 nonlinear island probe runner + script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pllo.experiments import (
    NonlinearIslandProbeConfig,
    run_nonlinear_island_experiments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_nonlinear_island_experiments.py"


def test_run_emits_28_cells_all_allclose() -> None:
    payload = run_nonlinear_island_experiments(NonlinearIslandProbeConfig())
    g = payload["global_summary"]
    assert g["num_cells"] >= 28
    assert g["all_allclose"] is True
    assert g["max_online_extra_matmul_count"] == 0


def test_norm_cells_carry_orthogonality_diagnostics() -> None:
    payload = run_nonlinear_island_experiments(NonlinearIslandProbeConfig())
    for cell in payload["norm_island_cells"]:
        assert cell["orthogonality_error"] is not None
        if cell["island"] == "layernorm_mean_preserving_affine_fold":
            assert cell["mean_preservation_error"] is not None


def test_mlp_cells_report_zero_online_extra_matmul() -> None:
    payload = run_nonlinear_island_experiments(NonlinearIslandProbeConfig())
    for cell in payload["mlp_island_cells"]:
        assert cell["online_extra_matmul_count"] == 0
        assert cell["metrics"]["allclose"] is True


def test_script_emits_all_three_artifacts(tmp_path) -> None:
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
    for filename in (
        "nonlinear_island_experiments.json",
        "nonlinear_island_experiments.csv",
        "nonlinear_island_experiments.md",
    ):
        assert (tmp_path / filename).exists(), filename

    md = (tmp_path / "nonlinear_island_experiments.md").read_text(encoding="utf-8")
    # Spec-mandated headers / phrases.
    assert "Operator-Compatible Mask Families" in md
    assert "Pad Placement Rule" in md
    assert "Compatible mask families are weaker than unrestricted dense masks" in md
    assert (
        "Permutation islands hide channel identity but do not hide"
        " coordinate-value multisets"
    ) in md
    # All required sections.
    for section in (
        "Experiment scope",
        "Norm-Compatible Island Results",
        "Affine Folding Results",
        "Activation Permutation Island Results",
        "SwiGLU Paired-Permutation Island Results",
        "Full MLP Island Results",
        "Online Cost Interpretation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert section in md, f"missing section: {section}"

    payload = json.loads(
        (tmp_path / "nonlinear_island_experiments.json").read_text(encoding="utf-8")
    )
    assert payload["global_summary"]["all_allclose"] is True
    assert payload["global_summary"]["max_online_extra_matmul_count"] == 0
    assert "all_allclose=True" in result.stdout
