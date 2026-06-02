"""Stage 7.2 — tests for the LoRA rank security proxy."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_rank_security_proxy import (
    LoRARankSecurityProxyConfig,
    run_lora_rank_security_proxy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_rank_security_proxy.py"


@pytest.fixture(scope="module")
def report_paired() -> dict:
    return run_lora_rank_security_proxy(
        LoRARankSecurityProxyConfig(
            seed=2026, d_in=16, d_out=8, true_ranks=(2, 4),
            padded_rank=8, num_trials=8, membership_trials_per_sample=3,
            dummy_strategy="paired_cancellation_dummy", dtype="float64",
        )
    )


@pytest.fixture(scope="module")
def report_zero() -> dict:
    return run_lora_rank_security_proxy(
        LoRARankSecurityProxyConfig(
            seed=2026, d_in=16, d_out=8, true_ranks=(2, 4),
            padded_rank=8, num_trials=8, membership_trials_per_sample=3,
            dummy_strategy="zero_dummy", dtype="float64",
        )
    )


# ---------------------------------------------------------------------------
# 1. run_lora_rank_security_proxy small config runs
# ---------------------------------------------------------------------------


def test_run_lora_rank_security_proxy_runs(report_paired: dict) -> None:
    assert report_paired["lora_rank_security_proxy_status"] == "implemented"


# ---------------------------------------------------------------------------
# 2. shape-level leakage section exists
# ---------------------------------------------------------------------------


def test_shape_level_leakage_section_present(report_paired: dict) -> None:
    shape = report_paired["shape_level_rank_leakage"]
    assert "no_padding" in shape
    assert "rank_padding" in shape


# ---------------------------------------------------------------------------
# 3. no_padding exposes true rank
# ---------------------------------------------------------------------------


def test_no_padding_exposes_true_rank(report_paired: dict) -> None:
    for entry in report_paired["shape_level_rank_leakage"]["no_padding"]:
        assert entry["exposed_rank_value"] == entry["true_rank"]
        assert entry["true_rank_hidden_from_shape"] is False


# ---------------------------------------------------------------------------
# 4. rank_padding hides true rank from shape
# ---------------------------------------------------------------------------


def test_rank_padding_hides_true_rank_from_shape(report_paired: dict) -> None:
    padded_rank = report_paired["config"]["padded_rank"]
    for entry in report_paired["shape_level_rank_leakage"]["rank_padding"]:
        assert entry["exposed_rank_value"] == padded_rank
        assert entry["true_rank_hidden_from_shape"] is True


# ---------------------------------------------------------------------------
# 5. spectral rank inference section exists
# ---------------------------------------------------------------------------


def test_spectral_rank_inference_section_present(report_paired: dict) -> None:
    section = report_paired["spectral_rank_inference"]
    assert "rows" in section
    assert len(section["rows"]) == len(report_paired["config"]["true_ranks"])
    for row in section["rows"]:
        assert "no_padding" in row
        assert "rank_padding" in row
        assert "risk_level" in row["rank_padding"]


# ---------------------------------------------------------------------------
# 6. gradient rank inference section exists
# ---------------------------------------------------------------------------


def test_gradient_rank_inference_section_present(report_paired: dict) -> None:
    section = report_paired["gradient_rank_inference"]
    assert "rows" in section
    assert len(section["rows"]) == len(report_paired["config"]["true_ranks"])
    for row in section["rows"]:
        assert "risk_level" in row
        assert "rank_inference_accuracy" in row


# ---------------------------------------------------------------------------
# 7. risk_level reported conservatively
# ---------------------------------------------------------------------------


def test_zero_dummy_reported_as_high_risk(report_zero: dict) -> None:
    for row in report_zero["spectral_rank_inference"]["rows"]:
        assert row["rank_padding"]["risk_level"] == "high"


def test_paired_dummy_not_low_risk(report_paired: dict) -> None:
    # paired_cancellation should NOT be reported as "low" — needs_more_evaluation
    # or medium/high is acceptable. Per constraint 12 we don't overclaim.
    for row in report_paired["spectral_rank_inference"]["rows"]:
        assert row["rank_padding"]["risk_level"] != "low"


# ---------------------------------------------------------------------------
# 8. JSON / CSV / Markdown generated
# ---------------------------------------------------------------------------


def test_runner_script_generates_artifacts(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--true-ranks", "2", "4",
            "--padded-rank", "8",
            "--num-trials", "4",
            "--membership-trials-per-sample", "3",
            "--d-in", "16", "--d-out", "8",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "lora_rank_security_proxy.md").read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Threat Model",
        "Shape-Level Rank Leakage",
        "Spectral Rank Inference Proxy",
        "Gradient Rank Inference Proxy",
        "Membership / Linkability Proxy",
        "Interpretation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert phrase in md, f"missing markdown section: {phrase!r}"
    md_lower = md.lower()
    assert "padded rank r_pad remains visible" in md_lower
    assert "not formal" in md_lower
    with (tmp_path / "lora_rank_security_proxy.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert "shape_level_rank_leakage" in sections
    assert "spectral_rank_inference" in sections
    assert "gradient_rank_inference" in sections
    assert "membership_linkability" in sections


# ---------------------------------------------------------------------------
# 9. no raw adapter / raw gradient / private data / mask in outputs
# ---------------------------------------------------------------------------


def test_no_raw_tensors_in_outputs(report_paired: dict) -> None:
    text = json.dumps(report_paired, default=str)
    assert "tensor(" not in text


def test_security_profile_unchanged(report_paired: dict) -> None:
    assert report_paired["security_profile"] == "proxy-evaluated, not formal"
    assert (
        report_paired["security_profile_detail_with_lora_rank_padding"]
        == "rank-padding-proxy-evaluated, not formal"
    )


def test_limitations_explicit(report_paired: dict) -> None:
    lims = " ".join(report_paired["limitations"]).lower()
    assert "not formal" in lims
    assert "no real tee" in lims
    assert "padded rank r_pad remains visible" in lims
    assert "zero_dummy" in lims  # explicit acknowledgement that zero_dummy leaks
