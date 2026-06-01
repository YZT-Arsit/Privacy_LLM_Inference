"""Stage 7.1 — tests for the LoRA gradient security proxy."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_gradient_security_proxy import (
    LoRAGradientSecurityProxyConfig,
    VALID_STRATEGIES,
    run_lora_gradient_security_proxy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_gradient_security_proxy.py"


@pytest.fixture(scope="module")
def report() -> dict:
    return run_lora_gradient_security_proxy(
        LoRAGradientSecurityProxyConfig(
            seed=2026, d_in=16, d_out=8, rank=3, alpha=1.0,
            num_trials=8, membership_trials_per_sample=4, dtype="float64",
        )
    )


# ---------------------------------------------------------------------------
# 1. run_lora_gradient_security_proxy small config runs
# ---------------------------------------------------------------------------


def test_run_lora_gradient_security_proxy_runs(report: dict) -> None:
    assert report["lora_gradient_security_proxy_status"] == "implemented"
    assert set(report["strategies"]) == set(VALID_STRATEGIES)


# ---------------------------------------------------------------------------
# 2. gradient extraction section exists
# ---------------------------------------------------------------------------


def test_gradient_extraction_section_present(report: dict) -> None:
    section = report["gradient_extraction_proxy"]
    assert isinstance(section, list)
    assert len(section) == len(VALID_STRATEGIES)
    baseline = next(
        e for e in section if e["strategy"] == "unmasked_gradient_baseline"
    )
    assert baseline["grad_a_recovery_rel_l2_mean"] < 1e-6
    assert baseline["grad_b_recovery_rel_l2_mean"] < 1e-6
    strong = next(
        e for e in section
        if e["strategy"] == "fresh_masks_fresh_u_with_pad"
    )
    assert strong["grad_a_recovery_rel_l2_mean"] > 0.5
    assert strong["grad_b_recovery_rel_l2_mean"] > 0.5


# ---------------------------------------------------------------------------
# 3. gradient membership section exists
# ---------------------------------------------------------------------------


def test_membership_section_present(report: dict) -> None:
    section = report["gradient_membership_style_linkability_proxy"]
    assert len(section) == len(VALID_STRATEGIES)
    seen = {m["strategy"] for m in section}
    assert seen == set(VALID_STRATEGIES)


# ---------------------------------------------------------------------------
# 4. gradient leakage accounting exists
# ---------------------------------------------------------------------------


def test_gradient_leakage_accounting_present(report: dict) -> None:
    table = report["gradient_leakage_accounting"]
    for s in VALID_STRATEGIES:
        assert s in table
    rows = table["fresh_masks_fresh_u_with_pad"]
    names = {r["name"] for r in rows}
    assert "grad_A / grad_B (plain)" in names
    assert "G_tilde (masked upstream gradient)" in names
    assert "grad_A_tilde / grad_B_tilde" in names
    assert "optimizer_state (SGD momentum / AdamW m, v)" in names
    grad_plain = next(
        r for r in rows if r["name"] == "grad_A / grad_B (plain)"
    )
    assert grad_plain["visibility"] == "trusted"
    grad_masked = next(
        r for r in rows if r["name"] == "grad_A_tilde / grad_B_tilde"
    )
    assert grad_masked["visibility"] == "gpu"


# ---------------------------------------------------------------------------
# 5. fresh masks reduce gradient linkability (or report conservative)
# ---------------------------------------------------------------------------


def test_fresh_masks_reduce_gradient_linkability_or_report_conservative(report: dict) -> None:
    section = report["gradient_membership_style_linkability_proxy"]
    fixed = next(m for m in section if m["strategy"] == "fixed_masks_fixed_u")
    fresh = next(m for m in section if m["strategy"] == "fresh_masks_fresh_u")
    fresh_pad = next(
        m for m in section if m["strategy"] == "fresh_masks_fresh_u_with_pad"
    )
    interp = report["interpretation"]["linkability_summary"]
    if fresh["membership_gradient_auc_proxy"] < fixed["membership_gradient_auc_proxy"] - 0.10:
        assert "reduce" in interp.lower()
    elif fresh_pad["membership_gradient_auc_proxy"] < fixed["membership_gradient_auc_proxy"] - 0.10:
        assert "reduce" in interp.lower()
    else:
        assert (
            "needs_more_evaluation" in interp.lower()
            or "did not" in interp.lower()
        )


# ---------------------------------------------------------------------------
# 6. rank leakage from gradient shape is explicitly reported
# ---------------------------------------------------------------------------


def test_rank_leakage_from_gradient_shape_reported(report: dict) -> None:
    for e in report["gradient_extraction_proxy"]:
        assert e["rank_visible_from_grad_shape"] is True
        assert e["configured_rank"] >= 1
        assert e["rank_signature_a"] >= 1
    note = report["interpretation"]["rank_visibility_note"]
    assert "rank" in note.lower()
    assert "grad_a_tilde" in note.lower() or "gradient" in note.lower()


# ---------------------------------------------------------------------------
# 7. JSON / CSV / Markdown generated
# ---------------------------------------------------------------------------


def test_runner_script_generates_artifacts(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-trials", "4",
            "--membership-trials-per-sample", "3",
            "--d-in", "16", "--d-out", "8", "--rank", "3",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "lora_gradient_security_proxy.md").read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Threat Model",
        "Gradient Extraction Proxy",
        "Gradient Membership-Style Linkability",
        "Gradient Leakage Accounting",
        "Interpretation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert phrase in md, f"missing markdown section: {phrase!r}"
    with (tmp_path / "lora_gradient_security_proxy.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert "gradient_extraction" in sections
    assert "gradient_leakage" in sections
    assert "gradient_membership_linkability" in sections


# ---------------------------------------------------------------------------
# 8. no raw gradients / adapter / private data / mask in outputs
# ---------------------------------------------------------------------------


def test_outputs_do_not_expose_raw_data(report: dict) -> None:
    text = json.dumps(report, default=str)
    assert "tensor(" not in text


def test_security_profile_unchanged(report: dict) -> None:
    assert report["security_profile"] == "proxy-evaluated, not formal"
    assert (
        report["security_profile_detail_with_lora_backward"]
        == "masked-gradient-proxy-evaluated, not formal"
    )


def test_limitations_explicit(report: dict) -> None:
    lims = " ".join(report["limitations"]).lower()
    assert "not formal" in lims
    assert "no real tee" in lims
    assert "rank" in lims
    assert "optimizer" in lims
