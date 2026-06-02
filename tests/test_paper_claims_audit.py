"""Stage 7.5 — tests for paper claims audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.paper_claims_audit import (
    PaperClaimsAuditConfig,
    run_paper_claims_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_paper_claims_audit.py"


# ---------------------------------------------------------------------------
# 1. JSON / MD / TeX generated
# ---------------------------------------------------------------------------


def test_audit_generates_artifacts(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    base = tmp_path / "paper_results"
    assert (base / "json" / "paper_claims_audit.json").exists()
    assert (base / "markdown" / "paper_claims_audit.md").exists()
    assert (base / "latex" / "paper_claims_audit.tex").exists()
    assert report["paper_claims_audit_status"] == "implemented"


# ---------------------------------------------------------------------------
# 2. supported / proxy_supported / unsupported buckets present
# ---------------------------------------------------------------------------


def test_three_buckets_present(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    statuses = {r["status"] for r in report["claims"]}
    assert "supported" in statuses
    assert "proxy_supported" in statuses
    assert "unsupported" in statuses


# ---------------------------------------------------------------------------
# 3. unsupported list contains formal security
# ---------------------------------------------------------------------------


def test_unsupported_includes_formal_security(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    claims_text = " ".join(
        r["claim"].lower() for r in report["claims"]
        if r["status"] == "unsupported"
    )
    assert "formal" in claims_text


# ---------------------------------------------------------------------------
# 4. unsupported list contains real TEE wall-time
# ---------------------------------------------------------------------------


def test_unsupported_includes_real_tee_wall_time(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    claims_text = " ".join(
        r["claim"].lower() for r in report["claims"]
        if r["status"] == "unsupported"
    )
    assert "real tee" in claims_text


# ---------------------------------------------------------------------------
# 5. unsupported list contains padded_rank hidden
# ---------------------------------------------------------------------------


def test_unsupported_includes_padded_rank_hidden(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    claims_text = " ".join(
        r["claim"].lower() for r in report["claims"]
        if r["status"] == "unsupported"
    )
    assert "padded_rank is hidden" in claims_text


# ---------------------------------------------------------------------------
# 6. every claim has a safe wording
# ---------------------------------------------------------------------------


def test_every_claim_has_safe_wording(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    for r in report["claims"]:
        assert r["paper_safe_wording"]


# ---------------------------------------------------------------------------
# 7. unsafe wording is present
# ---------------------------------------------------------------------------


def test_unsafe_wording_present(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    for r in report["claims"]:
        assert r["unsafe_wording_to_avoid"]


# ---------------------------------------------------------------------------
# 8. audit does not overclaim — unsupported items must have safe wording
#    that disclaims the property
# ---------------------------------------------------------------------------


def test_audit_does_not_overclaim(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    report = run_paper_claims_audit(cfg)
    for r in report["claims"]:
        if r["status"] == "unsupported":
            # No evidence artifacts should be referenced.
            assert r["evidence_artifacts"] == [] or r["evidence_artifacts"] is None
            # The safe wording must NOT say "secure", "provable", or "guarantee".
            unsafe_terms = ("provable", "guaranteed", "cryptographically secure")
            wording = r["paper_safe_wording"].lower()
            for term in unsafe_terms:
                assert term not in wording


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_runner_script_executes(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--paper-results-dir", str(tmp_path / "paper_results"),
        "--outputs-dir", str(PROJECT_ROOT / "outputs"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "paper_results" / "json" / "paper_claims_audit.json").exists()


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(tmp_path / "paper_results"),
        outputs_dir=str(PROJECT_ROOT / "outputs"),
    )
    run_paper_claims_audit(cfg)
    for sub in ("json", "markdown", "latex"):
        for path in (tmp_path / "paper_results" / sub).glob("*"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "tensor(" not in text, path
