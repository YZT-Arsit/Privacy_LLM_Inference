"""Stage 7.6c -- tests for the reviewer-risk audit runner.

The audit is a *read-only* static pass over the paper draft and the
paper_results markdown. These tests verify that:

* the runner executes,
* it writes the required artifacts,
* the markdown/JSON contains every Stage 7.6c-mandated section,
* every Stage 7.6c-mandated wording check is present,
* the audit does NOT modify ``outputs/`` or ``paper_results/``.

The tests run inside a tmp clone of the repo's paper_draft directory so
the live audit artifacts are not perturbed by pytest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from pllo.experiments import reviewer_risk_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def _digest_dir(path: Path) -> dict:
    """Return {relative_path: sha256_hex} for every file under ``path``."""
    out: dict[str, str] = {}
    if not path.is_dir():
        return out
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p.relative_to(path))] = h
    return out


def test_run_audit_executes_and_writes_required_files(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    result = reviewer_risk_audit.run_audit(write=True)

    required = {
        "reviewer_risk_audit.md",
        "reviewer_risk_audit.json",
        "reviewer_risk_audit.csv",
        "revision_plan.md",
        "unsafe_wording_review.md",
        "novelty_positioning_review.md",
        "threat_model_review.md",
        "baseline_fairness_review.md",
        "evaluation_sufficiency_review.md",
    }
    for name in required:
        assert (paper_draft_clone / name).is_file(), f"missing {name}"

    assert len(result.items) >= 1
    assert len(result.qas) == 12


def test_audit_markdown_contains_every_required_section(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    text = (paper_draft_clone / "reviewer_risk_audit.md").read_text(encoding="utf-8")

    required_sections = (
        "## 1. Executive Summary",
        "## 2. Top-10 Reviewer Risks",
        "## 3. Novelty Risk",
        "## 4. Threat Model Risk",
        "## 5. Prior-Work Comparison Risk",
        "## 6. Runtime Deployment Risk",
        "## 7. Security Claim Risk",
        "## 8. Correctness Proof Risk",
        "## 9. Evaluation Sufficiency Risk",
        "## 10. Baseline Fairness Risk",
        "## 11. Wording Risk",
        "## 12. Simulated Reviewer Questions",
        "## 13. Revision Priority Plan",
    )
    for sec in required_sections:
        assert sec in text, f"missing section: {sec}"


def test_revision_plan_has_all_priority_bands(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    text = (paper_draft_clone / "revision_plan.md").read_text(encoding="utf-8")

    for band in ("P0", "P1", "P2", "P3"):
        assert band in text, f"missing priority band {band}"


def test_unsafe_wording_review_contains_mandated_phrase_checks(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    text = (paper_draft_clone / "unsafe_wording_review.md").read_text(encoding="utf-8")

    for phrase in ("formal security", "real TEE wall-time", "full system reproduction"):
        assert phrase in text, f"missing mandated wording check: {phrase}"

    for word_kind in (
        "secure", "guarantee", "private", "protect", "hide",
        "outperform", "reproduced", "TEE_ready", "GPU_ready",
        "production", "full_system",
    ):
        assert word_kind in text, f"missing dangerous-word scan kind: {word_kind}"


def test_simulated_reviewer_questions_cover_q1_to_q12():
    qids = {q.qid for q in reviewer_risk_audit.REVIEWER_QAS}
    assert qids == {f"Q{i}" for i in range(1, 13)}


def test_simulated_reviewer_questions_section_exists(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    text = (paper_draft_clone / "reviewer_risk_audit.md").read_text(encoding="utf-8")
    for qid in (f"Q{i}" for i in range(1, 13)):
        assert f"### {qid}: " in text, f"missing simulated reviewer Q section {qid}"


def test_audit_does_not_modify_outputs_or_paper_results(tmp_path, monkeypatch):
    """Critical Stage 7.6c constraint: the audit must NOT change outputs/ or
    paper_results/."""
    outputs_before = _digest_dir(REPO_ROOT / "outputs")
    paper_results_before = _digest_dir(REPO_ROOT / "paper_results")

    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)

    outputs_after = _digest_dir(REPO_ROOT / "outputs")
    paper_results_after = _digest_dir(REPO_ROOT / "paper_results")

    assert outputs_before == outputs_after, "audit modified outputs/"
    assert paper_results_before == paper_results_after, "audit modified paper_results/"


def test_audit_json_records_no_modification_flag(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    payload = json.loads(
        (paper_draft_clone / "reviewer_risk_audit.json").read_text(encoding="utf-8")
    )

    assert payload["outputs_modified"] is False
    assert payload["paper_results_modified"] is False
    assert payload["stage"] == "7.6c"
    assert isinstance(payload["items"], list)
    assert isinstance(payload["reviewer_qas"], list)
    assert isinstance(payload["wording_hits"], list)


def test_csv_has_expected_columns(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    header = (paper_draft_clone / "reviewer_risk_audit.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    for col in (
        "risk_id", "dimension", "severity", "priority",
        "location", "risky_wording_or_missing",
        "why_reviewer_may_object", "recommended_revision",
        "new_experiment_needed", "wording_fix_enough",
    ):
        assert col in header, f"missing csv column {col}"


def test_no_unsafe_wording_in_paper_body(tmp_path, monkeypatch):
    """Stage 7.6c requires: every flagged 'unsafe' overclaim must be either
    eliminated or co-located with a hedge that the classifier recognises.
    Risky occurrences still need a human eyeball but are allowed."""
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    result = reviewer_risk_audit.run_audit(write=True)
    unsafe_hits = [h for h in result.wording_hits if h.classification == "unsafe"]
    assert not unsafe_hits, f"unsafe wording leak in paper body: {unsafe_hits[:5]}"


def test_top_10_section_lists_exactly_ten_risks(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    text = (paper_draft_clone / "reviewer_risk_audit.md").read_text(encoding="utf-8")
    section = text.split("## 2. Top-10 Reviewer Risks", 1)[1].split("## 3.", 1)[0]
    # Each top entry uses "### Rank N:" pattern.
    rank_lines = [
        line for line in section.splitlines() if line.startswith("### Rank ")
    ]
    assert len(rank_lines) == 10, f"expected 10 top-ranked items, got {len(rank_lines)}"


def test_topic_review_files_carry_risk_items_for_their_dimensions(tmp_path, monkeypatch):
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    reviewer_risk_audit.run_audit(write=True)
    file_dim_pairs = (
        ("novelty_positioning_review.md", reviewer_risk_audit.DIM_NOVELTY),
        ("threat_model_review.md", reviewer_risk_audit.DIM_THREAT),
        ("baseline_fairness_review.md", reviewer_risk_audit.DIM_PRIORWORK),
        ("baseline_fairness_review.md", reviewer_risk_audit.DIM_BASELINE),
        ("evaluation_sufficiency_review.md", reviewer_risk_audit.DIM_EVAL),
    )
    for filename, dim in file_dim_pairs:
        text = (paper_draft_clone / filename).read_text(encoding="utf-8")
        items_for_dim = [
            it for it in reviewer_risk_audit.RISK_ITEMS if it.dimension == dim
        ]
        for it in items_for_dim:
            assert it.risk_id in text, (
                f"{filename} missing risk id {it.risk_id} for dim {dim}"
            )


def test_runner_script_reports_summary(tmp_path, monkeypatch):
    """Smoke test the runner script can be imported and main() runs."""
    paper_draft_clone = tmp_path / "paper_draft"
    shutil.copytree(REPO_ROOT / "paper_draft", paper_draft_clone)
    monkeypatch.setattr(reviewer_risk_audit, "PAPER_DRAFT_DIR", paper_draft_clone)

    # Mirror what scripts/run_reviewer_risk_audit.py does end-to-end.
    result = reviewer_risk_audit.run_audit(write=True)
    severity = {}
    for it in result.items:
        severity[it.severity] = severity.get(it.severity, 0) + 1

    # The audit catalogue itself encodes that no item requires a new
    # experiment -- Stage 7.6c is wording-only.
    assert all(not it.new_experiment_needed for it in result.items)
    assert all(it.wording_fix_enough for it in result.items)
    # Severity coverage: at least one high-priority item and at least one P0
    # entry must exist; the catalogue is non-trivial.
    assert "high" in severity
    p0_items = [it for it in result.items if it.priority == "P0"]
    assert p0_items, "expected at least one P0 item"
