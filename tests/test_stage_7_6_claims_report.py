"""Stage 7.6 -- bounded claims-report writer tests.

These guard against the multi-GB report regression: by default the
writers must emit compact reports (no full occurrence list, capped
examples, aggregate CSV) and a hard size guard must prevent any huge
file from being written. CPU-only, no network.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from pllo.experiments.stage_7_6_claims_consistency import (
    ClaimsReportConfig,
    build_compact_report,
    write_reports,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts" / "run_stage_7_6_claims_consistency.py"

_PAPER_SAFE = (
    "masked-gradient LoRA provides algebraic equivalence for "
    "SGD/Momentum under orthogonal masks and proxy-evaluated "
    "leakage mitigation; it does not provide formal, "
    "cryptographic, or semantic security."
)


def _synthetic_report(
    n_occ: int, *, classification: str = "listed_as_unsafe_wording_to_avoid",
) -> dict:
    """A report-shaped dict with ``n_occ`` synthetic occurrences. Mirrors
    the keys ``build_claims_consistency_report`` produces, without
    scanning the repo (cheap, deterministic, memory-modest)."""
    phrases = ("formal security", "semantic security", "cryptographically secure")
    occ = [
        {
            "file": f"docs/file_{i % 11}.md",
            "line": i + 1,
            "phrase": phrases[i % len(phrases)],
            "match": phrases[i % len(phrases)],
            "snippet": f"... context for occurrence {i} ...",
            "classification": classification,
        }
        for i in range(n_occ)
    ]
    unsafe = n_occ if classification == "unsafe_wording_present" else 0
    return {
        "status": "ok",
        "stage": "7.6",
        "report": "stage_7_6_claims_consistency",
        "tracked_phrases": list(phrases),
        "files_scanned": [f"docs/file_{i}.md" for i in range(11)],
        "files_scanned_count": 11,
        "occurrences": occ,
        "summary_by_phrase": {},
        "total_unsafe_wording_present": unsafe,
        "total_listed_as_unsafe_wording_to_avoid": n_occ - unsafe,
        "passes_consistency_check": unsafe == 0,
        "formal_security_claim": False,
        "honesty_phrases": [],
        "limitations": [],
        "paper_safe_wording": _PAPER_SAFE,
    }


# 1.
def test_claims_report_default_omits_full_occurrences(tmp_path: Path) -> None:
    rep = _synthetic_report(1000)
    j, _c, _m = write_reports(rep, outputs_dir=str(tmp_path))
    data = json.loads(Path(j).read_text())
    assert "full_occurrences" not in data
    assert data["summary"]["total_occurrences"] == 1000
    assert data["truncation"]["full_occurrences_included"] is False


# 2.
def test_claims_report_examples_are_capped(tmp_path: Path) -> None:
    rep = _synthetic_report(1000)
    cfg = ClaimsReportConfig(max_examples_per_category=25,
                             max_examples_per_file=1000)
    j, _c, _m = write_reports(rep, outputs_dir=str(tmp_path), config=cfg)
    data = json.loads(Path(j).read_text())
    exs = data["examples_by_category"]["listed_as_unsafe_wording_to_avoid"]
    assert len(exs) == 25
    assert data["truncation"]["examples_truncated"] is True


# 3.
def test_claims_report_full_occurrences_requires_flag(tmp_path: Path) -> None:
    rep = _synthetic_report(1000)
    cfg = ClaimsReportConfig(write_full_occurrences=True,
                             max_full_occurrences=500)
    j, c, _m = write_reports(rep, outputs_dir=str(tmp_path), config=cfg)
    data = json.loads(Path(j).read_text())
    assert "full_occurrences" in data
    assert len(data["full_occurrences"]) <= 500
    assert data["truncation"]["full_occurrences_truncated"] is True
    # separate, capped occurrences CSV is written alongside.
    occ_csv = Path(c.replace(".csv", "_occurrences.csv"))
    assert occ_csv.is_file()
    with occ_csv.open() as fh:
        rows = list(csv.reader(fh))
    assert len(rows) - 1 <= 500  # minus header


# 4.
def test_claims_report_csv_is_aggregate_by_default(tmp_path: Path) -> None:
    rep = _synthetic_report(5000)
    _j, c, _m = write_reports(rep, outputs_dir=str(tmp_path))
    with Path(c).open() as fh:
        rows = list(csv.reader(fh))
    # Bounded: summary + categories + top files/terms + capped examples,
    # never one row per occurrence.
    assert len(rows) < 500
    assert len(rows) < 5000


# 5.
def test_claims_report_size_guard(tmp_path: Path) -> None:
    rep = _synthetic_report(1000)
    cfg = ClaimsReportConfig(max_report_mb=0.00001)  # ~10 bytes
    j, _c, _m = write_reports(rep, outputs_dir=str(tmp_path), config=cfg)
    data = json.loads(Path(j).read_text())
    assert data["report_size_guard_triggered"] is True
    assert "examples_by_category" in data and data["examples_by_category"] == {}
    # Summary counts survive the guard.
    assert data["summary"]["total_occurrences"] == 1000
    # No huge file written.
    assert Path(j).stat().st_size < 1_000_000


# 6.
def test_claims_scanner_cli_defaults_compact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "# Project\n\nWe do not claim formal security.\n", encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(CLI),
         "--repo-root", str(repo), "--output-dir", str(out)],
        capture_output=True, text=True, check=True)
    assert "Wrote:" in proc.stdout
    jpath = out / "stage_7_6_claims_consistency.json"
    assert jpath.is_file()
    assert (out / "stage_7_6_claims_consistency.csv").is_file()
    assert (out / "stage_7_6_claims_consistency.md").is_file()
    data = json.loads(jpath.read_text())
    assert "full_occurrences" not in data
    assert data["report_size_guard_triggered"] is False
    # default => no separate occurrences CSV
    assert not (out / "stage_7_6_claims_consistency_occurrences.csv").exists()


# 7.
def test_no_multigb_report_regression(tmp_path: Path) -> None:
    # A large occurrence count must still produce a tiny report by
    # default. 200k entries is enough to prove bounding without huge RAM.
    rep = _synthetic_report(200_000)
    j, c, m = write_reports(rep, outputs_dir=str(tmp_path))
    assert json.loads(Path(j).read_text())["summary"][
        "total_occurrences"] == 200_000
    for p in (j, c, m):
        assert Path(p).stat().st_size < 5 * 1024 * 1024  # < 5 MB


def test_compact_report_passes_through_security_flag() -> None:
    rep = _synthetic_report(10)
    compact = build_compact_report(rep)
    assert compact["formal_security_claim"] is False
    assert compact["passes_consistency_check"] is True
