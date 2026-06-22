"""Stage 7.3 tests -- repo hygiene, size guard, evidence manifest.

Small, CPU-only, no transformers / CUDA / internet. Size-guard tests use
tiny files with zero thresholds so they never write large files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.repo_hygiene import (
    RECOMMENDED_GITIGNORE_ENTRIES,
    RepoHygieneConfig,
    check_output_sizes,
    ensure_gitignore_entries,
    generate_evidence_manifest,
    run_repo_hygiene_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
PY = sys.executable

_REQUIRED_GITIGNORE = (
    "__pycache__/", "*.py[cod]", "*.pyo", ".pytest_cache/", ".ruff_cache/",
    ".mypy_cache/", ".coverage", "htmlcov/", "build/", "dist/", "*.egg-info/",
    "outputs/paper_artifacts/", "outputs/paper_sections/",
)


def _init_git_repo(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    def g(*a):
        subprocess.run(["git", *a], cwd=str(root), check=True,
                       capture_output=True)
    g("init")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "Test")
    return True


# 1.
def test_recommended_gitignore_entries_present() -> None:
    for e in _REQUIRED_GITIGNORE:
        assert e in RECOMMENDED_GITIGNORE_ENTRIES


# 2.
def test_ensure_gitignore_entries_idempotent(tmp_path: Path) -> None:
    first = ensure_gitignore_entries(tmp_path, RECOMMENDED_GITIGNORE_ENTRIES)
    assert first["created"] is True
    assert set(first["added_entries"]) == set(RECOMMENDED_GITIGNORE_ENTRIES)
    second = ensure_gitignore_entries(tmp_path, RECOMMENDED_GITIGNORE_ENTRIES)
    assert second["added_entries"] == []
    assert set(second["already_present_entries"]) == set(
        RECOMMENDED_GITIGNORE_ENTRIES)
    # No duplicate lines were introduced.
    lines = [ln.strip() for ln in
             (tmp_path / ".gitignore").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert len(lines) == len(set(lines))


# 3.
def test_hygiene_audit_detects_tracked_generated_candidates(
    tmp_path: Path,
) -> None:
    if not _init_git_repo(tmp_path):
        pytest.skip("git unavailable")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n")
    cache = tmp_path / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-311.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "result.json").write_text("{}")
    subprocess.run(["git", "add", "-f", "pkg/mod.py",
                    "pkg/__pycache__/mod.cpython-311.pyc",
                    "outputs/result.json"],
                   cwd=str(tmp_path), check=True, capture_output=True)

    report = run_repo_hygiene_audit(RepoHygieneConfig(repo_root=str(tmp_path)))
    assert report["git_available"] is True
    tracked = report["tracked_generated_candidates"]
    assert any(p.endswith("mod.cpython-311.pyc") for p in tracked)
    assert "outputs/result.json" in tracked
    # Tracked artifacts are flagged for MANUAL decision, never auto-deleted.
    assert report["manual_decision_needed"]["count"] >= 2
    assert all(not c.startswith("rm") or "outputs/result.json" not in c
               for c in report["safe_cleanup_commands"])


# 4.
def test_hygiene_audit_detects_untracked_generated_candidates(
    tmp_path: Path,
) -> None:
    if not _init_git_repo(tmp_path):
        pytest.skip("git unavailable")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "untracked_probe.json").write_text("{}")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "foo.cpython-311.pyc").write_bytes(b"\x00")

    report = run_repo_hygiene_audit(RepoHygieneConfig(repo_root=str(tmp_path)))
    untracked = report["untracked_generated_candidates"]
    assert "outputs/untracked_probe.json" in untracked
    # safe cleanup collapses the cache dir and lists the untracked file.
    cmds = " ".join(report["safe_cleanup_commands"])
    assert "outputs/untracked_probe.json" in cmds
    assert "__pycache__" in cmds


# 5.
def test_output_size_guard_warns(tmp_path: Path) -> None:
    (tmp_path / "small.json").write_text('{"a": 1}')
    rep = check_output_sizes(tmp_path, warn_mb=0, fail_mb=100)
    assert len(rep["warnings"]) == 1
    assert rep["failures"] == []
    assert rep["passed"] is True


# 6.
def test_output_size_guard_fails(tmp_path: Path) -> None:
    (tmp_path / "big.json").write_text('{"a": 1}')
    rep = check_output_sizes(tmp_path, warn_mb=0, fail_mb=0)
    assert len(rep["failures"]) == 1
    assert rep["passed"] is False
    # checkpoints are excluded from the guard
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 16)
    rep2 = check_output_sizes(tmp_path, warn_mb=0, fail_mb=0)
    assert all("safetensors" not in f["path"] for f in rep2["failures"])


# 7.
def test_evidence_manifest_contains_required_stages(tmp_path: Path) -> None:
    manifest = generate_evidence_manifest(output_dir=str(tmp_path))
    stages = {s["stage"] for s in manifest["stages"]}
    for required in ("6.4", "6.4.1", "6.5", "6.6", "6.7", "6.8", "6.9",
                     "7.0", "7.1", "7.6_scanner_fix"):
        assert required in stages, required
    assert (tmp_path / "evidence_manifest.json").is_file()
    assert (tmp_path / "evidence_manifest.md").is_file()


# 8.
def test_evidence_manifest_contains_global_caveats(tmp_path: Path) -> None:
    manifest = generate_evidence_manifest(output_dir=str(tmp_path))
    caveats = " ".join(manifest["global_caveats"]).lower()
    assert "no semantic, cryptographic, or formal security" in caveats
    assert "attention scores" in caveats
    assert "vocab permutation+scaling is weaker" in caveats
    md = (tmp_path / "evidence_manifest.md").read_text().lower()
    assert "global caveats" in md


# 9.
def test_evidence_manifest_does_not_include_large_payloads(
    tmp_path: Path,
) -> None:
    generate_evidence_manifest(output_dir=str(tmp_path))
    jtext = (tmp_path / "evidence_manifest.json").read_text()
    assert "tensor(" not in jtext
    import re
    assert re.search(r"(-?\d+\.\d+\s*,\s*){50,}", jtext) is None
    assert (tmp_path / "evidence_manifest.json").stat().st_size < 100_000
    assert (tmp_path / "evidence_manifest.md").stat().st_size < 100_000


# 10.
def test_lightweight_repro_script_exists() -> None:
    assert (SCRIPTS / "run_lightweight_repro.py").is_file()
    # it is importable / parseable (syntax check, no execution of subprocs)
    proc = subprocess.run([PY, str(SCRIPTS / "run_lightweight_repro.py"),
                           "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "lightweight" in proc.stdout.lower()


# 11.
def test_repo_hygiene_cli_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    proc = subprocess.run(
        [PY, str(SCRIPTS / "run_repo_hygiene_audit.py"),
         "--repo-root", str(REPO_ROOT), "--output", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    assert out.with_suffix(".md").is_file()
    report = json.loads(out.read_text())
    assert report["stage"] == "7.3_repo_hygiene"
    assert "recommended_gitignore_entries" in report


# 12.
def test_check_output_sizes_cli_exit_code(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text('{"a": 1}')
    # fail-mb 0 -> any file fails -> exit 1
    fail = subprocess.run(
        [PY, str(SCRIPTS / "check_output_sizes.py"),
         "--output-dir", str(tmp_path), "--warn-mb", "0", "--fail-mb", "0"],
        capture_output=True, text=True)
    assert fail.returncode == 1
    # generous thresholds -> exit 0
    ok = subprocess.run(
        [PY, str(SCRIPTS / "check_output_sizes.py"),
         "--output-dir", str(tmp_path), "--warn-mb", "10", "--fail-mb", "100"],
        capture_output=True, text=True)
    assert ok.returncode == 0
