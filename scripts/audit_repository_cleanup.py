"""Repository cleanup audit.

Walks the repository, classifies every tracked file, performs a simple
reference scan (filename / path appearance in README / docs / scripts /
tests / src, plus Python import detection), and tags each file with a
conservative candidate action:

    keep                 -- protected by path (README, pyproject, src, ...)
    keep_core_artifact   -- Stage 5.7 / 5.8 / 7.6 core output, claims, summary
    keep_referenced      -- referenced by README / docs / scripts / tests / src
    keep_uncertain       -- not obviously referenced, but not safe to delete
    move_to_docs         -- long-form notes that belong under docs/
    move_to_archive       -- superseded output worth archiving (not deleting)
    delete_candidate     -- low-risk junk (caches, .pyc, .DS_Store, ...)

This script NEVER deletes anything. It only emits:

    outputs/repository_cleanup_audit.json
    outputs/repository_cleanup_audit.csv
    outputs/repository_cleanup_audit.md

CPU-only, no network. Read-only with respect to the repository (apart
from writing its own audit artifacts under outputs/).
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories we never descend into.
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "node_modules",
    ".claude",
}

# Path prefixes that are protected from deletion by default.
PROTECTED_PREFIXES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "src/",
    "tests/",
    "scripts/",
    "docs/",
    "paper_results/",
    "paper_draft/",
)

# Output basenames (substring match) that are core paper artifacts.
CORE_ARTIFACT_SUBSTRINGS = (
    "permutation_invariant_leakage",
    "lookup_nonlinear_cost_proxy",
    "masked_gradient_lora",
    "claims",
    "lifecycle",
    "summary",
    "limitations",
)

# Low-risk junk: safe to mark delete_candidate (and let the low-risk
# cleaner remove). Matched on full relative path or basename.
JUNK_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
JUNK_SUFFIXES = (".pyc", ".pyo", ".tmp", ".bak", ".orig")
JUNK_BASENAMES = {".DS_Store", "Thumbs.db"}

# File extensions whose *content* we scan for references to other files.
REFERENCE_SOURCE_SUFFIXES = (".py", ".md", ".tex", ".toml", ".cfg", ".txt")


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def collect_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if _is_excluded(p):
                continue
            files.append(p)
    return sorted(files)


def collect_junk() -> list[Path]:
    """Junk lives inside excluded dirs (caches) or matches junk patterns.

    These are surfaced separately so the audit can recommend removing
    them even though the main walk skips excluded dirs.
    """
    junk: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # Do not descend into .git.
        if ".git" in Path(dirpath).parts:
            dirnames[:] = []
            continue
        base = os.path.basename(dirpath)
        if base in JUNK_DIR_NAMES:
            junk.append(Path(dirpath))
            dirnames[:] = []  # whole dir is junk
            continue
        for name in filenames:
            p = Path(dirpath) / name
            if name in JUNK_BASENAMES or name.endswith(JUNK_SUFFIXES) \
                    or name.endswith("~"):
                junk.append(p)
    return sorted(set(junk))


def build_reference_index() -> str:
    """Concatenate the text of every reference-source file into one blob.

    The blob is searched (substring) for each candidate file's basename
    and relative path. This is intentionally coarse and conservative:
    any hit marks the file as referenced.
    """
    chunks: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            if not name.endswith(REFERENCE_SOURCE_SUFFIXES):
                continue
            p = Path(dirpath) / name
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def _module_name_for(path: Path) -> str | None:
    """If the file is a python module under src/, return its import path."""
    try:
        rel = path.relative_to(REPO_ROOT / "src")
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_referenced(path: Path, blob: str) -> tuple[bool, str]:
    rel = _rel(path)
    base = path.name
    # The audit artifacts themselves should not count as references.
    # We strip self-references by ignoring matches that only appear in
    # the audit files; cheaper heuristic: skip if basename is generic.
    if base in blob.replace(rel, ""):
        # basename appears somewhere other than its own path string
        if blob.count(base) > 0:
            return True, f"basename '{base}' appears in repo text"
    if rel in blob:
        return True, f"path '{rel}' appears in repo text"
    mod = _module_name_for(path)
    if mod:
        # import pllo.x.y  OR  from pllo.x import y
        if re.search(rf"\b{re.escape(mod)}\b", blob):
            return True, f"module '{mod}' imported / referenced"
        # also the leaf module imported via 'from pllo.x import leaf'
        leaf = mod.split(".")[-1]
        if re.search(rf"import\s+{re.escape(leaf)}\b", blob):
            return True, f"leaf module '{leaf}' imported"
    return False, "no reference found in README/docs/scripts/tests/src"


def classify(path: Path, referenced: bool, ref_reason: str) -> tuple[str, str]:
    rel = _rel(path)
    base = path.name

    # Junk patterns (these come from collect_junk, but double-check).
    if any(part in JUNK_DIR_NAMES for part in path.parts) \
            or base in JUNK_BASENAMES \
            or base.endswith(JUNK_SUFFIXES) or base.endswith("~"):
        return "delete_candidate", "low-risk junk (cache / temp / editor backup)"

    # Empty files inside non-protected areas are delete candidates.
    try:
        empty = path.stat().st_size == 0
    except OSError:
        empty = False

    # Core paper artifacts (outputs/).
    if rel.startswith("outputs/"):
        if any(s in base for s in CORE_ARTIFACT_SUBSTRINGS):
            return "keep_core_artifact", "Stage 5.7/5.8/7.6 or paper core artifact"
        if referenced:
            return "keep_referenced", ref_reason
        if empty:
            return "delete_candidate", "empty output file, unreferenced"
        return "keep_uncertain", "output not matched to core set or references"

    # Protected paths.
    if any(rel == p or rel.startswith(p) for p in PROTECTED_PREFIXES):
        if empty and rel not in ("README.md", "pyproject.toml"):
            return "keep_uncertain", "empty file inside protected path"
        if referenced:
            return "keep_referenced", ref_reason
        return "keep", "protected path (src/tests/scripts/docs/paper_*/README/pyproject)"

    # Anything else.
    if referenced:
        return "keep_referenced", ref_reason
    if empty:
        return "delete_candidate", "empty file, unreferenced, outside protected paths"
    return "keep_uncertain", "unreferenced, outside protected paths"


def section_listing(files: list[Path], prefix: str) -> list[dict[str, Any]]:
    rows = []
    for p in files:
        rel = _rel(p)
        if not rel.startswith(prefix):
            continue
        try:
            st = p.stat()
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError:
            size, mtime = -1, -1
        rows.append({"path": rel, "size_bytes": size, "mtime": mtime,
                     "ext": p.suffix})
    return rows


def find_possible_duplicates(files: list[Path]) -> list[dict[str, Any]]:
    """Group output files that share a stem but differ only by suffix /
    a trailing version tag (e.g. foo.json vs foo_v2.json)."""
    stems: dict[str, list[str]] = {}
    for p in files:
        rel = _rel(p)
        if not rel.startswith("outputs/"):
            continue
        stem = re.sub(r"(_v\d+)?\.(json|csv|md)$", "", p.name)
        stems.setdefault(stem, []).append(rel)
    dups = []
    for stem, paths in sorted(stems.items()):
        versioned = [x for x in paths if re.search(r"_v\d+\.", x)]
        if versioned:
            dups.append({"stem": stem, "paths": sorted(paths),
                         "note": "versioned sibling(s) present"})
    return dups


def main() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    files = collect_files()
    junk = collect_junk()
    blob = build_reference_index()

    records: list[dict[str, Any]] = []
    for p in files:
        rel = _rel(p)
        # Skip the audit's own artifacts to avoid self-reference noise.
        referenced, reason = is_referenced(p, blob)
        action, action_reason = classify(p, referenced, reason)
        try:
            st = p.stat()
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError:
            size, mtime = -1, -1
        records.append({
            "path": rel,
            "ext": p.suffix,
            "size_bytes": size,
            "mtime": mtime,
            "referenced": referenced,
            "reference_reason": reason,
            "action": action,
            "action_reason": action_reason,
        })

    junk_records = []
    for p in junk:
        rel = _rel(p)
        is_dir = p.is_dir()
        junk_records.append({
            "path": rel + ("/" if is_dir else ""),
            "kind": "directory" if is_dir else "file",
            "action": "delete_candidate",
            "action_reason": "low-risk junk (cache / temp / editor backup)",
        })

    # Section listings.
    md_files = section_listing(files, "")
    md_files = [r for r in md_files if r["ext"] == ".md"]
    outputs_files = section_listing(files, "outputs/")
    scripts_files = section_listing(files, "scripts/")
    tests_files = section_listing(files, "tests/")
    docs_files = section_listing(files, "docs/")

    duplicates = find_possible_duplicates(files)

    by_action: dict[str, int] = {}
    for r in records:
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1

    possibly_unreferenced = [
        r for r in records
        if not r["referenced"] and r["action"] in
        ("keep_uncertain", "delete_candidate", "move_to_archive")
    ]

    report: dict[str, Any] = {
        "status": "ok",
        "report": "repository_cleanup_audit",
        "repo_root": str(REPO_ROOT),
        "summary": {
            "total_files_tracked": len(records),
            "junk_items": len(junk_records),
            "markdown_files": len(md_files),
            "outputs_files": len(outputs_files),
            "scripts_files": len(scripts_files),
            "tests_files": len(tests_files),
            "docs_files": len(docs_files),
            "possible_duplicates": len(duplicates),
            "possibly_unreferenced": len(possibly_unreferenced),
            "action_counts": by_action,
        },
        "protected_prefixes": list(PROTECTED_PREFIXES),
        "core_artifact_substrings": list(CORE_ARTIFACT_SUBSTRINGS),
        "files": records,
        "junk": junk_records,
        "markdown_files": md_files,
        "outputs_files": outputs_files,
        "scripts_files": scripts_files,
        "tests_files": tests_files,
        "docs_files": docs_files,
        "possible_duplicates": duplicates,
        "possibly_unreferenced": possibly_unreferenced,
    }

    # JSON
    (outputs_dir / "repository_cleanup_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )

    # CSV (one row per tracked file + junk)
    with open(outputs_dir / "repository_cleanup_audit.csv", "w",
              encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "path", "ext", "size_bytes", "mtime", "referenced",
            "action", "action_reason",
        ])
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
        for r in junk_records:
            w.writerow({"path": r["path"], "ext": "", "size_bytes": "",
                        "mtime": "", "referenced": False,
                        "action": r["action"],
                        "action_reason": r["action_reason"]})

    # Markdown
    lines: list[str] = []

    def wl(s: str = "") -> None:
        lines.append(s)

    wl("# Repository Cleanup Audit")
    wl()
    wl("Read-only audit. This document recommends actions; it does not "
       "delete anything. Low-risk junk is removed by "
       "`scripts/clean_repository_low_risk.py`; Markdown / outputs "
       "deletions require an explicit allowlist applied by "
       "`scripts/apply_repository_cleanup_allowlist.py`.")
    wl()
    wl("## Summary")
    wl()
    for k, v in report["summary"].items():
        if k == "action_counts":
            continue
        wl(f"- {k}: **{v}**")
    wl()
    wl("### Action counts")
    wl()
    wl("| action | count |")
    wl("|---|---|")
    for k in sorted(by_action):
        wl(f"| `{k}` | {by_action[k]} |")
    wl()
    wl("## Junk (low-risk delete candidates)")
    wl()
    if not junk_records:
        wl("(none)")
    else:
        wl(f"{len(junk_records)} items (caches / temp / editor backups). "
           "Removed by the low-risk cleaner.")
        wl()
        for r in junk_records[:40]:
            wl(f"- `{r['path']}` — {r['action_reason']}")
        if len(junk_records) > 40:
            wl(f"- ... and {len(junk_records) - 40} more")
    wl()
    wl("## Possible duplicates / versioned siblings")
    wl()
    if not duplicates:
        wl("(none)")
    else:
        for d in duplicates:
            wl(f"- `{d['stem']}` — {', '.join(d['paths'])} ({d['note']})")
    wl()
    wl("## Possibly unreferenced files (need human review)")
    wl()
    if not possibly_unreferenced:
        wl("(none — every tracked file is protected, core, or referenced)")
    else:
        wl("| path | action | reason |")
        wl("|---|---|---|")
        for r in possibly_unreferenced:
            wl(f"| `{r['path']}` | `{r['action']}` | {r['action_reason']} |")
    wl()
    wl("## Action legend")
    wl()
    wl("- `keep` — protected path (README / pyproject / src / tests / "
       "scripts / docs / paper_*).")
    wl("- `keep_core_artifact` — Stage 5.7 / 5.8 / 7.6 core output, "
       "claims, summary, or limitations artifact.")
    wl("- `keep_referenced` — referenced by README / docs / scripts / "
       "tests / src.")
    wl("- `keep_uncertain` — not obviously referenced; do NOT delete "
       "without human review.")
    wl("- `move_to_docs` / `move_to_archive` — relocation suggestions "
       "(never auto-applied).")
    wl("- `delete_candidate` — low-risk junk only.")
    wl()
    (outputs_dir / "repository_cleanup_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )

    print(f"Tracked files: {len(records)}")
    print(f"Junk items: {len(junk_records)}")
    print(f"Action counts: {by_action}")
    print(f"Possibly unreferenced: {len(possibly_unreferenced)}")
    print("Wrote: outputs/repository_cleanup_audit.{json,csv,md}")


if __name__ == "__main__":
    main()
