"""Apply an explicit repository-cleanup allowlist.

This is the ONLY sanctioned path for deleting / archiving / moving
Markdown or outputs files. It refuses to act on anything that is not
explicitly listed, that lives under a protected path, that does not
exist, or that is still referenced by README / docs / scripts / tests
/ src.

Allowlist format (outputs/repository_cleanup_allowlist.json):

    {
      "delete":  ["path/to/file.md"],
      "archive": ["outputs/old_file.json"],
      "move":    [{"from": "old/path.md", "to": "docs/archive/old/path.md"}]
    }

"archive" moves a file under outputs/archive/<original-path>.

Run with --dry-run (default) to preview; pass --apply to act.
Writes outputs/repository_cleanup_apply_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "outputs" / "repository_cleanup_allowlist.json"
ARCHIVE_ROOT = REPO_ROOT / "outputs" / "archive"

# Paths that may never be deleted (prefix match), regardless of allowlist,
# UNLESS the file is empty (size 0). Core artifacts are also protected.
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
)
PROTECTED_OUTPUT_SUBSTRINGS = (
    "permutation_invariant_leakage",
    "lookup_nonlinear_cost_proxy",
    "masked_gradient_lora",
    "claims",
    "lifecycle",
    "summary",
    "limitations",
)

REFERENCE_SOURCE_SUFFIXES = (".py", ".md", ".tex", ".toml", ".cfg", ".txt")
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                 ".claude", "node_modules"}


# Cleanup-control documents: a file appearing inside one of these does
# NOT count as a real reference (otherwise listing a file in the plan
# would make it un-deletable).
CONTROL_DOC_PREFIXES = ("repository_cleanup_",)


def build_reference_blob() -> str:
    chunks: list[str] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            if not name.endswith(REFERENCE_SOURCE_SUFFIXES):
                continue
            if any(name.startswith(pfx) for pfx in CONTROL_DOC_PREFIXES):
                continue  # skip cleanup-control docs
            p = Path(dirpath) / name
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def _is_protected(rel: str, *, empty: bool) -> tuple[bool, str]:
    if any(rel == p or rel.startswith(p) for p in PROTECTED_PREFIXES):
        if empty:
            return False, "protected-path but empty (allowed)"
        return True, "protected path"
    if rel.startswith("outputs/") and any(
            s in os.path.basename(rel) for s in PROTECTED_OUTPUT_SUBSTRINGS):
        return True, "protected core output artifact"
    return False, ""


def _reference_hits(rel: str, blob: str) -> list[str]:
    """Return references EXCLUDING the allowlist + audit/plan/report files
    themselves (a file being listed in the cleanup plan/allowlist does
    not count as a real reference)."""
    base = os.path.basename(rel)
    # Strip the cleanup-control documents from the blob first.
    control_names = (
        "repository_cleanup_allowlist",
        "repository_cleanup_plan",
        "repository_cleanup_audit",
        "repository_cleanup_final_report",
        "repository_cleanup_apply_report",
    )
    hits = []
    if base and blob.count(base) > 0:
        hits.append(f"basename '{base}' appears in repo text")
    if rel in blob:
        hits.append(f"path '{rel}' appears in repo text")
    # Heuristic discount: if the only place the name appears is a control
    # document, the caller should treat it as not-really-referenced. We
    # cannot cheaply attribute counts here, so we expose the raw hits and
    # let the report show them; the operator decides. To stay safe we keep
    # any hit as a blocker EXCEPT when the basename is clearly unique to
    # control docs (handled by the audit, which excluded them).
    _ = control_names
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually perform delete/archive/move (default dry-run)")
    ap.add_argument("--allowlist", default=str(ALLOWLIST_PATH))
    ap.add_argument("--ignore-references", action="store_true",
                    help="proceed even if a soft reference is found "
                         "(use only when references are known to be in "
                         "cleanup-control documents)")
    args = ap.parse_args()

    allow_path = Path(args.allowlist)
    if not allow_path.is_file():
        print(f"No allowlist at {allow_path}; nothing to do.")
        report = {"status": "no_allowlist", "allowlist": str(allow_path)}
        (REPO_ROOT / "outputs" / "repository_cleanup_apply_report.json"
         ).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    allow = json.loads(allow_path.read_text(encoding="utf-8"))
    blob = build_reference_blob()

    results: list[dict[str, Any]] = []

    def consider(rel: str, op: str, dest: str | None = None) -> dict[str, Any]:
        abs_path = REPO_ROOT / rel
        rec: dict[str, Any] = {"path": rel, "op": op, "dest": dest,
                               "performed": False}
        if not abs_path.exists():
            rec["decision"] = "skip_missing"
            rec["reason"] = "file does not exist"
            return rec
        empty = abs_path.is_file() and abs_path.stat().st_size == 0
        protected, why = _is_protected(rel, empty=empty)
        if protected:
            rec["decision"] = "refused_protected"
            rec["reason"] = why
            return rec
        hits = _reference_hits(rel, blob)
        if hits and not args.ignore_references:
            rec["decision"] = "refused_referenced"
            rec["reason"] = "; ".join(hits)
            return rec
        rec["reference_hits"] = hits
        rec["decision"] = "eligible"
        rec["reason"] = "passed all guards"
        if args.apply:
            if op == "delete":
                abs_path.unlink()
            elif op == "archive":
                target = ARCHIVE_ROOT / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(abs_path), str(target))
                rec["dest"] = str(target.relative_to(REPO_ROOT))
            elif op == "move":
                target = REPO_ROOT / (dest or "")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(abs_path), str(target))
            rec["performed"] = True
        return rec

    for rel in allow.get("delete", []):
        results.append(consider(rel, "delete"))
    for rel in allow.get("archive", []):
        results.append(consider(rel, "archive"))
    for entry in allow.get("move", []):
        results.append(consider(entry["from"], "move", entry.get("to")))

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] allowlist actions:")
    for r in results:
        flag = "DONE" if r["performed"] else r["decision"]
        print(f"  [{flag}] {r['op']} {r['path']} "
              f"{'-> ' + str(r['dest']) if r.get('dest') else ''} "
              f"({r['reason']})")

    report = {
        "status": "ok",
        "mode": mode,
        "allowlist": str(allow_path),
        "results": results,
        "counts": {
            "eligible": sum(1 for r in results if r["decision"] == "eligible"),
            "performed": sum(1 for r in results if r["performed"]),
            "refused_protected": sum(
                1 for r in results if r["decision"] == "refused_protected"),
            "refused_referenced": sum(
                1 for r in results if r["decision"] == "refused_referenced"),
            "skip_missing": sum(
                1 for r in results if r["decision"] == "skip_missing"),
        },
    }
    (REPO_ROOT / "outputs" / "repository_cleanup_apply_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"counts: {report['counts']}")


if __name__ == "__main__":
    main()
