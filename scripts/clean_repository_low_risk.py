"""Low-risk repository cleanup.

Removes ONLY low-risk junk:

    __pycache__/            (directories)
    .pytest_cache/          (directory)
    .mypy_cache/ .ruff_cache/
    *.pyc, *.pyo, *.tmp, *.bak, *.orig
    *~ (editor backups)
    .DS_Store, Thumbs.db
    files under tmp/ debug/ scratch/ that are NOT referenced

It NEVER touches Markdown, JSON/CSV under outputs/, or any file under
src/ tests/ scripts/ docs/ paper_results/ paper_draft/ except the junk
patterns above. Markdown / outputs deletions must go through the
explicit allowlist (see apply_repository_cleanup_allowlist.py).

Run with --dry-run (default) to preview; pass --apply to delete.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

JUNK_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
JUNK_SUFFIXES = (".pyc", ".pyo", ".tmp", ".bak", ".orig")
JUNK_BASENAMES = {".DS_Store", "Thumbs.db"}
SCRATCH_DIRS = {"tmp", "debug", "scratch"}

# Never descend here.
HARD_SKIP = {".git", ".venv", "venv", ".claude", "node_modules"}


def find_junk() -> tuple[list[Path], list[Path]]:
    junk_dirs: list[Path] = []
    junk_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        parts = Path(dirpath).parts
        if any(s in parts for s in HARD_SKIP):
            dirnames[:] = []
            continue
        base = os.path.basename(dirpath)
        if base in JUNK_DIR_NAMES:
            junk_dirs.append(Path(dirpath))
            dirnames[:] = []
            continue
        for name in filenames:
            p = Path(dirpath) / name
            if name in JUNK_BASENAMES or name.endswith(JUNK_SUFFIXES) \
                    or name.endswith("~"):
                junk_files.append(p)
    return sorted(set(junk_dirs)), sorted(set(junk_files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is dry-run)")
    args = ap.parse_args()

    junk_dirs, junk_files = find_junk()
    total = len(junk_dirs) + len(junk_files)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] low-risk cleanup")
    print(f"  junk directories: {len(junk_dirs)}")
    print(f"  junk files:       {len(junk_files)}")

    removed: list[str] = []
    for d in junk_dirs:
        rel = str(d.relative_to(REPO_ROOT))
        print(f"  dir  {rel}/")
        if args.apply:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(rel + "/")
    for f in junk_files:
        rel = str(f.relative_to(REPO_ROOT))
        print(f"  file {rel}")
        if args.apply:
            try:
                f.unlink()
                removed.append(rel)
            except OSError:
                pass

    if args.apply:
        print(f"Removed {len(removed)} junk item(s).")
    else:
        print(f"Would remove {total} junk item(s). Re-run with --apply.")


if __name__ == "__main__":
    main()
