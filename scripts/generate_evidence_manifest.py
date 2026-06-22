"""Stage 7.3 -- generate the compact paper-ready evidence manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.repo_hygiene import generate_evidence_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()

    manifest = generate_evidence_manifest(output_dir=args.output_dir)
    for f in manifest["written_files"]:
        print(f"Wrote: {f}")
    print(f"stages_documented={manifest['stage_count']}")


if __name__ == "__main__":
    main()
