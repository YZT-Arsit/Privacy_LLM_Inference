"""Stage 7.3 -- output-size guard CLI.

Exit code 0 if no file exceeds --fail-mb, else 1. Never deletes anything;
local model checkpoints are excluded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.repo_hygiene import check_output_sizes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--warn-mb", type=int, default=10)
    ap.add_argument("--fail-mb", type=int, default=100)
    args = ap.parse_args()

    report = check_output_sizes(args.output_dir, warn_mb=args.warn_mb,
                                fail_mb=args.fail_mb)
    print(f"output_dir={report['output_dir']} "
          f"total_size_mb={report['total_size_mb']}")
    if report["max_file"]:
        mf = report["max_file"]
        print(f"max_file={mf['path']} ({mf['size_mb']} MB)")
    for item in report["warnings"]:
        print(f"WARN  {item['path']} = {item['size_mb']} MB "
              f"(> {args.warn_mb} MB)")
    for item in report["failures"]:
        print(f"FAIL  {item['path']} = {item['size_mb']} MB "
              f"(> {args.fail_mb} MB)")
    if report["passed"]:
        print(f"OK: no file exceeds {args.fail_mb} MB.")
        return 0
    print(f"FAILED: {len(report['failures'])} file(s) exceed "
          f"{args.fail_mb} MB.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
