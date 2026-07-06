"""Summarize all attack JSONL results into CSV + Markdown (failed/blocked kept)."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
GLOB = str(REPO_ROOT / "outputs" / "attacks" / "**" / "*.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=None,
                   help="scan this dir (recursive) for *.jsonl instead of outputs/attacks/**")
    p.add_argument("--output-dir", default=None,
                   help="write the summary here instead of outputs/attacks/summary")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    scan = str(Path(args.input_dir) / "**" / "*.jsonl") if args.input_dir else GLOB
    if args.dry_run:
        print(f"[dry-run] scan {scan}")
        return
    from pllo.attacks.result_io import read_many_jsonl, results_to_csv, results_to_markdown
    paths = [x for x in glob.glob(scan, recursive=True) if "summary" not in x]
    results = read_many_jsonl(paths)
    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    d = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "attacks" / "summary"
    results_to_csv(results, d / "attack_results_summary.csv")
    results_to_markdown(results, d / "attack_results_summary.md", title="Attack results (all runs)")
    print(f"{len(results)} results from {len(paths)} files; by status: {counts}")
    print(f"wrote {d}/attack_results_summary.csv + .md")


if __name__ == "__main__":
    main()
