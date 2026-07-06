"""Build the security main table (Table B) from attack results.

measured (default): outputs/paper_security/security_attack_table_measured.{json,csv,md}.
qualitative (--qualitative-draft): reads configs/attack_qualitative_matrix.json,
writes a *_qualitative_draft table (clearly labelled; never overwrites measured).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "outputs" / "paper_security"
GLOB = str(REPO_ROOT / "outputs" / "attacks" / "**" / "*.jsonl")
QCFG = REPO_ROOT / "configs" / "attack_qualitative_matrix.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualitative-draft", action="store_true")
    p.add_argument("--input-dir", default=None,
                   help="scan this dir (recursive) for *.jsonl instead of outputs/attacks/**; "
                        "use to keep toy/tiny and per-model tables separate")
    p.add_argument("--output-dir", default=None,
                   help="write the table here instead of outputs/paper_security/")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    scan = str(Path(args.input_dir) / "**" / "*.jsonl") if args.input_dir else GLOB
    out = Path(args.output_dir) if args.output_dir else OUT
    if args.dry_run:
        print(f"[dry-run] mode={'qualitative_draft' if args.qualitative_draft else 'measured'} "
              f"scan={scan} out={out}")
        return
    from pllo.attacks.result_io import read_many_jsonl
    from pllo.attacks.table_builder import write_measured_table, write_qualitative_table
    if args.qualitative_draft:
        cfg = json.loads(QCFG.read_text())
        paths = write_qualitative_table(cfg, out)
        print("QUALITATIVE DRAFT (NOT measured):")
    else:
        results = read_many_jsonl(glob.glob(scan, recursive=True))
        paths = write_measured_table(results, out)
        print(f"MEASURED table from {len(results)} results:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
