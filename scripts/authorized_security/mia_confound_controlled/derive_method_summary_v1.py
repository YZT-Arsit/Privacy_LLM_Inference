#!/usr/bin/env python3
"""Derive three-seed G1/G2 method means from the paired raw table."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if args.output_csv.exists() or args.output_md.exists():
        raise RuntimeError("refusing to overwrite method summary")
    rows = list(csv.DictReader(args.paired_csv.open()))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["metric"]].append(row)
    output = []
    for metric, selected in grouped.items():
        selected.sort(key=lambda row: int(row["seed"]))
        g1 = np.asarray([float(row["g1_mean"]) for row in selected])
        g2 = np.asarray([float(row["g2_mean"]) for row in selected])
        delta = g2 - g1
        output.append({
            "metric": metric, "seeds": len(selected),
            "g1_mean": float(g1.mean()), "g1_seed_std": float(g1.std(ddof=1)),
            "g2_mean": float(g2.mean()), "g2_seed_std": float(g2.std(ddof=1)),
            "delta_mean_g2_minus_g1": float(delta.mean()),
            "relative_delta_percent_vs_g1": float(100 * delta.mean() / g1.mean()) if g1.mean() else "",
        })
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    lines = ["# G1/G2 Three-Seed Method Summary", "",
             "BLEU/chrF/ROUGE-L values are means of per-example sentence metrics used by the paired analysis; they are not corpus-level scores.", "",
             "| Metric | G1 mean ± seed SD | G2 mean ± seed SD | G2−G1 | Relative Δ |",
             "|---|---:|---:|---:|---:|"]
    for row in output:
        relative = row["relative_delta_percent_vs_g1"]
        lines.append(f"| {row['metric']} | {row['g1_mean']:.3f} ± {row['g1_seed_std']:.3f} | "
                     f"{row['g2_mean']:.3f} ± {row['g2_seed_std']:.3f} | "
                     f"{row['delta_mean_g2_minus_g1']:.3f} | "
                     f"{relative:.2f}% |" if relative != "" else "")
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
