#!/usr/bin/env python3
"""CPU-only paired G1/G2 analysis over the three frozen E2E seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sacrebleu


SEEDS = (7, 1234, 2025)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for left in a:
        previous = 0
        for index, right in enumerate(b, 1):
            old = dp[index]
            dp[index] = previous + 1 if left == right else max(dp[index], dp[index - 1])
            previous = old
    return dp[-1]


def rouge_l(text: str, references: list[str]) -> float:
    hypothesis = text.split()
    best = 0.0
    for reference in references:
        target = reference.split()
        if not hypothesis or not target:
            continue
        common = lcs(hypothesis, target)
        if common:
            precision, recall = common / len(hypothesis), common / len(target)
            best = max(best, 2 * precision * recall / (precision + recall))
    return 100.0 * best


def metrics(row: dict) -> dict[str, float]:
    text, references = row["generated_text"], row["references"]
    return {
        "BLEU": float(sacrebleu.sentence_bleu(text, references).score),
        "chrF": float(sacrebleu.sentence_chrf(text, references).score),
        "ROUGE_L": rouge_l(text, references),
        "length": float(row["generated_token_count"]),
        "repetition_pct": 100.0 * float(row["repetition_bigram_frac"]),
        "invalid_pct": 100.0 * float(bool(row["invalid_output"])),
    }


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_bootstrap(delta: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        means[start:start + count] = delta[indices].mean(axis=1)
    return percentile_ci(means)


def hierarchical_bootstrap(matrix: np.ndarray, samples: int, seed: int) -> list[float]:
    """Resample three seeds, then aligned examples within each selected seed."""
    rng = np.random.default_rng(seed)
    n_seed, n_example = matrix.shape
    values = np.empty(samples, dtype=float)
    for draw in range(samples):
        chosen_seeds = rng.integers(0, n_seed, size=n_seed)
        seed_means = []
        for selected in chosen_seeds:
            indices = rng.integers(0, n_example, size=n_example)
            seed_means.append(float(matrix[selected, indices].mean()))
        values[draw] = float(np.mean(seed_means))
    return percentile_ci(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g2-s1234", type=Path, default=None,
                        help="optional verified isolated copy of the frozen seed-1234 generation")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite isolated analysis directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    generation = args.repo / "results/aaai_private_base/generative_lora/generations"
    g1_paths = {seed: generation / f"g1_final_s{seed}_test.jsonl" for seed in SEEDS}
    g2_paths = {
        7: generation / "g2_protected_e2e_L12_s7_extension.jsonl",
        1234: args.g2_s1234 or generation / "g2_protected_e2e_L12_s1234_full.jsonl",
        2025: generation / "g2_protected_e2e_L12_s2025_extension.jsonl",
    }
    raw_rows: list[dict] = []
    delta_by_metric: dict[str, list[np.ndarray]] = {}
    source_hashes: dict[str, str] = {}
    for seed in SEEDS:
        g1_path, g2_path = g1_paths[seed], g2_paths[seed]
        source_hashes[str(g1_path.relative_to(args.repo))] = sha256(g1_path)
        source_hashes[str(g2_path.relative_to(args.repo))] = sha256(g2_path)
        g1 = {str(row["sample_id"]): row for row in load_jsonl(g1_path)}
        g2 = {str(row["sample_id"]): row for row in load_jsonl(g2_path)}
        ids = sorted(set(g1) & set(g2))
        if len(ids) != 500 or set(g1) != set(g2):
            raise RuntimeError(f"seed {seed}: expected exactly 500 aligned sample IDs")
        if any(g1[item]["input_hash"] != g2[item]["input_hash"] for item in ids):
            raise RuntimeError(f"seed {seed}: input hashes are not aligned")
        left = {item: metrics(g1[item]) for item in ids}
        right = {item: metrics(g2[item]) for item in ids}
        for metric_index, metric in enumerate(left[ids[0]]):
            g1_values = np.asarray([left[item][metric] for item in ids], dtype=float)
            g2_values = np.asarray([right[item][metric] for item in ids], dtype=float)
            delta = g2_values - g1_values
            delta_by_metric.setdefault(metric, []).append(delta)
            ci = paired_bootstrap(delta, args.bootstrap,
                                  args.bootstrap_seed + seed + 1009 * metric_index)
            std = float(delta.std(ddof=1))
            raw_rows.append({
                "seed": seed, "metric": metric, "n": len(ids),
                "g1_mean": float(g1_values.mean()), "g2_mean": float(g2_values.mean()),
                "mean_delta_g2_minus_g1": float(delta.mean()),
                "paired_bootstrap_ci95_low": ci[0], "paired_bootstrap_ci95_high": ci[1],
                "cohen_dz": float(delta.mean() / std) if std else math.nan,
                "positive": int(np.sum(delta > 0)), "negative": int(np.sum(delta < 0)),
                "zero": int(np.sum(delta == 0)),
            })

    aggregate_rows: list[dict] = []
    for metric_index, (metric, seed_deltas) in enumerate(delta_by_metric.items()):
        matrix = np.stack(seed_deltas)
        per_seed_means = matrix.mean(axis=1)
        ci = hierarchical_bootstrap(matrix, args.bootstrap,
                                    args.bootstrap_seed + 7919 * metric_index)
        aggregate_rows.append({
            "metric": metric, "seeds": len(SEEDS), "examples_per_seed": matrix.shape[1],
            "mean_delta_across_seeds": float(per_seed_means.mean()),
            "seed_std": float(per_seed_means.std(ddof=1)),
            "hierarchical_bootstrap_ci95_low": ci[0],
            "hierarchical_bootstrap_ci95_high": ci[1],
            "seed_7_delta": float(per_seed_means[0]),
            "seed_1234_delta": float(per_seed_means[1]),
            "seed_2025_delta": float(per_seed_means[2]),
        })

    write_csv(args.output / "per_seed_paired_metrics.csv", raw_rows)
    write_csv(args.output / "three_seed_paired_summary.csv", aggregate_rows)
    result = {
        "schema": "offline_three_seed_paired_g2_minus_g1", "version": "1.0",
        "seeds": list(SEEDS), "examples_per_seed": 500,
        "bootstrap_samples": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "within_seed_ci": "paired percentile bootstrap of mean per-example delta",
        "aggregate_ci": "hierarchical percentile bootstrap: seeds then aligned examples",
        "source_hashes": source_hashes, "per_seed": raw_rows, "aggregate": aggregate_rows,
    }
    (args.output / "paired_statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# G1/G2 Three-Seed Paired Statistics", "",
             "All results use existing frozen, sample-aligned generations. No inference was rerun. "
             "Positive deltas mean G2 minus the seed-matched G1.", "",
             "| Metric | Mean delta ± seed SD | Hierarchical 95% CI | s7 | s1234 | s2025 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate_rows:
        lines.append(f"| {row['metric']} | {row['mean_delta_across_seeds']:.3f} ± {row['seed_std']:.3f} | "
                     f"[{row['hierarchical_bootstrap_ci95_low']:.3f}, "
                     f"{row['hierarchical_bootstrap_ci95_high']:.3f}] | "
                     f"{row['seed_7_delta']:.3f} | {row['seed_1234_delta']:.3f} | "
                     f"{row['seed_2025_delta']:.3f} |")
    lines += ["", "## Interpretation", "",
              "These are exploratory three-seed intervals. With only three trained-model seeds, "
              "the seed-level uncertainty is necessarily imprecise; intervals are not equivalence tests.", "",
              "The raw per-seed means, paired intervals, effect sizes, and sign counts are in "
              "`per_seed_paired_metrics.csv`."]
    (args.output / "paired_statistics_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
