#!/usr/bin/env python3
"""Registry-gated G1/G2 corpus metrics and paired bootstrap audit."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from canonical_io import BUNDLE, load_canonical_jsonl

SEEDS = (7, 1234, 2025)
BOOTSTRAPS = 10_000


def reference_documents(rows: list[dict]) -> list[list[str]]:
    """Established E2E convention: pad absent references with the first reference."""
    max_refs = max(len(row["references"]) for row in rows)
    return [[row["references"][i] if i < len(row["references"]) else row["references"][0]
             for row in rows] for i in range(max_refs)]


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if left == right else max(current[-1], previous[j]))
        previous = current
    return previous[-1]


def rouge_l_f1(hypothesis: str, references: list[str]) -> float:
    """Maximum whitespace-token LCS F1 over references, on a 0--100 scale."""
    hyp = hypothesis.split()
    if not hyp:
        return 0.0
    best = 0.0
    for reference in references:
        ref = reference.split()
        if not ref:
            continue
        lcs = lcs_length(hyp, ref)
        precision, recall = lcs / len(hyp), lcs / len(ref)
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return 100.0 * best


def strict_pair(seed: int) -> tuple[list[dict], list[dict], dict, dict]:
    g1_raw, g1_entry = load_canonical_jsonl(f"g1_seed{seed}")
    g2_raw, g2_entry = load_canonical_jsonl(f"g2_seed{seed}")
    for name, rows in (("G1", g1_raw), ("G2", g2_raw)):
        counts = Counter(row["sample_id"] for row in rows)
        duplicates = [key for key, value in counts.items() if value != 1]
        if duplicates:
            raise RuntimeError(f"seed {seed} {name}: duplicate sample IDs")
    left = {row["sample_id"]: row for row in g1_raw}
    right = {row["sample_id"]: row for row in g2_raw}
    if set(left) != set(right) or len(left) != 500:
        raise RuntimeError(f"seed {seed}: expected exactly 500 identical sample IDs")
    ids = sorted(left)
    for sample_id in ids:
        a, b = left[sample_id], right[sample_id]
        for field in ("input_hash", "meaning_representation", "references"):
            if a.get(field) != b.get(field):
                raise RuntimeError(f"seed {seed} sample {sample_id}: mismatch in {field}")
    return [left[key] for key in ids], [right[key] for key in ids], g1_entry, g2_entry


def sufficient_statistics(rows: list[dict], metric) -> np.ndarray:
    hyps = [row["generated_text"] for row in rows]
    return np.asarray(metric._extract_corpus_statistics(hyps, reference_documents(rows)), dtype=float)


def score_from_sum(metric, values: np.ndarray) -> float:
    return float(metric._compute_score_from_stats(values.tolist()).score)


def per_example(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "mean_sentence_bleu": np.asarray([
            sacrebleu.sentence_bleu(row["generated_text"], row["references"]).score for row in rows
        ]),
        "mean_sentence_chrf": np.asarray([
            sacrebleu.sentence_chrf(row["generated_text"], row["references"]).score for row in rows
        ]),
        "rouge_l": np.asarray([rouge_l_f1(row["generated_text"], row["references"]) for row in rows]),
        "output_length": np.asarray([row["generated_token_count"] for row in rows], dtype=float),
        "invalid_rate": np.asarray([bool(row["invalid_output"]) for row in rows], dtype=float) * 100.0,
        "repetition_bigram_fraction": np.asarray([row["repetition_bigram_frac"] for row in rows], dtype=float) * 100.0,
        "repeated_output_rate": np.asarray([row["repetition_bigram_frac"] > 0 for row in rows], dtype=float) * 100.0,
    }


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    out_json = BUNDLE / "generation_metric_audit_v2.json"
    if out_json.exists():
        raise RuntimeError(f"refusing to overwrite versioned report: {out_json}")
    bleu, chrf = BLEU(), CHRF()
    results, bootstrap_by_seed = [], {}
    for seed in SEEDS:
        g1, g2, entry1, entry2 = strict_pair(seed)
        stats = {
            "corpus_bleu": (sufficient_statistics(g1, bleu), sufficient_statistics(g2, bleu), bleu),
            "corpus_chrf": (sufficient_statistics(g1, chrf), sufficient_statistics(g2, chrf), chrf),
        }
        ex1, ex2 = per_example(g1), per_example(g2)
        rng = np.random.default_rng(2026071400 + seed)
        boot = {key: np.empty(BOOTSTRAPS) for key in [*stats, *ex1]}
        for start in range(0, BOOTSTRAPS, 250):
            stop = min(start + 250, BOOTSTRAPS)
            indices = rng.integers(0, len(g1), size=(stop - start, len(g1)))
            for metric_name, (left, right, metric) in stats.items():
                sums1, sums2 = left[indices].sum(axis=1), right[indices].sum(axis=1)
                boot[metric_name][start:stop] = [
                    score_from_sum(metric, b) - score_from_sum(metric, a)
                    for a, b in zip(sums1, sums2)
                ]
            for metric_name in ex1:
                boot[metric_name][start:stop] = (ex2[metric_name][indices] - ex1[metric_name][indices]).mean(axis=1)
        values = {}
        for metric_name, (left, right, metric) in stats.items():
            a, b = score_from_sum(metric, left.sum(axis=0)), score_from_sum(metric, right.sum(axis=0))
            values[metric_name] = {"g1": a, "g2": b, "difference_g2_minus_g1": b - a,
                                   "paired_bootstrap_ci95": interval(boot[metric_name])}
        for metric_name in ex1:
            a, b = float(ex1[metric_name].mean()), float(ex2[metric_name].mean())
            values[metric_name] = {"g1": a, "g2": b, "difference_g2_minus_g1": b - a,
                                   "paired_bootstrap_ci95": interval(boot[metric_name])}
        results.append({"seed": seed, "matched_samples": len(g1), "g1_path": entry1["path"],
                        "g1_sha256": entry1["sha256"], "g2_path": entry2["path"],
                        "g2_sha256": entry2["sha256"], "metrics": values})
        bootstrap_by_seed[seed] = boot

    metric_names = list(results[0]["metrics"])
    seed_summary = {}
    hierarchical = {}
    hrng = np.random.default_rng(2026071499)
    for metric_name in metric_names:
        a = np.asarray([row["metrics"][metric_name]["g1"] for row in results])
        b = np.asarray([row["metrics"][metric_name]["g2"] for row in results])
        d = b - a
        seed_summary[metric_name] = {
            "g1_mean": float(a.mean()), "g1_sample_sd": float(a.std(ddof=1)),
            "g2_mean": float(b.mean()), "g2_sample_sd": float(b.std(ddof=1)),
            "difference_mean": float(d.mean()), "difference_sample_sd": float(d.std(ddof=1)),
        }
        h = np.empty(BOOTSTRAPS)
        for i in range(BOOTSTRAPS):
            selected = hrng.choice(SEEDS, size=len(SEEDS), replace=True)
            h[i] = np.mean([bootstrap_by_seed[int(seed)][metric_name][hrng.integers(BOOTSTRAPS)]
                            for seed in selected])
        hierarchical[metric_name] = {
            "mean": float(h.mean()), "exploratory_interval95": interval(h),
            "interpretation": "Exploratory seed-and-example resampling interval; not an equivalence test."
        }

    report = {
        "schema": "g1_g2_generation_metric_audit", "version": "2.0",
        "primary_metrics": ["corpus_bleu", "corpus_chrf", "rouge_l"],
        "rouge_l_definition": "Arithmetic mean over matched examples of 100 × maximum whitespace-token LCS F1 across that example's references.",
        "bootstrap": {"replicates": BOOTSTRAPS,
                      "method": "Resample matched sample IDs with replacement; recompute complete corpus BLEU/chrF from resampled sufficient statistics and all other aggregate metrics from the resampled examples."},
        "metric_software": {"sacrebleu_version": sacrebleu.__version__,
                            "bleu_signature": str(bleu.get_signature()),
                            "chrf_signature": str(chrf.get_signature())},
        "seed_results": results, "across_seed_descriptive_statistics": seed_summary,
        "hierarchical_exploratory_intervals": hierarchical,
        "claim_guard": "No three-seed interval in this report is an equivalence test.",
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    fields = ["seed", "metric", "g1", "g2", "difference_g2_minus_g1", "ci95_low", "ci95_high"]
    with (BUNDLE / "generation_metric_audit_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in results:
            for metric_name, value in row["metrics"].items():
                writer.writerow({"seed": row["seed"], "metric": metric_name, "g1": value["g1"],
                                 "g2": value["g2"], "difference_g2_minus_g1": value["difference_g2_minus_g1"],
                                 "ci95_low": value["paired_bootstrap_ci95"][0],
                                 "ci95_high": value["paired_bootstrap_ci95"][1]})

    lines = ["# G1/G2 Corpus and Diagnostic Metric Audit (v2)", "",
             "Primary paper metrics are corpus BLEU, corpus chrF, and explicitly aggregated ROUGE-L. "
             "Mean sentence metrics are retained as secondary diagnostics.", "",
             "ROUGE-L is the arithmetic mean over examples of 100 × the maximum whitespace-token LCS F1 across references.", "",
             "Each paired 95% interval uses 10,000 matched-ID bootstrap replicates and recomputes the complete corpus metric.", "",
             "## Primary corpus metrics", "",
             "| Seed | Metric | G1 | G2 | G2−G1 | Paired 95% CI |", "|---:|---|---:|---:|---:|---:|"]
    for row in results:
        for metric_name in ("corpus_bleu", "corpus_chrf", "rouge_l"):
            value = row["metrics"][metric_name]; lo, hi = value["paired_bootstrap_ci95"]
            lines.append(f"| {row['seed']} | {metric_name} | {value['g1']:.3f} | {value['g2']:.3f} | "
                         f"{value['difference_g2_minus_g1']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Secondary diagnostics", "",
              "| Seed | Metric | G1 | G2 | G2−G1 | Paired 95% CI |", "|---:|---|---:|---:|---:|---:|"]
    for row in results:
        for metric_name in ("mean_sentence_bleu", "mean_sentence_chrf", "output_length", "invalid_rate",
                            "repetition_bigram_fraction", "repeated_output_rate"):
            value = row["metrics"][metric_name]; lo, hi = value["paired_bootstrap_ci95"]
            lines.append(f"| {row['seed']} | {metric_name} | {value['g1']:.3f} | {value['g2']:.3f} | "
                         f"{value['difference_g2_minus_g1']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Across-seed descriptive statistics", "",
              "| Metric | G1 mean ± SD | G2 mean ± SD | Mean Δ ± SD | Hierarchical exploratory 95% interval |",
              "|---|---:|---:|---:|---:|"]
    for metric_name in metric_names:
        s, h = seed_summary[metric_name], hierarchical[metric_name]
        lo, hi = h["exploratory_interval95"]
        lines.append(f"| {metric_name} | {s['g1_mean']:.3f} ± {s['g1_sample_sd']:.3f} | "
                     f"{s['g2_mean']:.3f} ± {s['g2_sample_sd']:.3f} | "
                     f"{s['difference_mean']:+.3f} ± {s['difference_sample_sd']:.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "The hierarchical intervals are exploratory seed-and-example resampling intervals. "
              "They are not equivalence tests and do not establish strict equivalence.", "",
              "All six inputs were selected through the canonical registry and verified by size and SHA-256 before parsing."]
    (BUNDLE / "generation_metric_audit_v2.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
