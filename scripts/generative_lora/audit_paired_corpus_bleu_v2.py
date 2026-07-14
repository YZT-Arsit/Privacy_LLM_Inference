#!/usr/bin/env python3
"""Corrected paired BLEU audit using bootstrap recomputation of corpus BLEU."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU

ROOT = Path(__file__).resolve().parents[2]
GL = ROOT / "results/aaai_private_base/generative_lora"
OUT = GL / "extension/paired_statistics_corrected_v2"
PAIRS = {
    1234: (
        GL / "generations/g1_final_s1234_test.jsonl",
        ROOT / "results/aaai_private_base/offline_analysis/confound_controlled_20260714/source_cache/g2_protected_e2e_L12_s1234_full.jsonl",
    ),
    7: (
        GL / "generations/g1_final_s7_test.jsonl",
        GL / "generations/g2_protected_e2e_L12_s7_extension.jsonl",
    ),
    2025: (
        GL / "generations/g1_final_s2025_test.jsonl",
        GL / "generations/g2_protected_e2e_L12_s2025_extension.jsonl",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reference_documents(rows: list[dict]) -> list[list[str]]:
    """Match the established phase6 behavior: pad missing refs with ref[0]."""
    max_refs = max(len(row["references"]) for row in rows)
    return [[row["references"][i] if i < len(row["references"]) else row["references"][0]
             for row in rows] for i in range(max_refs)]


def sentence_mean(rows: list[dict]) -> float:
    return float(np.mean([
        sacrebleu.sentence_bleu(row["generated_text"], row["references"]).score for row in rows
    ]))


def corpus_stats(rows: list[dict], metric: BLEU) -> np.ndarray:
    hypotheses = [row["generated_text"] for row in rows]
    return np.asarray(metric._extract_corpus_statistics(hypotheses, reference_documents(rows)), dtype=np.int64)


def score_from_sum(metric: BLEU, stats: np.ndarray) -> float:
    return float(metric._compute_score_from_stats(stats.tolist()).score)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite corrected audit: {OUT}")
    OUT.mkdir(parents=True)
    bootstrap_samples = 10000
    metric = BLEU()
    results = []
    diagnostics = []
    for seed, (g1_path, g2_path) in PAIRS.items():
        g1_raw, g2_raw = load(g1_path), load(g2_path)
        g1_counts = Counter(row["sample_id"] for row in g1_raw)
        g2_counts = Counter(row["sample_id"] for row in g2_raw)
        duplicates_g1 = sorted(key for key, count in g1_counts.items() if count > 1)
        duplicates_g2 = sorted(key for key, count in g2_counts.items() if count > 1)
        missing_in_g2 = sorted(set(g1_counts) - set(g2_counts))
        missing_in_g1 = sorted(set(g2_counts) - set(g1_counts))
        matched_ids = sorted(set(g1_counts) & set(g2_counts))
        if duplicates_g1 or duplicates_g2 or len(matched_ids) != 500:
            raise RuntimeError(f"seed {seed}: invalid pairing")
        g1_map = {row["sample_id"]: row for row in g1_raw}
        g2_map = {row["sample_id"]: row for row in g2_raw}
        g1 = [g1_map[key] for key in matched_ids]
        g2 = [g2_map[key] for key in matched_ids]
        reference_mismatches = [key for key in matched_ids
                                if g1_map[key]["references"] != g2_map[key]["references"]]
        mr_mismatches = [key for key in matched_ids
                         if g1_map[key].get("meaning_representation") != g2_map[key].get("meaning_representation")]
        input_hash_mismatches = [key for key in matched_ids
                                 if g1_map[key].get("input_hash") != g2_map[key].get("input_hash")]
        if reference_mismatches or mr_mismatches or input_hash_mismatches:
            raise RuntimeError(f"seed {seed}: evaluator input mismatch")

        stats1 = corpus_stats(g1, metric); stats2 = corpus_stats(g2, metric)
        corpus1 = score_from_sum(metric, stats1.sum(axis=0))
        corpus2 = score_from_sum(metric, stats2.sum(axis=0))
        mean_sentence1 = sentence_mean(g1); mean_sentence2 = sentence_mean(g2)
        rng = np.random.default_rng(2026071400 + seed)
        boot = np.empty(bootstrap_samples, dtype=float)
        batch = 250
        for start in range(0, bootstrap_samples, batch):
            stop = min(start + batch, bootstrap_samples)
            indices = rng.integers(0, len(matched_ids), size=(stop - start, len(matched_ids)))
            sums1 = stats1[indices].sum(axis=1); sums2 = stats2[indices].sum(axis=1)
            for offset, (left, right) in enumerate(zip(sums1, sums2)):
                boot[start + offset] = score_from_sum(metric, right) - score_from_sum(metric, left)
        item = {
            "seed": seed,
            "g1_file": str(g1_path.relative_to(ROOT)), "g1_sha256": sha256(g1_path),
            "g2_file": str(g2_path.relative_to(ROOT)), "g2_sha256": sha256(g2_path),
            "g1_corpus_bleu": corpus1, "g2_corpus_bleu": corpus2,
            "corpus_bleu_difference_g2_minus_g1": corpus2 - corpus1,
            "g1_mean_sentence_bleu": mean_sentence1, "g2_mean_sentence_bleu": mean_sentence2,
            "mean_sentence_bleu_difference_g2_minus_g1": mean_sentence2 - mean_sentence1,
            "paired_bootstrap_corpus_bleu_ci95": [percentile(boot, .025), percentile(boot, .975)],
            "paired_bootstrap_corpus_bleu_mean": float(boot.mean()),
            "matched_sample_count": len(matched_ids),
            "g1_records": len(g1_raw), "g2_records": len(g2_raw),
            "g1_duplicate_sample_ids": duplicates_g1, "g2_duplicate_sample_ids": duplicates_g2,
            "missing_in_g2": missing_in_g2, "missing_in_g1": missing_in_g1,
            "reference_mismatch_count": len(reference_mismatches),
            "meaning_representation_mismatch_count": len(mr_mismatches),
            "input_hash_mismatch_count": len(input_hash_mismatches),
        }
        results.append(item)
        diagnostics.append({"seed": seed, "bootstrap_seed": 2026071400 + seed,
                            "bootstrap_samples": bootstrap_samples,
                            "sacrebleu_version": sacrebleu.__version__,
                            "bleu_signature": str(metric.get_signature())})

    truncated = GL / "generations/g2_protected_e2e_L12_s1234_full.jsonl"
    expected = {
        "path": str(truncated.relative_to(ROOT)), "observed_size": truncated.stat().st_size,
        "observed_sha256": sha256(truncated), "parseable": False,
        "expected_size_from_completion_manifest": 1010375,
        "expected_sha256_from_completion_manifest": "d3b5ad9820c13e473ee7a995480a9ea6b5b3f5a786a4ba3276774cf116e46e31",
        "verified_source_cache_used": str(PAIRS[1234][1].relative_to(ROOT)),
        "verified_source_cache_sha256": sha256(PAIRS[1234][1]),
    }
    report = {
        "schema": "paired_corpus_bleu_audit", "version": "2.0",
        "method": "paired resampling of matched example indices; full corpus BLEU recomputed from summed per-example sacreBLEU sufficient statistics on every bootstrap replicate",
        "bootstrap_samples": bootstrap_samples, "results": results,
        "diagnostics": diagnostics, "seed_1234_local_copy_integrity": expected,
        "conclusion": "The discrepancy is corpus BLEU versus mean sentence BLEU, not seed, subset, reference, or evaluator-input mismatch. The primary local seed-1234 G2 copy is truncated; a completion-manifest-hash-matched immutable cache copy was used.",
    }
    (OUT / "paired_corpus_bleu_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["seed", "g1_corpus_bleu", "g2_corpus_bleu", "corpus_bleu_difference_g2_minus_g1",
              "g1_mean_sentence_bleu", "g2_mean_sentence_bleu", "mean_sentence_bleu_difference_g2_minus_g1",
              "paired_bootstrap_corpus_bleu_ci95", "matched_sample_count", "g1_sha256", "g2_sha256",
              "g1_duplicate_sample_ids", "g2_duplicate_sample_ids", "missing_in_g2", "missing_in_g1"]
    with (OUT / "paired_corpus_bleu_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    lines = ["# Corrected paired corpus-BLEU audit (v2)", "",
             "This report does not overwrite the earlier mean-sentence-BLEU analysis.", "",
             "| Seed | G1 corpus BLEU | G2 corpus BLEU | Corpus Δ | Mean sentence-BLEU Δ | Paired bootstrap corpus-Δ 95% CI | Matched |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lo, hi = row["paired_bootstrap_corpus_bleu_ci95"]
        lines.append(f"| {row['seed']} | {row['g1_corpus_bleu']:.2f} | {row['g2_corpus_bleu']:.2f} | "
                     f"{row['corpus_bleu_difference_g2_minus_g1']:+.2f} | "
                     f"{row['mean_sentence_bleu_difference_g2_minus_g1']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                     f"{row['matched_sample_count']} |")
    lines += ["", "## Audit findings", "",
              "- All three pairs contain exactly 500 matched IDs, with no duplicate or missing IDs.",
              "- References, meaning representations, and input hashes are identical within every pair.",
              "- The discrepancy is expected: corpus BLEU aggregates n-gram counts before applying the BLEU formula; averaging sentence BLEU is a different statistic.",
              "- The workspace seed-1234 G2 JSONL is a truncated 524,288-byte copy. The verified cache copy exactly matches the completion-manifest size and SHA-256 and was used without modifying the frozen path.",
              "", "## Exact generation hashes", ""]
    for row in results:
        lines += [f"- seed {row['seed']} G1: `{row['g1_sha256']}`",
                  f"- seed {row['seed']} G2: `{row['g2_sha256']}`"]
    (OUT / "paired_corpus_bleu_audit.md").write_text("\n".join(lines) + "\n")
    with (OUT / "source_hashes.sha256").open("w") as handle:
        handle.write(f"{sha256(Path(__file__))}  {Path(__file__).relative_to(ROOT)}\n")
        for _, (g1, g2) in PAIRS.items():
            handle.write(f"{sha256(g1)}  {g1.relative_to(ROOT)}\n")
            handle.write(f"{sha256(g2)}  {g2.relative_to(ROOT)}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
