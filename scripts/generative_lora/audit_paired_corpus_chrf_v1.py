#!/usr/bin/env python3
"""Audit chrF using paired corpus-level and hierarchical corpus-level bootstrap."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
GL = ROOT / "results/aaai_private_base/generative_lora"
OUT = GL / "extension/paired_statistics_corrected_v3_chrf"
SEEDS = (7, 1234, 2025)
PAIRS = {
    7: (GL / "generations/g1_final_s7_test.jsonl",
        GL / "generations/g2_protected_e2e_L12_s7_extension.jsonl"),
    1234: (GL / "generations/g1_final_s1234_test.jsonl",
           ROOT / "results/aaai_private_base/offline_analysis/confound_controlled_20260714/source_cache/g2_protected_e2e_L12_s1234_full.jsonl"),
    2025: (GL / "generations/g1_final_s2025_test.jsonl",
           GL / "generations/g2_protected_e2e_L12_s2025_extension.jsonl"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def refs(rows: list[dict]) -> list[list[str]]:
    max_refs = max(len(row["references"]) for row in rows)
    return [[row["references"][i] if i < len(row["references"]) else row["references"][0]
             for row in rows] for i in range(max_refs)]


def score(metric: CHRF, stats: np.ndarray) -> float:
    return float(metric._compute_score_from_stats(stats.tolist()).score)


def extract(metric: CHRF, rows: list[dict]) -> np.ndarray:
    hypotheses = [row["generated_text"] for row in rows]
    return np.asarray(metric._extract_corpus_statistics(hypotheses, refs(rows)), dtype=np.int64)


def bootstrap_seed(metric: CHRF, left: np.ndarray, right: np.ndarray,
                   samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); n = left.shape[0]
    values = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        stop = min(start + 250, samples)
        indices = rng.integers(0, n, size=(stop - start, n))
        ls = left[indices].sum(axis=1); rs = right[indices].sum(axis=1)
        for offset, (a, b) in enumerate(zip(ls, rs)):
            values[start + offset] = score(metric, b) - score(metric, a)
    return values


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite corrected chrF audit: {OUT}")
    OUT.mkdir(parents=True)
    samples, base_seed = 20000, 20260714
    metric = CHRF()
    per_seed = []
    stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in SEEDS:
        p1, p2 = PAIRS[seed]
        r1, r2 = load(p1), load(p2)
        m1 = {row["sample_id"]: row for row in r1}; m2 = {row["sample_id"]: row for row in r2}
        ids = sorted(set(m1) & set(m2))
        if len(ids) != 500 or len(m1) != 500 or len(m2) != 500 or set(m1) != set(m2):
            raise RuntimeError(f"seed {seed}: pairing failure")
        if any(m1[i]["references"] != m2[i]["references"] or
               m1[i].get("input_hash") != m2[i].get("input_hash") for i in ids):
            raise RuntimeError(f"seed {seed}: evaluator input mismatch")
        left_rows, right_rows = [m1[i] for i in ids], [m2[i] for i in ids]
        left, right = extract(metric, left_rows), extract(metric, right_rows)
        stats[seed] = left, right
        c1, c2 = score(metric, left.sum(axis=0)), score(metric, right.sum(axis=0))
        old1 = float(np.mean([sacrebleu.sentence_chrf(row["generated_text"], row["references"]).score
                             for row in left_rows]))
        old2 = float(np.mean([sacrebleu.sentence_chrf(row["generated_text"], row["references"]).score
                             for row in right_rows]))
        boot = bootstrap_seed(metric, left, right, samples, base_seed + seed)
        per_seed.append({
            "seed": seed, "matched_samples": len(ids),
            "g1_corpus_chrf": c1, "g2_corpus_chrf": c2,
            "corpus_chrf_delta_g2_minus_g1": c2 - c1,
            "paired_corpus_chrf_ci95_low": float(np.quantile(boot, .025)),
            "paired_corpus_chrf_ci95_high": float(np.quantile(boot, .975)),
            "bootstrap_corpus_chrf_mean_delta": float(boot.mean()),
            "g1_mean_sentence_chrf": old1, "g2_mean_sentence_chrf": old2,
            "mean_sentence_chrf_delta_g2_minus_g1": old2 - old1,
            "g1_sha256": sha256(p1), "g2_sha256": sha256(p2),
            "missing_ids": 0, "duplicate_ids": 0, "reference_mismatches": 0,
        })

    rng = np.random.default_rng(base_seed + 99991)
    hierarchy = np.empty(samples, dtype=float)
    for draw in range(samples):
        chosen = rng.integers(0, len(SEEDS), size=len(SEEDS))
        deltas = []
        for selected in chosen:
            seed = SEEDS[int(selected)]; left, right = stats[seed]
            idx = rng.integers(0, left.shape[0], size=left.shape[0])
            deltas.append(score(metric, right[idx].sum(axis=0)) - score(metric, left[idx].sum(axis=0)))
        hierarchy[draw] = float(np.mean(deltas))
    point_deltas = np.asarray([row["corpus_chrf_delta_g2_minus_g1"] for row in per_seed])
    aggregate = {
        "seeds": list(SEEDS), "examples_per_seed": 500,
        "mean_per_seed_corpus_chrf_delta": float(point_deltas.mean()),
        "seed_std": float(point_deltas.std(ddof=1)),
        "hierarchical_corpus_chrf_ci95": [float(np.quantile(hierarchy, .025)),
                                           float(np.quantile(hierarchy, .975))],
        "hierarchical_bootstrap_mean": float(hierarchy.mean()),
        "old_mean_sentence_chrf_delta": float(np.mean([
            row["mean_sentence_chrf_delta_g2_minus_g1"] for row in per_seed])),
        "old_reported_hierarchical_sentence_chrf_ci95": [-2.147507349147633, -0.35867902431435833],
    }
    result = {
        "schema": "paired_corpus_chrf_audit", "version": "1.0",
        "sacrebleu_version": sacrebleu.__version__,
        "chrf_configuration": {"char_order": 6, "word_order": 0, "beta": 2,
                               "lowercase": False, "whitespace": False, "eps_smoothing": False},
        "bootstrap_samples": samples, "bootstrap_seed": base_seed,
        "old_method": "hierarchical bootstrap of per-example sentence-chrF differences",
        "corrected_method": "resample paired example indices and recompute corpus chrF from aggregate sufficient statistics; hierarchical draws resample seeds and then paired indices within every selected seed",
        "per_seed": per_seed, "aggregate": aggregate,
    }
    (OUT / "paired_corpus_chrf_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(per_seed[0])
    with (OUT / "per_seed_corpus_chrf.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(per_seed)
    lines = ["# Corrected paired corpus-chrF audit (v1)", "",
             "The earlier hierarchical interval resampled per-example sentence-chrF differences. This version recomputes corpus chrF after every paired bootstrap draw.", "",
             "| Seed | G1 corpus chrF | G2 corpus chrF | Corpus Δ | Mean sentence-chrF Δ | Paired corpus-Δ 95% CI |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in per_seed:
        lines.append(f"| {row['seed']} | {row['g1_corpus_chrf']:.2f} | {row['g2_corpus_chrf']:.2f} | "
                     f"{row['corpus_chrf_delta_g2_minus_g1']:+.2f} | "
                     f"{row['mean_sentence_chrf_delta_g2_minus_g1']:+.3f} | "
                     f"[{row['paired_corpus_chrf_ci95_low']:+.3f}, {row['paired_corpus_chrf_ci95_high']:+.3f}] |")
    lo, hi = aggregate["hierarchical_corpus_chrf_ci95"]
    lines += ["", "## Three-seed hierarchical result", "",
              f"- Mean of the three seed-level corpus-chrF deltas: {aggregate['mean_per_seed_corpus_chrf_delta']:+.3f}.",
              f"- Hierarchical paired corpus-chrF 95% CI: [{lo:+.3f}, {hi:+.3f}].",
              f"- Earlier sentence-chrF hierarchical CI: [{aggregate['old_reported_hierarchical_sentence_chrf_ci95'][0]:+.3f}, {aggregate['old_reported_hierarchical_sentence_chrf_ci95'][1]:+.3f}].",
              "", "All pairs contain exactly 500 aligned examples with identical references and input hashes; no missing or duplicate IDs were found."]
    (OUT / "paired_corpus_chrf_audit.md").write_text("\n".join(lines) + "\n")
    with (OUT / "source_hashes.sha256").open("w") as handle:
        handle.write(f"{sha256(Path(__file__))}  {Path(__file__).relative_to(ROOT)}\n")
        for seed in SEEDS:
            for path in PAIRS[seed]:
                handle.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
