"""Paired G2-vs-G1 closure using only the frozen 500-example generation files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import sacrebleu


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def rouge_l(text, refs):
    hyp = text.split()
    best = 0.0
    for ref in refs:
        target = ref.split()
        if not hyp or not target:
            continue
        common = lcs(hyp, target)
        if common:
            precision = common / len(hyp)
            recall = common / len(target)
            best = max(best, 2 * precision * recall / (precision + recall))
    return 100.0 * best


def features(row):
    text, refs = row["generated_text"], row["references"]
    return {
        "BLEU": sacrebleu.sentence_bleu(text, refs).score,
        "chrF": sacrebleu.sentence_chrf(text, refs).score,
        "ROUGE_L": rouge_l(text, refs),
        "length": float(row["generated_token_count"]),
        "repetition_pct": 100.0 * float(row["repetition_bigram_frac"]),
        "invalid_pct": 100.0 * float(bool(row["invalid_output"])),
    }


def percentile(sorted_values, q):
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def summarize(deltas, rng, samples):
    n = len(deltas)
    observed = statistics.mean(deltas)
    boot = []
    for _ in range(samples):
        boot.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    return {
        "n": n,
        "mean_delta": observed,
        "median_delta": statistics.median(deltas),
        "bootstrap_ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "effect_size_cohen_dz": observed / sd if sd > 0 else None,
        "n_positive": sum(x > 0 for x in deltas),
        "n_negative": sum(x < 0 for x in deltas),
        "n_zero": sum(x == 0 for x in deltas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", required=True)
    parser.add_argument("--g2", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    if out_json.exists() or out_md.exists():
        raise SystemExit("refusing to overwrite paired-closure output")

    g1_path, g2_path = Path(args.g1), Path(args.g2)
    g1 = {row["sample_id"]: row for row in load_jsonl(g1_path)}
    g2 = {row["sample_id"]: row for row in load_jsonl(g2_path)}
    ids = sorted(set(g1) & set(g2))
    if len(ids) != 500:
        raise RuntimeError(f"expected 500 aligned examples, found {len(ids)}")
    f1 = {i: features(g1[i]) for i in ids}
    f2 = {i: features(g2[i]) for i in ids}
    metrics = {}
    for metric in f1[ids[0]]:
        rng = random.Random(args.seed + sum(map(ord, metric)))
        metrics[metric] = summarize([f2[i][metric] - f1[i][metric] for i in ids], rng, args.bootstrap)

    result = {
        "schema": "paired_g2_minus_g1_closure",
        "comparison": "G2_L12_s1234 minus G1_s1234",
        "n": len(ids),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "ci_method": "paired percentile bootstrap of mean per-example delta",
        "source_hashes": {
            str(g1_path): hashlib.sha256(g1_path.read_bytes()).hexdigest(),
            str(g2_path): hashlib.sha256(g2_path.read_bytes()).hexdigest(),
        },
        "metrics": metrics,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    lines = [
        "# Final paired analysis: G2 − G1 (seed 1234)", "",
        "All statistics use the existing 500 aligned frozen E2E outputs; generation was not rerun.", "",
        "| Metric | Mean delta | 95% paired bootstrap CI | Cohen dz | + / − / 0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        lo, hi = item["bootstrap_ci95"]
        dz = "N/A" if item["effect_size_cohen_dz"] is None else f'{item["effect_size_cohen_dz"]:.3f}'
        lines.append(f'| {name} | {item["mean_delta"]:.3f} | [{lo:.3f}, {hi:.3f}] | {dz} | '
                     f'{item["n_positive"]}/{item["n_negative"]}/{item["n_zero"]} |')
    bleu_ci = metrics["BLEU"]["bootstrap_ci95"]
    chrf_ci = metrics["chrF"]["bootstrap_ci95"]
    lines.extend(["", "## Interpretation", "",
                  "G2 is competitive with G1 on the frozen evaluation set. The paired sentence-BLEU "
                  + ("interval includes zero, so no significant BLEU difference is established."
                     if bleu_ci[0] <= 0 <= bleu_ci[1] else "interval excludes zero."),
                  "chrF is lower for G2" + (" with a paired interval excluding zero."
                  if chrf_ci[1] < 0 else "; its paired interval does not establish a significant degradation."),
                  "No equivalence or superiority claim is made."])
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
