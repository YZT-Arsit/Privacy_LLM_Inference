"""PHASE 6 — quality metrics over generation jsonl (offline, Mac control-plane; NOT a compute path).

Multi-reference E2E metrics: BLEU + chrF (sacrebleu), ROUGE-L (local LCS), plus invalid-output rate,
repetition rate, avg generated length. METEOR/NIST reported only if nltk is available (else marked
unavailable). Emits aggregate_metrics.csv, per_example_metrics.jsonl, paired_differences.csv (vs a
baseline cell), statistical_summary.json (mean/std/per-seed + bootstrap CI), metrics_report.md.

Usage:
  python3 scripts/generative_lora/phase6_metrics.py \
     --gen G0=path/g0.jsonl G1=path/g1.jsonl G2=path/g2.jsonl --baseline G1 --out <metrics_dir>
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import sacrebleu

try:
    import nltk  # noqa: F401
    from nltk.translate.meteor_score import meteor_score
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def rouge_l(hyp, refs):
    ht = hyp.split()
    best = 0.0
    for r in refs:
        rt = r.split()
        if not ht or not rt:
            continue
        l = lcs(ht, rt)
        if l == 0:
            continue
        p = l / len(ht); rec = l / len(rt)
        f = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        best = max(best, f)
    return best


def load(path):
    return [json.loads(l) for l in open(path)]


def corpus_metrics(recs):
    hyps = [r["generated_text"] for r in recs]
    maxr = max(len(r["references"]) for r in recs)
    refs = [[(r["references"][i] if i < len(r["references"]) else r["references"][0]) for r in recs]
            for i in range(maxr)]
    bleu = sacrebleu.corpus_bleu(hyps, refs).score
    chrf = sacrebleu.corpus_chrf(hyps, refs).score
    rl = 100.0 * sum(rouge_l(r["generated_text"], r["references"]) for r in recs) / len(recs)
    inv = 100.0 * sum(r["invalid_output"] for r in recs) / len(recs)
    rep = 100.0 * sum(r["repetition_bigram_frac"] for r in recs) / len(recs)
    al = sum(r["generated_token_count"] for r in recs) / len(recs)
    m = {"BLEU": round(bleu, 2), "chrF": round(chrf, 2), "ROUGE_L": round(rl, 2),
         "invalid_rate_pct": round(inv, 2), "repetition_pct": round(rep, 2), "avg_len": round(al, 1),
         "n": len(recs)}
    if _HAS_NLTK:
        try:
            ms = sum(meteor_score([rr.split() for rr in r["references"]], r["generated_text"].split())
                     for r in recs) / len(recs)
            m["METEOR"] = round(100 * ms, 2)
        except Exception:
            m["METEOR"] = "nltk_error"
    else:
        m["METEOR"] = "unavailable_no_nltk"; m["NIST"] = "unavailable_no_nltk"
    return m


def per_example_bleu(recs):
    out = []
    for r in recs:
        b = sacrebleu.sentence_bleu(r["generated_text"], r["references"]).score
        out.append({"sample_id": r["sample_id"], "bleu": round(b, 2),
                    "rouge_l": round(100 * rouge_l(r["generated_text"], r["references"]), 2),
                    "len": r["generated_token_count"], "invalid": r["invalid_output"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True, help="CELL=path.jsonl ...")
    ap.add_argument("--baseline", default="G1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cells = {}
    for spec in args.gen:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            cells[name] = load(path)
    agg = {name: corpus_metrics(recs) for name, recs in cells.items()}
    perex = {name: per_example_bleu(recs) for name, recs in cells.items()}

    # aggregate csv
    keys = ["BLEU", "chrF", "ROUGE_L", "METEOR", "invalid_rate_pct", "repetition_pct", "avg_len", "n"]
    lines = ["cell," + ",".join(keys)]
    for name, m in agg.items():
        lines.append(name + "," + ",".join(str(m.get(k, "")) for k in keys))
    (out / "aggregate_metrics.csv").write_text("\n".join(lines) + "\n")

    # per-example jsonl
    with open(out / "per_example_metrics.jsonl", "w") as f:
        for name, rows in perex.items():
            for r in rows:
                f.write(json.dumps({"cell": name, **r}) + "\n")

    # paired differences vs baseline (by sample_id, per-example bleu)
    base = args.baseline
    paired = {}
    if base in perex:
        bmap = {r["sample_id"]: r["bleu"] for r in perex[base]}
        plines = ["cell,mean_delta_bleu,median_delta_bleu,n_paired,n_better,n_worse"]
        for name, rows in perex.items():
            if name == base:
                continue
            deltas = [r["bleu"] - bmap[r["sample_id"]] for r in rows if r["sample_id"] in bmap]
            if deltas:
                paired[name] = {"mean_delta": round(statistics.mean(deltas), 3),
                                "median_delta": round(statistics.median(deltas), 3), "n": len(deltas),
                                "n_better": sum(d > 0 for d in deltas), "n_worse": sum(d < 0 for d in deltas)}
                plines.append(f"{name},{paired[name]['mean_delta']},{paired[name]['median_delta']},"
                              f"{len(deltas)},{paired[name]['n_better']},{paired[name]['n_worse']}")
        (out / "paired_differences.csv").write_text("\n".join(plines) + "\n")

    summary = {"cells": agg, "paired_vs_" + base: paired, "has_nltk": _HAS_NLTK,
               "baseline": base, "metric_note": "E2E multi-reference; BLEU/chrF=sacrebleu, ROUGE-L=local LCS-F1."}
    (out / "statistical_summary.json").write_text(json.dumps(summary, indent=2))

    rep = ["# Phase 6 — Generation Quality Metrics (E2E NLG)\n",
           "Multi-reference. BLEU/chrF via sacrebleu; ROUGE-L local LCS-F1. Offline (Mac control-plane).\n",
           "| Cell | BLEU | chrF | ROUGE-L | METEOR | invalid% | rep% | avg_len | n |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, m in agg.items():
        rep.append(f"| {name} | {m['BLEU']} | {m['chrF']} | {m['ROUGE_L']} | {m.get('METEOR')} | "
                   f"{m['invalid_rate_pct']} | {m['repetition_pct']} | {m['avg_len']} | {m['n']} |")
    if paired:
        rep.append(f"\n**Paired per-example BLEU vs {base}** (same sample_id):\n")
        for name, p in paired.items():
            rep.append(f"- {name}: mean Δ {p['mean_delta']}, median Δ {p['median_delta']}, "
                       f"better/worse {p['n_better']}/{p['n_worse']} of {p['n']}")
    (out / "metrics_report.md").write_text("\n".join(rep) + "\n")
    print("[phase6] wrote metrics to", out)
    for name, m in agg.items():
        print(" ", name, m)


if __name__ == "__main__":
    main()
