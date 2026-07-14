#!/usr/bin/env python3
"""Build versioned generation, cost, claim, limitation, and audit tables from parsed artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from canonical_io import BUNDLE, REPO, load_canonical_jsonl, sha256
from recompute_generation_metrics_v2 import reference_documents, rouge_l_f1, score_from_sum, sufficient_statistics

GL = REPO / "results/aaai_private_base/generative_lora"


def g0_metrics() -> dict:
    rows, entry = load_canonical_jsonl("g0_base")
    bleu, chrf = BLEU(), CHRF()
    return {"corpus_bleu": score_from_sum(bleu, sufficient_statistics(rows, bleu).sum(axis=0)),
            "corpus_chrf": score_from_sum(chrf, sufficient_statistics(rows, chrf).sum(axis=0)),
            "rouge_l": float(np.mean([rouge_l_f1(r["generated_text"], r["references"]) for r in rows])),
            "output_length": float(np.mean([r["generated_token_count"] for r in rows])),
            "invalid_rate": 100 * float(np.mean([r["invalid_output"] for r in rows])),
            "repetition_bigram_fraction": 100 * float(np.mean([r["repetition_bigram_frac"] for r in rows])),
            "path": entry["path"], "sha256": entry["sha256"]}


def generation_table() -> dict:
    audit = json.loads((BUNDLE / "generation_metric_audit_v2.json").read_text()); base = g0_metrics()
    rows = [{"method": "G0", "seed": "", **{k: base[k] for k in ("corpus_bleu", "corpus_chrf", "rouge_l", "output_length", "invalid_rate", "repetition_bigram_fraction")},
             "paired_bleu_ci95": "", "source": base["path"], "provenance": "MEASURED"}]
    for item in audit["seed_results"]:
        for method in ("G1", "G2"):
            side = method.lower()
            metrics = item["metrics"]
            rows.append({"method": method, "seed": item["seed"],
                         **{name: metrics[name][side] for name in ("corpus_bleu", "corpus_chrf", "rouge_l", "output_length", "invalid_rate", "repetition_bigram_fraction")},
                         "paired_bleu_ci95": json.dumps(metrics["corpus_bleu"]["paired_bootstrap_ci95"]) if method == "G2" else "",
                         "source": item[f"{side}_path"], "provenance": "MEASURED"})
    fields = list(rows[0])
    with (BUNDLE / "final_generation_table_v2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Final Generation Table (v2)", "",
             "Corpus metrics are paper-facing; all generation JSONL inputs were registry-selected and hash-verified.", "",
             "| Method | Seed | Corpus BLEU | Corpus chrF | ROUGE-L | Length | Invalid % | Repetition bigram % | Paired G2−G1 BLEU CI | Provenance |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        ci = row["paired_bleu_ci95"]
        if ci:
            low, high = json.loads(ci); ci = f"[{low:+.3f}, {high:+.3f}]"
        else: ci = "—"
        lines.append(f"| {row['method']} | {row['seed'] or '—'} | {row['corpus_bleu']:.3f} | {row['corpus_chrf']:.3f} | "
                     f"{row['rouge_l']:.3f} | {row['output_length']:.3f} | {row['invalid_rate']:.3f} | "
                     f"{row['repetition_bigram_fraction']:.3f} | {ci} | {row['provenance']} |")
    s = audit["across_seed_descriptive_statistics"]
    lines += ["", f"Across seeds, corpus BLEU is {s['corpus_bleu']['g1_mean']:.3f} ± {s['corpus_bleu']['g1_sample_sd']:.3f} for G1 and "
              f"{s['corpus_bleu']['g2_mean']:.3f} ± {s['corpus_bleu']['g2_sample_sd']:.3f} for G2 (mean G2−G1 {s['corpus_bleu']['difference_mean']:+.3f}).",
              "The paired confidence intervals test zero difference within each seed. They are not equivalence intervals."]
    (BUNDLE / "final_generation_table_v2.md").write_text("\n".join(lines) + "\n")
    return audit


def unified_cost_table() -> None:
    source = REPO / "results/aaai_private_base/offline_analysis/confound_controlled_20260714/cost/unified_cost_table.csv"
    if sha256(source) != "1d09a2fb6dca19e9d9b9e3102e706a9e7de80003cd9dc765af30298e6bcf9af8":
        raise RuntimeError("frozen cost source hash mismatch")
    with source.open() as f: frozen = list(csv.DictReader(f))
    rows = []
    def add(method, seed="", status="AVAILABLE", train=None, generation=None, total=None, tps=None, adapter=None,
            provenance="MISSING", field_provenance=None, source_path=""):
        fp = field_provenance or {key: provenance for key in ("train", "generation", "total", "tps", "adapter")}
        rows.append({"method": method, "seed": seed, "status": status, "train_wall_sec": train,
                     "generation_wall_sec": generation, "total_wall_sec": total, "tokens_per_sec": tps,
                     "adapter_bytes": adapter, "train_provenance": fp["train"],
                     "generation_provenance": fp["generation"], "total_provenance": fp["total"],
                     "tokens_per_sec_provenance": fp["tps"], "adapter_bytes_provenance": fp["adapter"],
                     "source": source_path})
    profile = json.loads((GL / "generations/g0_base_test.jsonl.profile.json").read_text())
    add("G0", generation=profile["wall_sec"], total=profile["wall_sec"], tps=profile["tokens_per_sec"],
        field_provenance={"train": "MISSING", "generation": "MEASURED", "total": "MEASURED",
                          "tps": "MEASURED", "adapter": "MISSING"},
        source_path="g0_base_test.jsonl.profile.json")
    for method, label in (("G1_plaintext", "G1 FP32"), ("G2_protected_L12", "G2 Exact")):
        selected = [r for r in frozen if r["method"] == method]
        for r in selected:
            add(label, r["seed"], train=float(r["train_wall_sec"]), generation=float(r["generation_wall_sec"]),
                total=float(r["total_measured_wall_sec"]), tps=float(r["tokens_per_sec"]),
                adapter=int(r["adapter_size_bytes"]), provenance="MEASURED", source_path=str(source.relative_to(REPO)))
        for field, key in (("train", "train_wall_sec"), ("generation", "generation_wall_sec"),
                           ("total", "total_measured_wall_sec"), ("tps", "tokens_per_sec"), ("adapter", "adapter_size_bytes")):
            pass
        add(label + " mean", "3 seeds", train=float(np.mean([float(r["train_wall_sec"]) for r in selected])),
            generation=float(np.mean([float(r["generation_wall_sec"]) for r in selected])),
            total=float(np.mean([float(r["total_measured_wall_sec"]) for r in selected])),
            tps=float(np.mean([float(r["tokens_per_sec"]) for r in selected])),
            adapter=float(np.mean([int(r["adapter_size_bytes"]) for r in selected])),
            provenance="DERIVED_FROM_MEASURED_COUNTERS", source_path=str(source.relative_to(REPO)))
    add("G1 BF16 matched", status="MISSING", provenance="MISSING", source_path="No matched 750-step/500-generation BF16 artifact")
    snapshot = json.loads((BUNDLE / "target_ablation_snapshot_v2.json").read_text())
    for event in snapshot["cells"]:
        if event["terminal"]:
            train = sum(json.loads((GL / "protected_runs" / f"{event['run_tag']}.json").read_text())["trajectory"][i]["wall"] for i in range(750))
            gen_profile = json.loads((GL / "generations" / f"g2_protected_{event['run_tag']}.jsonl.profile.json").read_text())
            add(event["cell"], 1234, train=train, generation=gen_profile["wall_sec"], total=train + gen_profile["wall_sec"],
                tps=event["protected_generation_tokens_per_sec"], adapter=event["adapter_bytes"],
                field_provenance={"train": "DERIVED_FROM_MEASURED_COUNTERS", "generation": "MEASURED",
                                  "total": "DERIVED_FROM_MEASURED_COUNTERS", "tps": "MEASURED", "adapter": "MEASURED"},
                source_path="target_ablation_snapshot_v2.json")
        else: add(event["cell"], 1234, status="PENDING", provenance="MISSING", source_path="terminal manifest absent")
    add("ObfuscaTune-style", status="MISSING", provenance="MISSING", source_path="CPU fidelity preparation only")
    fields = list(rows[0])
    with (BUNDLE / "unified_cost_table_v2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    def val(v): return "MISSING" if v in (None, "") else f"{float(v):.3f}"
    lines = ["# Unified Cost Table (v2)", "", "Tags: MEASURED, DERIVED_FROM_MEASURED_COUNTERS, PROJECTED, MISSING.", "",
             "| Method | Seed | Status | Train s [tag] | Generation s [tag] | Total s [tag] | tok/s [tag] | Adapter bytes [tag] |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['seed'] or '—'} | {r['status']} | {val(r['train_wall_sec'])} [{r['train_provenance']}] | "
                     f"{val(r['generation_wall_sec'])} [{r['generation_provenance']}] | "
                     f"{val(r['total_wall_sec'])} [{r['total_provenance']}] | "
                     f"{val(r['tokens_per_sec'])} [{r['tokens_per_sec_provenance']}] | "
                     f"{val(r['adapter_bytes'])} [{r['adapter_bytes_provenance']}] |")
    lines += ["", "No full matched G1 BF16 or ObfuscaTune-style system artifact exists; both remain MISSING. T2–T4 remain pending until strict terminal-manifest admission succeeds."]
    (BUNDLE / "unified_cost_table_v2.md").write_text("\n".join(lines) + "\n")


def narratives(audit: dict) -> None:
    bleu = audit["across_seed_descriptive_statistics"]["corpus_bleu"]
    matrix = f"""# Claim–Evidence Matrix (v2)

| Claim | Status | Evidence | Guardrail |
|---|---|---|---|
| Protected LoRA achieves competitive corpus-level generative quality | SUPPORTED WITH METRIC QUALIFICATION | Mean corpus BLEU: G1 {bleu['g1_mean']:.3f}, G2 {bleu['g2_mean']:.3f}; mean Δ {bleu['difference_mean']:+.3f} | Corpus chrF is lower for G2 on average; report the full metric table |
| Mean corpus BLEU is close to plaintext across three seeds | SUPPORTED DESCRIPTIVELY | Three-seed mean Δ {bleu['difference_mean']:+.3f} | “Close” is descriptive; no equivalence margin was preregistered |
| One seed improves and two show no significant corpus-BLEU difference | SUPPORTED | Seed 1234 CI excludes zero; seed 7 and 2025 CIs include zero | Within-seed paired bootstrap inference only |
| Protected runs exhibit greater across-seed variability | SUPPORTED DESCRIPTIVELY | Corpus-BLEU SD: G2 {bleu['g2_sample_sd']:.3f} vs G1 {bleu['g1_sample_sd']:.3f} | Only three seeds |
| Strict equivalence | NOT ESTABLISHED | No equivalence margin or TOST | Do not infer equivalence from failure to reject zero difference |
| BF16-matched interpretation | PENDING | No matched terminal artifact | Await GPU owner |
| Final T1–T4 cost-quality result | PENDING | Only T1 terminal | T2/T3 running; T4 pending at snapshot |
| Deployable transformed-adapter closure | PENDING | Preparation/negative controls only | Requires deployment artifact |
| Corrected MIA result | PENDING | CPU evaluator ready; shadow collections absent | Old source-confounded AUC=1 excluded |
| ObfuscaTune system comparison | PENDING | 52 CPU fidelity tests only | No matched system measurement |
"""
    (BUNDLE / "claim_evidence_matrix_v2.md").write_text(matrix)
    limitations = """# Limitations (v2)

- Corpus BLEU, corpus chrF, and ROUGE-L do not move uniformly; competitive BLEU must not be generalized to every quality metric.
- Three seeds characterize variability weakly and do not support a formal equivalence conclusion.
- The hierarchical seed-and-example intervals are exploratory, not equivalence tests.
- G2 seed-1234 uses a manifest-matched canonical cache because the ordinary workspace copy is truncated; both copies remain preserved.
- G1 BF16-matched, complete T2–T4, corrected MIA, deployable transformed-adapter, and ObfuscaTune system results are unavailable.
- TPR at 0.1% FPR is resolution-limited in small MIA folds; empirical negative counts and FPR resolution must accompany it.
- T1 retained trusted roundtrip timing but not a trusted-compute/transport-only profiling split; those fields remain MISSING.
- Projected TDX state and tensor bytes omit service, serialization, framing, and authentication overhead.
"""
    (BUNDLE / "limitations_v2.md").write_text(limitations)
    audit_text = """# Reproducibility Audit (v2)

- Canonical generation inputs are selected by logical ID through `canonical_generation_artifact_registry.json`.
- Size and SHA-256 are verified against `canonical_hash_allowlist.json` before JSON parsing; mismatch fails closed.
- The damaged and verified G2 seed-1234 copies were not overwritten.
- Corpus BLEU/chrF bootstrap replicates resample matched IDs and recompute full corpus scores from exact additive sufficient statistics.
- ROUGE-L aggregation is an explicitly documented arithmetic mean of per-example maximum-reference LCS F1.
- Target ingestion is append-only and admits only 750-step, 500-generation, finite, attested, hash-verified bundles; smoke runs are excluded.
- Corrected MIA preprocessing is fit inside the attack-training fold; labels remain outside collected feature records.
- No GPU/TDX action, shared-registry write, active-job script edit, or commit is performed by these tools.
- The ingestion ledger contains an initial T1 observation and a provenance-only corrected T1 event from the idempotence/schema repair; the latest JSON snapshot is canonical.
"""
    (BUNDLE / "reproducibility_audit_v2.md").write_text(audit_text)


def artifact_index() -> None:
    files = sorted(path for path in BUNDLE.rglob("*") if path.is_file() and path.name != "artifact_index_v2.md")
    lines = ["# Artifact Index (v2)", "", "| Artifact | Bytes | SHA-256 |", "|---|---:|---|"]
    for path in files:
        lines.append(f"| `{path.relative_to(BUNDLE)}` | {path.stat().st_size} | `{sha256(path)}` |")
    (BUNDLE / "artifact_index_v2.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    audit = generation_table(); unified_cost_table(); narratives(audit); artifact_index()


if __name__ == "__main__": main()
