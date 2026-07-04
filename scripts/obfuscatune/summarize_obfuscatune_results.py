"""Summarize ObfuscaTune outputs into a single Markdown + CSV table.

Scans ``outputs/obfuscatune/{correctness,condition_sweep,latency,lora_smoke}``
for JSON results and emits:
    outputs/obfuscatune/summary/obfuscatune_summary.md
    outputs/obfuscatune/summary/obfuscatune_summary.csv

Also emits a unified-schema comparison stub (``obfuscatune_comparison_rows.json``)
using the shared schema so ObfuscaTune rows can later be merged with our
amulet-style / trusted-shortcut rows. Never fabricates rows: only what is found.

Example:
    python scripts/obfuscatune/summarize_obfuscatune_results.py
"""

from __future__ import annotations

import argparse
import json

from _common import OUTPUT_ROOT, write_json


def _load_all(subdir: str) -> list[dict]:
    d = OUTPUT_ROOT / subdir
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        print(f"[dry-run] would scan {OUTPUT_ROOT}/{{correctness,condition_sweep,latency,lora_smoke}}")
        return

    md = ["# ObfuscaTune baseline — results summary", ""]
    csv_rows = [["section", "mode_or_kappa", "metric", "value"]]
    comparison_rows: list[dict] = []

    # correctness
    corr = _load_all("correctness")
    if corr:
        md += ["## Correctness (obfuscated vs unprotected logits)", "",
               "| mode | max_abs_err | rel_L2 | argmax_match | nan/inf |",
               "|---|---|---|---|---|"]
        for doc in corr:
            for r in doc.get("results", []):
                md.append(f"| {r['mode']} | {r['max_abs_error_vs_unprotected']:.3e} | "
                          f"{r['relative_l2_error']:.3e} | {r['logits_argmax_match_rate']:.3f} | "
                          f"{r['nan_or_inf_count']} |")
                csv_rows.append(["correctness", r["mode"], "max_abs_error",
                                 r["max_abs_error_vs_unprotected"]])
        md.append("")

    # condition sweep
    sweep = _load_all("condition_sweep")
    if sweep:
        md += ["## Condition-number sweep (Table 2 trend)", "",
               "| kappa | matrix | rel_L2 | max_abs | argmax_match |",
               "|---|---|---|---|---|"]
        for doc in sweep:
            for r in doc.get("results", []):
                cn = "random" if r["condition_number"] is None else f"{r['condition_number']:g}"
                md.append(f"| {cn} | {r['matrix_type']} | {r['relative_l2_error']:.3e} | "
                          f"{r['max_abs_error']:.3e} | {r['logits_argmax_match_rate']:.3f} |")
                csv_rows.append(["condition_sweep", cn, "relative_l2_error", r["relative_l2_error"]])
        md.append("")

    # latency
    lat = _load_all("latency")
    if lat:
        md += ["## Latency (CPU simulator proxy — NOT a real-TEE measurement)", "",
               "| mode | mean_ms | p50 | p95 | slowdown |", "|---|---|---|---|---|"]
        for doc in lat:
            for r in doc.get("results", []):
                sd = r["slowdown_vs_unprotected"]
                md.append(f"| {r['mode']} | {r['wall_time_ms_mean']:.2f} | "
                          f"{r['wall_time_ms_p50']:.2f} | {r['wall_time_ms_p95']:.2f} | "
                          f"{sd:.2f}x |" if sd else
                          f"| {r['mode']} | {r['wall_time_ms_mean']:.2f} | "
                          f"{r['wall_time_ms_p50']:.2f} | {r['wall_time_ms_p95']:.2f} | - |")
                csv_rows.append(["latency", r["mode"], "wall_time_ms_mean", r["wall_time_ms_mean"]])
        md.append("")

    # lora smoke
    lora = _load_all("lora_smoke")
    if lora:
        md += ["## LoRA smoke (loop runs + loss drops; NOT a paper accuracy result)", "",
               "| backend | initial_loss | final_loss | decreased |", "|---|---|---|---|"]
        for doc in lora:
            c = doc.get("config", {})
            md.append(f"| {c.get('lora_backend', '?')} | {doc['initial_loss']:.4f} | "
                      f"{doc['final_loss']:.4f} | {doc['loss_decreased']} |")
            csv_rows.append(["lora_smoke", c.get("lora_backend", "?"), "final_loss", doc["final_loss"]])
        md.append("")

    if not (corr or sweep or lat or lora):
        md += ["_No results found. Run the correctness / sweep / latency / lora scripts first._", ""]

    md += ["## Note on cross-scheme comparison", "",
           "ObfuscaTune and our amulet-style / trusted-shortcut scheme have "
           "**different threat models** (ObfuscaTune: secret model weights + "
           "private data under an authenticated TEE, with plaintext Q/K/V exposed "
           "outside; ours: public base weights, protecting user input / LoRA / KV "
           "cache / logits). Only correctness and cost columns are directly "
           "comparable; the `security_proxy.notes` field must be kept.", ""]

    base = OUTPUT_ROOT / "summary"
    base.mkdir(parents=True, exist_ok=True)
    (base / "obfuscatune_summary.md").write_text("\n".join(md) + "\n")
    (base / "obfuscatune_summary.csv").write_text(
        "\n".join(",".join(str(c) for c in row) for row in csv_rows) + "\n")
    write_json(base / "obfuscatune_comparison_rows.json", comparison_rows)
    print(f"wrote {base}/obfuscatune_summary.md (+ .csv)")


if __name__ == "__main__":
    main()
