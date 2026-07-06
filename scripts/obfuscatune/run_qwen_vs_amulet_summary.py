"""Merge ObfuscaTune-Qwen results with amulet-style results into one table.

Reads ObfuscaTune-Qwen JSON rows from ``outputs/obfuscatune/qwen_*`` and,
optionally, amulet-style rows from a directory / files given via --amulet-glob.
Emits a unified markdown + csv. Because the two schemes have DIFFERENT threat
models, the security columns are annotated, not merged into a single verdict.

Example:
    python scripts/obfuscatune/run_qwen_vs_amulet_summary.py
    python scripts/obfuscatune/run_qwen_vs_amulet_summary.py --amulet-glob 'results/ours_amulet/**/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from _common import OUTPUT_ROOT
from _qwen_common import REPO_ROOT

from pllo.experiments.experiment_registry import (  # noqa: E402
    OBFUSCATUNE_QWEN_METHODS,
)


def _load_json(pattern: str) -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(pattern, recursive=True)):
        try:
            doc = json.loads(Path(fp).read_text())
        except Exception:
            continue
        rows.append({"_file": fp, "_doc": doc})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amulet-glob", default=None,
                   help="optional glob of amulet-style result JSONs to include")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print(f"[dry-run] scan {OUTPUT_ROOT}/qwen_* ; amulet-glob={args.amulet_glob}")
        return

    obf = []
    for sub in ("qwen_correctness", "qwen_prefill_decode", "qwen_generation"):
        obf += _load_json(str(OUTPUT_ROOT / sub / "*.json"))

    md = ["# ObfuscaTune-Qwen vs amulet-style — unified comparison", "",
          "> ObfuscaTune and our amulet-style / trusted-shortcut scheme have "
          "**different threat models**. Correctness + cost are comparable; "
          "security columns describe different guarantees (see notes).", "",
          "## ObfuscaTune-Qwen rows", "",
          "| method | mode | max_abs_err | argmax/token match | kv_cache | public_qkv |",
          "|---|---|---|---|---|---|"]
    csv = [["source", "method", "mode", "max_abs_error", "match", "protected_kv_cache", "public_qkv"]]
    for r in obf:
        d = r["_doc"]
        method = d.get("method", "?")
        mode = d.get("mode", d.get("prefill") and "prefill/decode" or "?")
        corr = d.get("correctness", {})
        mae = corr.get("max_abs_error")
        if mae is None and "prefill" in d:
            mae = d["prefill"].get("max_abs_error")
        match = corr.get("logits_argmax_match_rate", corr.get("token_match_rate"))
        sp = d.get("security_proxy", {})
        kv = sp.get("protected_kv_cache", "false")
        pq = sp.get("public_qkv", True)
        mae_s = f"{mae:.2e}" if isinstance(mae, (int, float)) else str(mae)
        match_s = f"{match:.3f}" if isinstance(match, (int, float)) else str(match)
        md.append(f"| {method} | {mode} | {mae_s} | {match_s} | {kv} | {pq} |")
        csv.append([Path(r["_file"]).name, method, str(mode), mae_s, match_s, str(kv), str(pq)])

    if args.amulet_glob:
        amulet = _load_json(args.amulet_glob)
        md += ["", "## Amulet-style rows (as found)", "",
               f"_loaded {len(amulet)} file(s) from `{args.amulet_glob}`_", ""]
        for r in amulet:
            md.append(f"- `{Path(r['_file']).name}`")
    else:
        md += ["", "_No --amulet-glob given; ObfuscaTune-Qwen rows only._", ""]

    md += ["", "## Registered ObfuscaTune-Qwen methods", "",
           "| method | protects_base_model | protects_kv_cache | uses_tee | comparable_to_amulet |",
           "|---|---|---|---|---|"]
    for name, spec in OBFUSCATUNE_QWEN_METHODS.items():
        md.append(f"| {name} | {spec.protects_base_model} | {spec.protects_kv_cache} | "
                  f"{spec.uses_tee} | {spec.comparable_to_amulet_style} |")
    md += ["", "**Threat-model note:** " + next(iter(OBFUSCATUNE_QWEN_METHODS.values())).notes, ""]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "qwen_vs_amulet_summary.md").write_text("\n".join(md) + "\n")
    (OUTPUT_ROOT / "qwen_vs_amulet_summary.csv").write_text(
        "\n".join(",".join(c for c in row) for row in csv) + "\n")
    print(f"wrote {OUTPUT_ROOT}/qwen_vs_amulet_summary.md (+ .csv); {len(obf)} obfuscatune rows")


if __name__ == "__main__":
    main()
