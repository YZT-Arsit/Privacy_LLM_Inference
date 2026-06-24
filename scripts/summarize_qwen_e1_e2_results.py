"""Aggregate flattened E1/E2 JSON outputs into paper-ready tables.

Consumes the flattened E1 reports + E2 token-scaling reports (natural and/or
fixed-padded) and emits a Markdown table set, a machine-readable CSV, and a JSON
summary. Validates the paper-critical invariants before writing anything: every
paper-critical field must be present and non-None (except ``padded_seq_len`` in
``natural_prompt`` mode), ``attention_mask_used`` must be true,
``tee_used_on_gpu`` must be false, and no plaintext/secret fields may have leaked.

stdlib only. Example::

    python scripts/summarize_qwen_e1_e2_results.py \\
        --e1-natural outputs/e1_nolora_qwen_natural_s128_d64_flat.json \\
        --e1-padded  outputs/e1_nolora_qwen_padded_s128_d64_flat.json \\
        --e2-natural outputs/e2_token_scaling_qwen_natural_s128_flat.json \\
        --e2-padded  outputs/e2_token_scaling_qwen_padded_s128_flat.json \\
        --output-md   outputs/qwen_e1_e2_paper_tables.md \\
        --output-csv  outputs/qwen_e1_e2_paper_tables.csv \\
        --output-json outputs/qwen_e1_e2_paper_tables.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Table 1: E1 main no-LoRA generation.
E1_FIELDS = [
    "context_mode", "seq_len_requested", "effective_prompt_len",
    "padded_seq_len", "max_new_tokens", "attention_mask_used",
    "teacher_forced_top1_match_rate_hf_plain",
    "teacher_forced_top1_match_rate_hf_masked",
    "teacher_forced_top1_match_rate_plain_masked",
    "plain_vs_masked_token_match_rate", "topk_overlap", "logits_max_abs_error",
    "logits_mean_abs_error", "logits_relative_l2_error", "latency_s",
    "peak_gpu_memory_mb", "trusted_bytes", "gpu_bytes", "tee_used_on_gpu",
]
# Table 2: E2 token scaling.
E2_FIELDS = [
    "context_mode", "max_new_tokens", "seq_len_requested",
    "effective_prompt_len", "padded_seq_len",
    "teacher_forced_top1_match_rate_hf_masked",
    "teacher_forced_top1_match_rate_plain_masked",
    "plain_vs_masked_token_match_rate", "topk_overlap", "logits_max_abs_error",
    "latency_s", "peak_gpu_memory_mb", "trusted_bytes", "gpu_bytes",
    "tee_used_on_gpu",
]
NOTES = [
    "natural_prompt mode: seq_len_requested is a prompt budget while "
    "effective_prompt_len is the actual tokenized prompt length.",
    "fixed_padded mode: the input is padded to seq_len_requested with "
    "attention_mask_used=true.",
    "long bf16 free-running HF exact sequence equality is NOT the primary "
    "correctness criterion; teacher-forced top-1 and plain-vs-masked agreement "
    "are the main correctness metrics.",
    "tee_used_on_gpu=false means no TEE computation runs on the H800.",
]


class ValidationError(Exception):
    pass


def _check_record(rec: dict, fields: list[str], where: str) -> list[str]:
    """Return a list of validation violations for one record (E1 report / E2 row)."""
    problems: list[str] = []
    natural = rec.get("context_mode") == "natural_prompt"
    for f in fields:
        if f not in rec:
            problems.append(f"{where}: missing field {f!r}")
            continue
        if rec[f] is None:
            if f == "padded_seq_len" and natural:
                continue                         # allowed: no padding in natural
            problems.append(f"{where}: field {f!r} is None")
    if rec.get("attention_mask_used") is not True:
        problems.append(f"{where}: attention_mask_used must be true "
                        f"(got {rec.get('attention_mask_used')!r})")
    if rec.get("tee_used_on_gpu") is not False:
        problems.append(f"{where}: tee_used_on_gpu must be false "
                        f"(got {rec.get('tee_used_on_gpu')!r})")
    for sec in ("leaked_secret_fields", "gpu_visible_plaintext_fields"):
        # these may live on the E1 report / E2 row; if present they must be empty
        if sec in rec and rec[sec]:
            problems.append(f"{where}: {sec} non-empty: {rec[sec]!r}")
    return problems


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def collect(args) -> tuple[list[dict], list[dict], list[str]]:
    """Load inputs into E1 rows + E2 rows; returns (e1_rows, e2_rows, problems)."""
    e1_rows: list[dict] = []
    e2_rows: list[dict] = []
    problems: list[str] = []

    for label, path in (("e1-natural", args.e1_natural),
                        ("e1-padded", args.e1_padded)):
        rep = _load(path)
        if rep is None:
            continue
        problems += _check_record(rep, E1_FIELDS, f"{label}({path})")
        e1_rows.append({k: rep.get(k) for k in E1_FIELDS})

    for label, path in (("e2-natural", args.e2_natural),
                        ("e2-padded", args.e2_padded)):
        rep = _load(path)
        if rep is None:
            continue
        rows = rep.get("rows", [])
        if not rows:
            problems.append(f"{label}({path}): no rows")
        for i, row in enumerate(rows):
            # context_mode may live on the parent report, not each row
            if row.get("context_mode") is None and rep.get("context_mode"):
                row = {**row, "context_mode": rep["context_mode"]}
            problems += _check_record(row, E2_FIELDS, f"{label}({path}) row[{i}]")
            e2_rows.append({k: row.get(k) for k in E2_FIELDS})
    return e1_rows, e2_rows, problems


def _md_table(rows: list[dict], fields: list[str]) -> list[str]:
    if not rows:
        return ["_(no rows)_", ""]
    out = ["| " + " | ".join(fields) + " |",
           "|" + "|".join(["---"] * len(fields)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(f)) for f in fields) + " |")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-natural", default=None)
    ap.add_argument("--e1-padded", default=None)
    ap.add_argument("--e2-natural", default=None)
    ap.add_argument("--e2-padded", default=None)
    ap.add_argument("--output-md", default="outputs/qwen_e1_e2_paper_tables.md")
    ap.add_argument("--output-csv", default="outputs/qwen_e1_e2_paper_tables.csv")
    ap.add_argument("--output-json",
                    default="outputs/qwen_e1_e2_paper_tables.json")
    ap.add_argument("--allow-validation-failure", action="store_true",
                    help="write outputs + report problems but exit 0 (debug only)")
    args = ap.parse_args()

    if not any((args.e1_natural, args.e1_padded, args.e2_natural,
                args.e2_padded)):
        ap.error("provide at least one of --e1-natural/--e1-padded/"
                 "--e2-natural/--e2-padded")

    e1_rows, e2_rows, problems = collect(args)

    if problems and not args.allow_validation_failure:
        print("VALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)

    summary = {
        "stage": "qwen_e1_e2_paper_tables",
        "validation_passed": not problems,
        "validation_problems": problems,
        "e1_table": e1_rows,
        "e2_table": e2_rows,
        "notes": NOTES,
    }

    # JSON
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown
    if args.output_md:
        L = ["# Qwen2.5-7B no-LoRA E1/E2 paper tables", "",
             f"validation_passed: **{summary['validation_passed']}**", "",
             "## Table 1 — E1 main no-LoRA masked generation", ""]
        L += _md_table(e1_rows, E1_FIELDS)
        L += ["## Table 2 — E2 token scaling", ""]
        L += _md_table(e2_rows, E2_FIELDS)
        L += ["## Notes", ""] + [f"- {n}" for n in NOTES]
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(L) + "\n", encoding="utf-8")

    # CSV (one combined file with a 'table' column + union of fields)
    if args.output_csv:
        union = ["table"] + E1_FIELDS + [f for f in E2_FIELDS
                                         if f not in E1_FIELDS]
        p = Path(args.output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=union)
            w.writeheader()
            for r in e1_rows:
                w.writerow({"table": "E1", **r})
            for r in e2_rows:
                w.writerow({"table": "E2", **r})

    print("=== Qwen E1/E2 paper tables ===")
    print(f"validation_passed={summary['validation_passed']} "
          f"e1_rows={len(e1_rows)} e2_rows={len(e2_rows)}")
    if problems:
        print(f"validation_problems={len(problems)} (allowed by flag)")
    for path in (args.output_md, args.output_csv, args.output_json):
        if path:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
