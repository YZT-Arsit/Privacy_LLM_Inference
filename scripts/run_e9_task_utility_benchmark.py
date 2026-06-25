"""E9 runner: task-utility benchmark over a normalized JSONL dataset.

Backend-pluggable. With no ``--model-path`` / ``--gpu-worker-url`` it runs the
deterministic stub predictor and emits an honest ``dry_run=True,
paper_ready=False`` report (so it is safe in CI / on a laptop). Writes JSON, a
Markdown summary, and a one-row CSV.

Example (dry run)::

    python scripts/run_e9_task_utility_benchmark.py \\
        --dataset-jsonl outputs/bench/mmlu_test.jsonl \\
        --backend plaintext_local --task-type multiple_choice \\
        --max-examples 200 \\
        --output-json outputs/e9_mmlu.json \\
        --output-md outputs/e9_mmlu.md \\
        --output-csv outputs/e9_mmlu.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.runners import (  # noqa: E402
    BACKENDS,
    RealBackendUnavailable,
    run_benchmark,
)

_CSV_FIELDS = [
    "stage", "dataset", "task_type", "backend", "model_name", "num_examples",
    "metric_name", "metric_value", "accuracy", "macro_f1", "rouge_l",
    "numeric_exact_match", "exact_match", "latency_s", "latency_per_example_s",
    "audit_passed", "tee_used_on_gpu", "worker_has_mask_secrets",
    "dry_run", "paper_ready",
]


def _render_md(r: dict) -> str:
    L = ["# E9 — Task-utility benchmark", "",
         "_dry_run=%s paper_ready=%s backend=%s_"
         % (r["dry_run"], r["paper_ready"], r["backend"]), "",
         "## Summary", "",
         "| field | value |", "| --- | --- |"]
    for k in ("dataset", "task_type", "backend", "model_name", "num_examples",
              "metric_name", "metric_value", "accuracy", "macro_f1", "rouge_l",
              "numeric_exact_match", "exact_match",
              "token_match_rate_to_plain_reference", "latency_s",
              "latency_per_example_s", "audit_passed", "tee_used_on_gpu",
              "worker_has_mask_secrets", "gpu_visible_plaintext_fields",
              "leaked_secret_fields", "dry_run", "paper_ready"):
        L.append("| %s | %s |" % (k, r.get(k)))
    L += ["", "_Stub/dry-run reports are never paper_ready. No downloads; "
          "local data only._", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--task-type", default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--model-name", default="stub")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--folded-lora-package-path", default=None)
    ap.add_argument("--attestation-evidence", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--backend", default="plaintext_local", choices=BACKENDS)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--audit", action="store_true", default=True)
    ap.add_argument("--no-audit", dest="audit", action="store_false")
    ap.add_argument("--require-real", action="store_true", default=False,
                    help="fail (exit 3) instead of falling back to the stub if "
                         "the real backend cannot be constructed")
    ap.add_argument("--output-json", default="outputs/e9_task_utility.json")
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    ds = Path(args.dataset_jsonl)
    if not ds.is_file():
        print("ERROR: dataset JSONL not found: %s" % ds, file=sys.stderr)
        return 2

    try:
        report = run_benchmark(
            ds, backend=args.backend, task_type=args.task_type,
            max_examples=args.max_examples if args.max_examples > 0 else None,
            model_name=args.model_name, model_path=args.model_path,
            gpu_worker_url=args.gpu_worker_url,
            embedding_path=args.embedding_path,
            folded_lora_package_path=args.folded_lora_package_path,
            attestation_evidence=args.attestation_evidence,
            expected_mr_td=args.expected_mr_td, seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens, dtype=args.dtype,
            device=args.device, audit=args.audit,
            require_real=args.require_real)
    except RealBackendUnavailable as exc:
        print("ERROR: --require-real set but real backend unavailable: %s"
              % exc, file=sys.stderr)
        return 3

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_render_md(report), encoding="utf-8")
    if args.output_csv:
        p = Path(args.output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            w.writeheader()
            w.writerow({k: report.get(k) for k in _CSV_FIELDS})

    print("=== E9 task-utility benchmark ===")
    print("dataset=%s task_type=%s backend=%s n=%s"
          % (report["dataset"], report["task_type"], report["backend"],
             report["num_examples"]))
    print("metric %s=%s" % (report["metric_name"], report["metric_value"]))
    print("dry_run=%s paper_ready=%s audit_passed=%s"
          % (report["dry_run"], report["paper_ready"], report["audit_passed"]))
    print("\nE9 REPORT WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
