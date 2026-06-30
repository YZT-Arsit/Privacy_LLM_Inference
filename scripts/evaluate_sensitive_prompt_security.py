"""Evaluate sensitive-prompt leakage on GPU-visible artifacts.

Scans the GPU-visible transcript, worker-health records, and the run report for
(a) the dataset's fabricated ``sensitive_spans`` and (b) forbidden field names
(input_ids / plaintext / raw mask / N / N_inv / raw pad / recovery / token ids).
Allowed GPU-visible artifacts: folded ``*_tilde``, ``xpad_tilde`` / ``cpad_tilde``,
masked-basis staged artifacts, public commitments. Outputs a security report; exit
non-zero if anything leaks.

Example::

    python scripts/evaluate_sensitive_prompt_security.py \\
      --dataset-jsonl <SENSITIVE_JSONL> \\
      --response-jsonl <RESP.jsonl> --transcript-jsonl <TRANSCRIPT.jsonl> \\
      --report-json <REPORT.json> --worker-health-jsonl <WH.jsonl> \\
      --output-json outputs/.../sensitive_security_report.json \\
      --output-md   outputs/.../sensitive_security_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.security.sensitive_scan import scan_sensitive_leakage  # noqa: E402


def _jsonl(path):
    rows = []
    p = Path(path) if path else None
    if not p or not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return rows


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    except Exception:                                            # noqa: BLE001
        return None


def _md(rep):
    return "\n".join([
        "# Sensitive-prompt security scan", "",
        "_leakage_pass=%s spans=%d transcript_entries=%d_"
        % (rep["leakage_pass"], rep["num_sensitive_spans"],
           rep["num_transcript_entries"]), "",
        "- leaked_fields: %s" % rep["leaked_fields"],
        "- gpu_visible_span_leaks: %d" % rep["gpu_visible_span_leaks"],
        "- raw_input_protected: %s" % rep["raw_input_protected"], ""]) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--response-jsonl", default=None)
    ap.add_argument("--transcript-jsonl", default=None)
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--worker-health-jsonl", default=None)
    ap.add_argument("--error-log", default=None)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    dataset_rows = _jsonl(args.dataset_jsonl)
    error_logs = None
    if args.error_log and Path(args.error_log).exists():
        error_logs = Path(args.error_log).read_text(
            encoding="utf-8", errors="replace").splitlines()

    rep = scan_sensitive_leakage(
        dataset_rows=dataset_rows,
        transcript_entries=_jsonl(args.transcript_jsonl),
        worker_health=_jsonl(args.worker_health_jsonl),
        report=_json(args.report_json), error_logs=error_logs)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(_md(rep), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in (
        "leakage_pass", "num_sensitive_spans", "leaked_fields",
        "gpu_visible_span_leaks")}, indent=2))
    return 0 if rep["leakage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
