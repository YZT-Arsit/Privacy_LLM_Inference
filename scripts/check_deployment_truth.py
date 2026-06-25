"""Deployment truth checker CLI.

Parses one or more remote/folded/attested decode reports and prints, per report,
what was actually demonstrated plus the allowed / forbidden paper claims. Pure
parsing -- no torch / CUDA / H800 / TDX.

Example::

    python scripts/check_deployment_truth.py \\
        --result-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \\
        --result-json outputs/qwen7b_folded_remote_decode_probe.json \\
        --output-json outputs/deployment_truth.json \\
        --output-md  outputs/deployment_truth.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.deployment_truth import (  # noqa: E402
    deployment_truth_report,
    load_json,
)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return out


def render_md(combined: dict) -> str:
    L = ["# Deployment truth", "",
         "Per-report inference of what was actually demonstrated and the "
         "claims it does / does not support.", ""]
    rows = []
    for res in combined["results"]:
        t = res["truth"]
        rows.append([res.get("file"), t.get("gpu_real"), t.get("tee_real"),
                     t.get("attestation_verified"), t.get("lora_enabled"),
                     len(res.get("allowed_claims", [])),
                     len(res.get("forbidden_claims", []))])
    L += _md_table(
        ["file", "gpu_real", "tee_real", "attestation_verified", "lora_enabled",
         "allowed_claims", "forbidden_claims"], rows)
    for res in combined["results"]:
        L += ["", "## %s" % res.get("file"), "",
              "- source_stage: %s  dry_run: %s"
              % (_fmt(res.get("source_stage")), _fmt(res.get("dry_run"))),
              "- allowed_claims: %s"
              % (", ".join(res.get("allowed_claims", [])) or "(none)"),
              "- forbidden_claims: %s"
              % (", ".join(res.get("forbidden_claims", [])) or "(none)"),
              "- warnings: %s"
              % (", ".join(res.get("warnings", [])) or "(none)")]
    L += [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-json", action="append", required=True,
                    help="path to a decode report JSON (repeatable)")
    ap.add_argument("--output-json", default="outputs/deployment_truth.json")
    ap.add_argument("--output-md", default="outputs/deployment_truth.md")
    args = ap.parse_args()

    results = []
    for path in args.result_json:
        rep = load_json(path)
        res = deployment_truth_report(rep if isinstance(rep, dict) else {})
        res = dict(res)
        res["file"] = path
        res["loaded"] = rep is not None
        results.append(res)

    combined = {"stage": "deployment_truth", "results": results}

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(combined, indent=2, default=str),
                     encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_md(combined), encoding="utf-8")

    print("=== deployment truth ===")
    for res in results:
        t = res["truth"]
        print("%s: gpu_real=%s tee_real=%s attestation_verified=%s "
              "lora_enabled=%s allowed=%d forbidden=%d"
              % (res["file"], t.get("gpu_real"), t.get("tee_real"),
                 t.get("attestation_verified"), t.get("lora_enabled"),
                 len(res.get("allowed_claims", [])),
                 len(res.get("forbidden_claims", []))))
    print("\nDEPLOYMENT TRUTH WRITTEN (%d report(s))" % len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
