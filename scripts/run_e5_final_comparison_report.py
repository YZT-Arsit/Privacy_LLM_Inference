"""E5 runner: paper-ready comparison + security matrix.

Consolidates the prior experiment JSON outputs (E1/E2, H800 local package probes,
E3 remote scaling, TDX-lite + TDX-attested remote decode, E4 setup cost) into a
correctness table, deployment table, security matrix, cost table, and limitations
list, rendered as JSON + Markdown + LaTeX. Also writes a paper-ready evaluation
Markdown. Pure parsing -- no torch / CUDA / checkpoint required; missing inputs
are reported as not-provided (never assumed).

Example::

    python scripts/run_e5_final_comparison_report.py \\
        --e1-json outputs/e1_qwen_no_lora.json \\
        --e2-json outputs/e2_qwen_scaling.json \\
        --local-prefill-json outputs/qwen7b_folded_full_prefill_28layer_probe.json \\
        --local-logits-json outputs/qwen7b_folded_full_onestep_logits_probe.json \\
        --local-decode-json outputs/qwen7b_folded_full_decode_probe.json \\
        --remote-scaling-json outputs/e3_remote_decode_scaling.json \\
        --tdx-lite-json outputs/tdx_qwen7b_folded_remote_lite_decode_probe_cuda_artifact.json \\
        --tdx-attested-json outputs/tdx_attested_qwen7b_folded_remote_decode_probe.json \\
        --setup-cost-json outputs/e4_setup_cost_report.json \\
        --output-json outputs/e5_final_comparison_report.json \\
        --output-md  outputs/e5_final_comparison_report.md \\
        --output-tex outputs/e5_final_comparison_table.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e5_final_comparison import (  # noqa: E402
    build_e5_report,
    load_json,
    render_e5_md,
    render_e5_tex,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-json", default=None)
    ap.add_argument("--e2-json", default=None)
    ap.add_argument("--local-prefill-json", default=None)
    ap.add_argument("--local-logits-json", default=None)
    ap.add_argument("--local-decode-json", default=None)
    ap.add_argument("--remote-scaling-json", default=None)
    ap.add_argument("--tdx-lite-json", default=None)
    ap.add_argument("--tdx-attested-json", default=None)
    ap.add_argument("--setup-cost-json", default=None)
    ap.add_argument("--output-json",
                    default="outputs/e5_final_comparison_report.json")
    ap.add_argument("--output-md",
                    default="outputs/e5_final_comparison_report.md")
    ap.add_argument("--output-tex",
                    default="outputs/e5_final_comparison_table.tex")
    ap.add_argument("--paper-ready-md",
                    default="outputs/paper_ready_final_evaluation.md",
                    help="paper-ready evaluation Markdown (E3-E5 narrative + E5 "
                         "tables); set empty to skip")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    inputs = {
        "e1": load_json(args.e1_json), "e2": load_json(args.e2_json),
        "local_prefill": load_json(args.local_prefill_json),
        "local_logits": load_json(args.local_logits_json),
        "local_decode": load_json(args.local_decode_json),
        "remote_scaling": load_json(args.remote_scaling_json),
        "tdx_lite": load_json(args.tdx_lite_json),
        "tdx_attested": load_json(args.tdx_attested_json),
        "setup_cost": load_json(args.setup_cost_json),
    }
    report = build_e5_report(inputs)
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))
    md = render_e5_md(report)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    if args.output_tex:
        p = Path(args.output_tex)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e5_tex(report), encoding="utf-8")
    if args.paper_ready_md:
        p = Path(args.paper_ready_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_paper_ready_md(report), encoding="utf-8")

    prov = report["inputs_provided"]
    print("=== E5 final comparison ===")
    print("inputs_provided: " + ", ".join(
        "%s=%s" % (k, v) for k, v in prov.items()))
    print("correctness rows: %d  deployment rows: %d  security rows: %d"
          % (len(report["correctness"]), len(report["deployment"]),
             len(report["security_matrix"]["matrix"])))
    print("audit_cross_check_ok=%s"
          % report["security_matrix"]["audit_cross_check_ok"])
    print("\nE5 REPORT WRITTEN")
    return 0


def _paper_ready_md(report: dict) -> str:
    head = [
        "# Paper-ready final evaluation (E3–E5)", "",
        "This file consolidates the evaluation of the privacy-preserving folded-",
        "package inference system for Qwen2.5-7B. It is generated from the prior",
        "experiment JSON outputs; entries from missing inputs are marked",
        "not-provided rather than assumed.", "",
        "## Overview", "",
        "- **E3** — scaling/performance sweep for remote package-backed decode.",
        "- **E4** — setup/provisioning cost + amortization.",
        "- **E5** — correctness/deployment/security/cost consolidation (below).",
        "",
    ]
    return "\n".join(head) + "\n" + render_e5_md(report) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
