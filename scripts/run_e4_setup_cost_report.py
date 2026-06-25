"""E4 runner: setup / provisioning cost + amortization report.

Consolidates the one-time setup facts for the folded-package deployment (folded
package generation/size/load + boundary embedding artifact size/hash) from the
package manifest on disk and/or prior build/verify/inspection/load-probe JSON
outputs, then estimates transfer time per bandwidth and amortized per-session
setup cost. Pure parsing + arithmetic -- no torch / CUDA / checkpoint required.

Example::

    python scripts/run_e4_setup_cost_report.py \\
        --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --embedding-artifact-path /root/.../qwen7b_boundary_artifact_cuda \\
        --build-json outputs/folded_build_full.json \\
        --verify-json outputs/folded_verify_full.json \\
        --inspection-json outputs/folded_full_inspection.json \\
        --load-probe-json outputs/qwen7b_folded_full_load_probe.json \\
        --bandwidth-mbps-list 100,500,1000,5000 \\
        --amortize-sessions-list 1,10,100,1000 \\
        --output-json outputs/e4_setup_cost_report.json \\
        --output-md  outputs/e4_setup_cost_report.md \\
        --output-csv outputs/e4_setup_cost_report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e4_setup_cost import (  # noqa: E402
    build_e4_report,
    gather_facts,
    load_json,
    render_e4_csv,
    render_e4_md,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)


def _csv_floats(s):
    return [float(p) for p in str(s).replace(" ", "").split(",") if p != ""]


def _csv_ints(s):
    return [int(p) for p in str(s).replace(" ", "").split(",") if p != ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folded-package-path", default=None)
    ap.add_argument("--embedding-artifact-path", default=None)
    ap.add_argument("--build-json", default=None)
    ap.add_argument("--verify-json", default=None)
    ap.add_argument("--inspection-json", default=None)
    ap.add_argument("--load-probe-json", default=None)
    ap.add_argument("--bandwidth-mbps-list", default="100,500,1000,5000")
    ap.add_argument("--amortize-sessions-list", default="1,10,100,1000")
    ap.add_argument("--output-json", default="outputs/e4_setup_cost_report.json")
    ap.add_argument("--output-md", default="outputs/e4_setup_cost_report.md")
    ap.add_argument("--output-csv", default="outputs/e4_setup_cost_report.csv")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    facts = gather_facts(
        folded_package_path=args.folded_package_path,
        embedding_artifact_path=args.embedding_artifact_path,
        build_json=load_json(args.build_json),
        verify_json=load_json(args.verify_json),
        inspection_json=load_json(args.inspection_json),
        load_probe_json=load_json(args.load_probe_json))
    report = build_e4_report(
        facts, bandwidth_mbps_list=_csv_floats(args.bandwidth_mbps_list),
        sessions_list=_csv_ints(args.amortize_sessions_list))
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e4_md(report), encoding="utf-8")
    if args.output_csv:
        p = Path(args.output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e4_csv(report), encoding="utf-8")

    print("=== E4 setup cost + amortization ===")
    print("folded_package_size_gb=%s (store_dtype=%s, if_bf16=%s)"
          % (report["folded_package_size_gb"],
             report["folded_package_store_dtype"],
             report["folded_package_size_if_bf16_gb"]))
    print("num_layers=%s num_shards=%s manifest_hash=%s"
          % (report["num_layers"], report["num_shards"],
             report["manifest_hash"]))
    print("generation_time_s=%s package_load_time_s=%s one_time_setup_s=%s"
          % (report["generation_time_s"], report["package_load_time_s"],
             report["one_time_setup_s"]))
    print("boundary_embedding_artifact_size_gb=%s (trusted_only=%s, "
          "contains_mask_secrets=%s)"
          % (report["boundary_embedding_artifact_size_gb"],
             report["boundary_artifact_trusted_only"],
             report["boundary_artifact_contains_mask_secrets"]))
    print("package_verify_passed=%s" % report["package_verify_passed"])
    print("\nE4 REPORT WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
