"""E8 runner: private-LoRA final report (correctness / security / cost / training).

Consolidates the LoRA experiment JSON outputs into four tables rendered as JSON +
Markdown. Pure parsing -- no torch / CUDA / checkpoint. Missing inputs are
reported as not-provided (never assumed).

Example::

    python scripts/run_e8_lora_final_report.py \\
        --local-json outputs/qwen7b_lora_folded_local_probe.json \\
        --remote-json outputs/qwen7b_lora_folded_remote_decode_probe.json \\
        --attested-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json \\
        --lora-build-json outputs/qwen7b_lora_folded_build.json \\
        --lora-verify-json outputs/qwen7b_lora_folded_verify.json \\
        --base-decode-json outputs/qwen7b_folded_full_decode_probe.json \\
        --training-json outputs/lora_private_training_probe.json \\
        --output-json outputs/e8_lora_final_report.json \\
        --output-md  outputs/e8_lora_final_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e8_lora_report import (  # noqa: E402
    build_e8_report,
    load_json,
    render_e8_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-json", default=None)
    ap.add_argument("--remote-json", default=None)
    ap.add_argument("--attested-json", default=None)
    ap.add_argument("--lora-build-json", default=None)
    ap.add_argument("--lora-verify-json", default=None)
    ap.add_argument("--base-decode-json", default=None)
    ap.add_argument("--training-json", default=None)
    ap.add_argument("--output-json", default="outputs/e8_lora_final_report.json")
    ap.add_argument("--output-md", default="outputs/e8_lora_final_report.md")
    args = ap.parse_args()

    inputs = {
        "local": load_json(args.local_json),
        "remote": load_json(args.remote_json),
        "attested": load_json(args.attested_json),
        "lora_build": load_json(args.lora_build_json),
        "lora_verify": load_json(args.lora_verify_json),
        "base_decode": load_json(args.base_decode_json),
        "training": load_json(args.training_json),
    }
    report = build_e8_report(inputs)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e8_md(report), encoding="utf-8")

    prov = report["inputs_provided"]
    print("=== E8 private LoRA final report ===")
    print("inputs_provided: " + ", ".join("%s=%s" % (k, v)
                                          for k, v in prov.items()))
    print("security audit_cross_check_ok=%s"
          % report["security_matrix"]["audit_cross_check_ok"])
    t = report["training"]
    if t.get("provided"):
        print("training: loss_before=%s loss_after=%s loss_decreased=%s"
              % (t["loss_before"], t["loss_after"], t["loss_decreased"]))
    print("\nE8 REPORT WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
