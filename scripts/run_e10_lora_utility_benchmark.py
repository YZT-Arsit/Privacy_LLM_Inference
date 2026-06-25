"""E10 runner: private folded-LoRA utility-preservation report.

Consolidates E9 task-utility benchmark reports (``metric_value``) for base /
plaintext-LoRA / folded-LoRA (/ optional TDX-attested folded-LoRA) into a single
utility-preservation + security report. Pure parsing -- no torch / CUDA / model.

Example::

    python scripts/run_e10_lora_utility_benchmark.py \\
        --dataset-name sst2 --task-type classification --metric-name accuracy \\
        --base-json            outputs/e9_sst2_base.json \\
        --plaintext-lora-json  outputs/e9_sst2_plaintext_lora.json \\
        --folded-lora-json     outputs/e9_sst2_folded_lora_remote.json \\
        --tdx-attested-folded-lora-json outputs/e9_sst2_tdx_attested_lora.json \\
        --no-lora-decode-json  outputs/qwen7b_folded_full_decode_probe.json \\
        --lora-decode-json     outputs/qwen7b_lora_folded_remote_decode_probe.json \\
        --lora-verify-json     outputs/qwen7b_lora_folded_verify.json \\
        --output-json outputs/e10_lora_utility.json \\
        --output-md   outputs/e10_lora_utility.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.lora_utility import (  # noqa: E402
    build_lora_utility_report,
    render_lora_utility_md,
)


def _load(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def _compare_decodes(no_lora, lora):
    """Reuse validate_lora_effect.compare_decodes (token + metric effect)."""
    if no_lora is None and lora is None:
        return None
    spec = importlib.util.spec_from_file_location(
        "vle", REPO_ROOT / "scripts" / "validate_lora_effect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compare_decodes(no_lora, lora)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--task-type", default=None)
    ap.add_argument("--metric-name", default=None)
    ap.add_argument("--base-json", default=None)
    ap.add_argument("--plaintext-lora-json", default=None)
    ap.add_argument("--folded-lora-json", default=None)
    ap.add_argument("--tdx-attested-folded-lora-json", default=None)
    ap.add_argument("--no-lora-decode-json", default=None)
    ap.add_argument("--lora-decode-json", default=None)
    ap.add_argument("--lora-verify-json", default=None)
    ap.add_argument("--preserve-threshold", type=float, default=0.9)
    ap.add_argument("--output-json", default="outputs/e10_lora_utility.json")
    ap.add_argument("--output-md", default="outputs/e10_lora_utility.md")
    args = ap.parse_args()

    no_lora = _load(args.no_lora_decode_json)
    lora = _load(args.lora_decode_json)
    effect = _compare_decodes(no_lora, lora)

    report = build_lora_utility_report({
        "dataset_name": args.dataset_name, "task_type": args.task_type,
        "metric_name": args.metric_name,
        "base": _load(args.base_json),
        "plaintext_lora": _load(args.plaintext_lora_json),
        "folded_lora": _load(args.folded_lora_json),
        "tdx_attested_folded_lora": _load(args.tdx_attested_folded_lora_json),
        "effect": effect, "lora_verify": _load(args.lora_verify_json),
        "preserve_threshold": args.preserve_threshold,
    })

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_lora_utility_md(report), encoding="utf-8")

    print("=== E10 LoRA utility preservation ===")
    print("dataset=%s task=%s metric=%s (dry_run=%s paper_ready=%s)"
          % (report["dataset_name"], report["task_type"], report["metric_name"],
             report["dry_run"], report["paper_ready"]))
    print("base=%s plaintext_lora=%s folded_lora=%s attested=%s"
          % (report["base_metric"], report["plaintext_lora_metric"],
             report["folded_lora_metric"],
             report["tdx_attested_folded_lora_metric"]))
    print("gain_plaintext=%s gain_folded=%s preserved_ratio=%s preserves_gain=%s"
          % (report["lora_gain_plaintext"], report["lora_gain_folded"],
             report["lora_gain_preserved_ratio"],
             report["folded_lora_preserves_gain"]))
    print("security_ok=%s utility_preserved=%s"
          % (report["security_ok"], report["utility_preserved"]))
    print("\nE10 REPORT WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
