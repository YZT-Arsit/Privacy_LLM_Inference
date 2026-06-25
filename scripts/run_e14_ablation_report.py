"""E14 ablation report CLI.

Gathers already-produced experiment report JSONs by flag and emits the per-axis
correctness/security/performance DELTA report (boundary full-reference vs lite,
F32 vs BF16 folded storage, LoRA rank-mask on/off [safe-fixture only], attested
vs non-attested, max_new_tokens scaling).

Example::

    python scripts/run_e14_ablation_report.py \\
        --full-reference-decode-json outputs/decode_full_ref.json \\
        --lite-decode-json outputs/decode_lite.json \\
        --attested-decode-json outputs/decode_attested.json \\
        --nonattested-decode-json outputs/decode_nonattested.json \\
        --max-new-tokens-decode-json outputs/decode_mnt8.json \\
        --max-new-tokens-decode-json outputs/decode_mnt32.json \\
        --nonlinear-backend current \\
        --output-json outputs/e14_ablation.json \\
        --output-md outputs/e14_ablation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.ablation_report import (  # noqa: E402
    build_ablation_report,
    load_json,
    render_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-reference-decode-json", default=None)
    ap.add_argument("--lite-decode-json", default=None)
    ap.add_argument("--f32-build-json", default=None)
    ap.add_argument("--bf16-build-json", default=None)
    ap.add_argument("--lora-rankmask-on-json", default=None)
    ap.add_argument("--lora-rankmask-off-json", default=None)
    ap.add_argument("--attested-decode-json", default=None)
    ap.add_argument("--nonattested-decode-json", default=None)
    ap.add_argument("--max-new-tokens-decode-json", action="append", default=[])
    ap.add_argument("--safe-fixture-mode", action="store_true", default=False)
    ap.add_argument("--nonlinear-backend", default=None)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    inputs = {
        "full_reference_decode": load_json(args.full_reference_decode_json),
        "lite_decode": load_json(args.lite_decode_json),
        "f32_build": load_json(args.f32_build_json),
        "bf16_build": load_json(args.bf16_build_json),
        "lora_rankmask_on": load_json(args.lora_rankmask_on_json),
        "lora_rankmask_off": load_json(args.lora_rankmask_off_json),
        "attested_decode": load_json(args.attested_decode_json),
        "nonattested_decode": load_json(args.nonattested_decode_json),
        "max_new_tokens_decode": [
            r for r in (load_json(p) for p in args.max_new_tokens_decode_json)
            if r is not None],
        "safe_fixture_mode": args.safe_fixture_mode,
    }

    report = build_ablation_report(
        inputs, nonlinear_backend=args.nonlinear_backend)

    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        om = Path(args.output_md)
        om.parent.mkdir(parents=True, exist_ok=True)
        om.write_text(render_md(report), encoding="utf-8")

    print("=== E14 ablation report ===")
    print("axes_available=%s" % report["axes_available"])
    print("axes_unavailable=%s" % report["axes_unavailable"])
    print("paper_ready=%s dry_run=%s" % (report["paper_ready"],
                                         report["dry_run"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
