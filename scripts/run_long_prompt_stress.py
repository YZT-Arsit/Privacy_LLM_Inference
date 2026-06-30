"""Long-prompt stress runner (LongBench-1024-lite / SensitivePrompt-1024).

NOT an official LongBench score: inputs are capped to 1024 tokens and this run is
used for latency / scaling / long-prompt security stress only. Delegates to the
unified AAAI runner (resume/retry/staged/gate reused), then stamps
``official_longbench_score=false`` + the stress reason into the report.

Example::

    python scripts/run_long_prompt_stress.py --dataset longbench_1024_lite \\
      --dataset-jsonl <LB_LITE_JSONL> --backend folded_remote \\
      --model-path <MODEL> --gpu-worker-url <URL> --embedding-path <EMB> \\
      --nonlinear-backend A_rightmul --seq-len 1024 --max-new-tokens 512 \\
      --require-real --tdx-boundary-client --attestation-evidence-json <EV> \\
      --output-response-jsonl outputs/.../responses.jsonl \\
      --output-report-json    outputs/.../report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_INNER = REPO_ROOT / "scripts" / "run_aaai_generation_benchmark.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["longbench_1024_lite", "sensitive_prompt_1024"])
    ap.add_argument("--length-buckets", default="128,512,1024",
                    help="recorded in the report (stress bucketing)")
    args, passthrough = ap.parse_known_args()

    # locate the inner report path so we can stamp the stress disclaimer
    report_path = None
    for i, a in enumerate(passthrough):
        if a == "--output-report-json" and i + 1 < len(passthrough):
            report_path = passthrough[i + 1]

    cmd = [sys.executable, str(_INNER), "--dataset", args.dataset] + passthrough
    print("[long-stress] -> %s" % " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)

    if report_path and Path(report_path).exists():
        try:
            rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
            rep["official_longbench_score"] = False
            rep["long_prompt_stress"] = True
            rep["length_buckets"] = [int(b) for b in
                                     args.length_buckets.split(",") if b.strip()]
            rep["reason"] = ("seq_len fixed to 1024; used for "
                             "stress/scaling/security only")
            Path(report_path).write_text(json.dumps(rep, indent=2, default=str),
                                         encoding="utf-8")
        except Exception:                                        # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
