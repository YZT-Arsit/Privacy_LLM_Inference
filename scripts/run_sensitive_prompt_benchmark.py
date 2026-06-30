"""SensitivePrompt-1024 benchmark runner (privacy stress) -- delegates to the
unified AAAI runner (resume/retry/status/heartbeat/staged-schedule/gate reused),
forwarding the security-trace flags so a GPU-visible transcript can be scanned for
leaks afterwards by ``scripts/evaluate_sensitive_prompt_security.py``.

Example::

    python scripts/run_sensitive_prompt_benchmark.py \\
      --dataset-jsonl <SENSITIVE_JSONL> --backend folded_remote \\
      --model-path <MODEL> --gpu-worker-url <URL> --embedding-path <EMB> \\
      --nonlinear-backend A_rightmul --seq-len 1024 --max-new-tokens 512 \\
      --require-real --tdx-boundary-client --attestation-evidence-json <EV> \\
      --paper-facing-aaai --record-transcript \\
      --transcript-jsonl outputs/aaai/qwen/folded_remote/sensitive/transcript.jsonl \\
      --output-response-jsonl outputs/aaai/qwen/folded_remote/sensitive/responses.jsonl \\
      --output-report-json    outputs/aaai/qwen/folded_remote/sensitive/report.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_INNER = REPO_ROOT / "scripts" / "run_aaai_generation_benchmark.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # security-trace flags consumed here; everything else forwarded
    ap.add_argument("--record-transcript", action="store_true", default=False)
    ap.add_argument("--transcript-jsonl", default=None)
    ap.add_argument("--security-scan-spans", action="store_true", default=False)
    args, passthrough = ap.parse_known_args()

    cmd = [sys.executable, str(_INNER),
           "--dataset", "sensitive_prompt_1024"] + passthrough
    if args.record_transcript:
        # the inner runner audits every GPU request; ask it to also persist the
        # GPU-visible transcript for the offline leakage scan.
        cmd += ["--trace-worker-timings"]
        if args.transcript_jsonl:
            cmd += ["--trace-output-jsonl", args.transcript_jsonl]
    print("[sensitive] -> %s" % " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if args.security_scan_spans and rc == 0:
        print("[sensitive] run scripts/evaluate_sensitive_prompt_security.py next "
              "to scan %s for span leaks." % (args.transcript_jsonl or
                                              "(no transcript)"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
