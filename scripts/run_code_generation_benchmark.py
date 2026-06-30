"""Code-generation benchmark runner (HumanEval / MBPP) -- delegates to the unified
AAAI runner so resume / retry / status / heartbeat / staged-schedule / paper-facing
gate are reused verbatim. The response JSONL's ``response`` field is the model
completion; scoring (pass@1) is a separate offline step
(``scripts/evaluate_code_generation.py``), never run inside the TEE/GPU path.

Example::

    python scripts/run_code_generation_benchmark.py --dataset humaneval \\
      --dataset-jsonl <HE_JSONL> --backend folded_remote \\
      --model-path <MODEL> --gpu-worker-url <URL> --embedding-path <EMB> \\
      --nonlinear-backend A_rightmul --seq-len 1024 --max-new-tokens 512 \\
      --require-real --tdx-boundary-client --attestation-evidence-json <EV> \\
      --paper-facing-aaai --run-id he_ours \\
      --output-response-jsonl outputs/aaai/qwen/folded_remote/humaneval/responses.jsonl \\
      --output-report-json    outputs/aaai/qwen/folded_remote/humaneval/report.json
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
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    args, passthrough = ap.parse_known_args()
    cmd = [sys.executable, str(_INNER), "--dataset", args.dataset] + passthrough
    print("[code-gen] -> %s" % " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
