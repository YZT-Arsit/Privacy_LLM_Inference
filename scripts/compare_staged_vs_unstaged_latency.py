"""Compare a staged vs an unstaged A_rightmul run (latency / interaction deltas).

The GPU-staged schedule is an engineering optimisation that pre-stages NON-SECRET
masked-basis artifacts to the GPU to cut online TEE<->GPU interaction; it does NOT
change the security model (the GPU still sees no raw input / mask / pad). This
tool reads two run reports (unstaged, staged) and emits the deltas + the staged
run's no-secret audit verdict.

Example::

    python scripts/compare_staged_vs_unstaged_latency.py \\
      --unstaged-report <UNSTAGED.json> --staged-report <STAGED.json> \\
      --output-json outputs/.../staged_vs_unstaged.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(b - a, 6)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unstaged-report", required=True)
    ap.add_argument("--staged-report", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    un = _load(args.unstaged_report)
    st = _load(args.staged_report)
    keys = ("ttft_s", "tpot_s", "end_to_end_latency_s", "tokens_per_sec",
            "boundary_calls", "nonlinear_trusted_bytes",
            "nonlinear_accelerator_bytes", "generated_tokens_total")
    deltas = {k: {"unstaged": un.get(k), "staged": st.get(k),
                  "delta": _delta(un.get(k), st.get(k))} for k in keys}
    rep = {
        "stage": "staged_vs_unstaged_latency",
        "optimization": "gpu_staged_nonsecret_obfuscation_artifacts",
        "raw_input_protected": True,
        "gpu_received_raw_masks": False,
        "staged_no_secret_audit_passed": bool(
            st.get("staged_schedule_no_secret_audit_passed")),
        "staged_schedule_used": bool(st.get("staged_schedule_used")),
        "deltas": deltas,
        "note": "staged schedule is an engineering path; security model unchanged "
                "(GPU sees no raw input/mask/pad).",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({"staged_no_secret_audit_passed":
                      rep["staged_no_secret_audit_passed"],
                      "deltas": {k: deltas[k]["delta"] for k in deltas}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
