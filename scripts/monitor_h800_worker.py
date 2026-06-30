"""Health monitor for the untrusted H800 GPU worker.

Polls ``<gpu-worker-url>/health`` every ``--interval-sec`` via the resilient
client and appends a JSONL health record (alive, nonlinear backend + execution
evidence, package loaded, compatible-mask flag, GPU memory if reported) to
``outputs/status/worker_health_<run_id>.jsonl``. Read-only GET; never sends a
prompt or any secret. ``--once`` prints a single snapshot. stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.protocol.resilient_remote import ResilientRemoteGpuWorker  # noqa: E402


def _snapshot(url, timeout):
    client = ResilientRemoteGpuWorker(url, per_request_timeout=timeout,
                                      max_retries=2, backoff_base_sec=0.3)
    try:
        h = client.health()
        alive = True
        err = None
    except Exception as exc:                                     # noqa: BLE001
        h, alive, err = {}, False, "%s: %s" % (type(exc).__name__, exc)
    finally:
        client.close()
    ev = (h or {}).get("nonlinear_execution_evidence") or {}
    return {
        "time": None,                # filled by caller (clock indirection-free)
        "worker_url": url,
        "alive": alive,
        "error": err,
        "nonlinear_backend": (h or {}).get("nonlinear_backend"),
        "nonlinear_op_backend": (h or {}).get("nonlinear_op_backend"),
        "compatible_masks_verified": (h or {}).get("compatible_masks_verified"),
        "package_loaded": (h or {}).get("folded_package_loaded"),
        "folded_package_path": (h or {}).get("folded_package_path"),
        "tee_used_on_gpu": (h or {}).get("tee_used_on_gpu"),
        "nonlinear_trusted_calls": ev.get("nonlinear_trusted_calls"),
        "nonlinear_execution_status": ev.get("nonlinear_execution_status"),
        "peak_gpu_memory_mb": (h or {}).get("peak_gpu_memory_mb"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-worker-url", required=True)
    ap.add_argument("--run-id", default="worker")
    ap.add_argument("--output-dir", default="outputs/status")
    ap.add_argument("--interval-sec", type=float, default=10.0)
    ap.add_argument("--timeout-sec", type=float, default=10.0)
    ap.add_argument("--once", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("worker_health_%s.jsonl" % args.run_id)
    while True:
        snap = _snapshot(args.gpu_worker_url, args.timeout_sec)
        snap["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(snap) + "\n")
        print("[worker] alive=%s nonlinear=%s pkg=%s trusted_calls=%s gpu_mb=%s"
              % (snap["alive"], snap["nonlinear_backend"], snap["package_loaded"],
                 snap["nonlinear_trusted_calls"], snap["peak_gpu_memory_mb"]),
              flush=True)
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
