#!/usr/bin/env python
"""Start the untrusted GPU HTTP worker for PLLO remote execution.

The server exposes:
  GET  /health
  POST /init
  POST /prefill
  POST /decode

It hosts pllo.protocol.remote.GpuWorkerServer and runs the selected GPU backend.
For AAAI/H800 deployment, use backend=qwen7b_folded_package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pllo.protocol.remote import run_gpu_worker_server  # noqa: E402


def _bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("PLLO_GPU_WORKER_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PLLO_GPU_WORKER_PORT", "18082")))
    ap.add_argument("--backend", "--gpu-backend", dest="backend", default="qwen7b_folded_package")

    ap.add_argument("--folded-package-path", default=os.environ.get("PLLO_FOLDED_PACKAGE_DIR"))
    ap.add_argument("--folded-lora-package-path", default=None)
    ap.add_argument("--device", default=os.environ.get("PLLO_DEVICE", "cuda"))
    ap.add_argument("--dtype", default=os.environ.get("PLLO_DTYPE", "bfloat16"))
    ap.add_argument("--nonlinear-backend", default=os.environ.get("PLLO_NONLINEAR_BACKEND", "A_rightmul"))
    ap.add_argument("--nonlinear-lift-k", type=int, default=2)
    ap.add_argument("--nonlinear-seed", type=int, default=2035)
    ap.add_argument("--verify-on-init", type=_bool, default=True)
    ap.add_argument("--resident-folded-weights", action="store_true")
    ap.add_argument("--native-logits-wire", action="store_true",
                    help="return masked logits in native bf16 (half the wire "
                    "bytes); bit-identical after the boundary's bf16->fp32 upcast")
    ap.add_argument("--fold-dtype-override", default=None,
                    help="Force the folded compute dtype (e.g. float32) instead "
                    "of the boundary meta fold_dtype; use when package shards are "
                    "stored above the meta precision. Purely numerical -- no "
                    "design/security change.")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--backend-json", default=None,
                    help="Optional JSON object merged into backend kwargs.")

    args = ap.parse_args()

    kwargs = {}
    if args.backend_json:
        kwargs.update(json.loads(args.backend_json))

    if args.backend == "qwen7b_folded_package":
        if not args.folded_package_path:
            ap.error("--folded-package-path is required for qwen7b_folded_package")
        kwargs.update({
            "folded_package_path": args.folded_package_path,
            "folded_lora_package_path": args.folded_lora_package_path,
            "device": args.device,
            "dtype": args.dtype,
            "verify_on_init": args.verify_on_init,
            "nonlinear_backend": args.nonlinear_backend,
            "nonlinear_lift_k": args.nonlinear_lift_k,
            "nonlinear_seed": args.nonlinear_seed,
            "resident_folded_weights": bool(args.resident_folded_weights),
            "native_logits_wire": bool(args.native_logits_wire),
            "fold_dtype_override": args.fold_dtype_override,
        })

    print("[run_gpu_worker_server] backend_kwargs=" + json.dumps(
        {k: str(v) for k, v in kwargs.items()}, ensure_ascii=False), flush=True)

    run_gpu_worker_server(
        host=args.host,
        port=args.port,
        backend_name=args.backend,
        backend_kwargs=kwargs,
        audit=not args.no_audit,
    )


if __name__ == "__main__":
    main()
