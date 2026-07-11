"""Gate 2 §5 — start the trusted training service INSIDE the TDX guest.

Fail-closed: with --require-real-tdx it refuses to start unless /dev/tdx_guest is
present. Binds 127.0.0.1 only (tunnel-only exposure), SAFE codec, authenticated
session required, real_tdx attestation mode (real quote per /train/challenge).

Deploy the src/pllo tree unchanged and run in the guest conda env:

    python scripts/gate2_service.py --optimizer-mode gpu_masked_sgd --port 18091 \
        --require-real-tdx --run-id gate2_sgd

No secrets are logged. The ephemeral ECDH key lives only in the process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The safe wire codec excludes float64; the service runs float32 over the wire
# (fp64 exactness was proven in-process at Gate 1).
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

from pllo.experiments.real_tdx_attestation import (  # noqa: E402
    compute_service_runtime_hash,
    digest_config,
    service_artifact_hashes,
)
from pllo.experiments.real_tdx_training_service import (  # noqa: E402
    TrustedTrainingService,
    serve_http,
)


def build_config(args) -> dict:
    cfg = {
        "model_id": args.model_id,
        "lr": args.lr, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
        "weight_decay": args.weight_decay, "dtype": "float32",
        "optimizer_mode": args.optimizer_mode,
        "optimizer": args.optimizer,
        "gradient_convention": args.gradient_convention,
        "logits_mask_family": args.logits_mask_family,
        "session_timeout_s": args.session_timeout_s,
    }
    cfg["config_digest"] = digest_config({k: v for k, v in cfg.items()})
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizer-mode", required=True,
                    choices=["gpu_masked_sgd", "gpu_masked_momentum_sgd", "trusted_adamw"])
    ap.add_argument("--optimizer", default=None,
                    help="sgd|momentum_sgd for masked-SGD modes; adamw for trusted_adamw")
    ap.add_argument("--gradient-convention", default="nout_dual",
                    choices=["nout_dual", "independent_mout"])
    ap.add_argument("--logits-mask-family", default="dense",
                    choices=["dense", "vocab_permutation"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-id", default="synthetic-gate2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--session-timeout-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--require-real-tdx", action="store_true", default=False)
    ap.add_argument("--quote-workdir", default="/tmp/pllo_gate2_quote")
    ap.add_argument("--print-identity", action="store_true", default=False,
                    help="print runtime_hash + config_digest + artifact hashes and exit")
    args = ap.parse_args()

    if args.optimizer is None:
        args.optimizer = "adamw" if args.optimizer_mode == "trusted_adamw" else "sgd"

    if args.require_real_tdx and not Path("/dev/tdx_guest").exists():
        print("FAIL-CLOSED: --require-real-tdx but /dev/tdx_guest is absent; refusing "
              "to start (no CPU fallback).", file=sys.stderr)
        return 2

    cfg = build_config(args)
    runtime_hash = compute_service_runtime_hash(cfg["config_digest"])
    if args.print_identity:
        print(json.dumps({
            "run_id": args.run_id, "config_digest": cfg["config_digest"],
            "runtime_hash": runtime_hash, "optimizer_mode": args.optimizer_mode,
            "gradient_convention": args.gradient_convention, "model_id": args.model_id,
            "artifact_hashes": service_artifact_hashes()}, indent=2))
        return 0

    mode = "real_tdx" if args.require_real_tdx else "cpu_contract"
    svc = TrustedTrainingService(run_id=args.run_id, config=cfg, mode=mode,
                                 require_real_tdx=args.require_real_tdx, seed=args.seed,
                                 dtype=_DTYPES.get(cfg["dtype"], torch.float32))
    httpd = serve_http(svc, host=args.host, port=args.port,
                       require_authenticated_session=True,
                       quote_workdir=args.quote_workdir)
    print(json.dumps({"event": "listening", "host": args.host, "port": args.port,
                      "mode": mode, "run_id": args.run_id,
                      "optimizer_mode": args.optimizer_mode,
                      "gradient_convention": args.gradient_convention,
                      "runtime_hash": runtime_hash,
                      "config_digest": cfg["config_digest"],
                      "guest_ephemeral_public_hex": httpd.pllo_session.guest_public_hex}),
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
