#!/usr/bin/env bash
# Gate 2 — start the trusted TRAINING service INSIDE the real TDX guest.
#
# Fail-closed: runs with require_real_tdx=1 so the service refuses to start unless
# /dev/tdx_guest is present and the runtime hash binds to a real quote. No CPU
# fallback. Run this ON the guest (39.96.4.252), not locally.
#
# Usage (on the guest):
#   PLLO_TRAIN_PORT=18091 bash scripts/start_tdx_trusted_service.sh <run_id> <config_digest>
set -euo pipefail

RUN_ID="${1:?run_id required}"
CONFIG_DIGEST="${2:?config_digest required}"
PORT="${PLLO_TRAIN_PORT:-18091}"
HOST="${PLLO_TRAIN_HOST:-0.0.0.0}"
PY="${PLLO_PY:-/root/miniconda3/envs/tdx310/bin/python}"
ROOT="${PLLO_ROOT:-/root/privacy_llm_obfuscation}"

if [ ! -e /dev/tdx_guest ]; then
  echo "FAIL-CLOSED: /dev/tdx_guest not present -- refusing to start trusted service." >&2
  exit 3
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
exec "$PY" - "$RUN_ID" "$CONFIG_DIGEST" "$HOST" "$PORT" <<'PYEOF'
import sys
from pllo.experiments.real_tdx_training_service import TrustedTrainingService, serve_http
run_id, config_digest, host, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# NOTE: a real deployment loads the run's lora_manifest/config from a trusted-side
# spec file. This entrypoint constructs the service in require_real_tdx=1 mode; the
# GPU client drives init_session over the tunnel with the matching config_digest.
cfg = {"config_digest": config_digest, "lr": 1e-3, "dtype": "bfloat16"}
svc = TrustedTrainingService(run_id=run_id, config=cfg, mode="real_tdx",
                             require_real_tdx=True)
httpd = serve_http(svc, host=host, port=port)
print(f"[tdx-trusted-service] run_id={run_id} listening {host}:{port} (require_real_tdx=1)")
httpd.serve_forever()
PYEOF
