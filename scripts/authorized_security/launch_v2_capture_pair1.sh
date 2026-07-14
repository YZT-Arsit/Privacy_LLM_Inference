#!/bin/bash
set -euo pipefail

ROOT=/root/pllo_pb
OUT="$ROOT/results/aaai_private_base/authorized_security_eval/corpora/e2e_test_v2_s1234.jsonl"
LOG="$ROOT/results/aaai_private_base/authorized_security_eval/monitor/v2_capture_s1234.log"
META="$ROOT/results/aaai_private_base/authorized_security_eval/monitor/v2_capture_s1234"

cd "$ROOT"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
if [[ -e "$OUT" || -e "$OUT.profile.json" ]]; then
  echo "refusing duplicate V2 capture: output already exists" >&2
  exit 20
fi
if pgrep -af 'a10_batch_runner.py|protected_generate.py' >/dev/null; then
  echo "refusing V2 capture while a protected GPU job is active" >&2
  exit 21
fi

sha256sum \
  scripts/authorized_security/capture_v2_prefill.py \
  scripts/authorized_security/view_contracts.py \
  results/aaai_private_base/generative_lora/data/e2e_test_gen.json \
  results/aaai_private_base/generative_lora/protected_runs/e2e_L12_s1234_full.adapter.pt \
  > "$META.source_and_input_hashes.sha256"
date --iso-8601=seconds > "$META.started_at.txt"

export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/scripts/authorized_security"
export PB_PKG_DIR="$ROOT/results/aaai_private_base/private_package/gpu_package"
export PB_CKPT_DIR=/root/qwen25_05b
export HF_HUB_OFFLINE=1

python3 scripts/authorized_security/capture_v2_prefill.py \
  --view V2 \
  --adapter results/aaai_private_base/generative_lora/protected_runs/e2e_L12_s1234_full.adapter.pt \
  --queries results/aaai_private_base/generative_lora/data/e2e_test_gen.json \
  --output "$OUT" \
  --run-id e2e_L12_s1234_full \
  --max-samples 500 \
  --dtype fp32

date --iso-8601=seconds > "$META.completed_at.txt"
sha256sum "$OUT" "$OUT.profile.json" > "$META.output_hashes.sha256"
