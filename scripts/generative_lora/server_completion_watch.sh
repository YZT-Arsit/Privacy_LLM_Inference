#!/usr/bin/env bash
# Wait for the durable G2 driver terminal marker, then run one immediate server-side collection.
set -euo pipefail

REPO=/root/pllo_pb
LOG=$REPO/g2_e2e_L12_s1234_full.log
AUDIT=$REPO/scripts/generative_lora/server_progress_audit.py
OUT=$REPO/results/aaai_private_base/generative_lora/monitor/e2e_L12_s1234_full

while true; do
    if grep -qx 'G2_ALL_DONE' "$LOG" 2>/dev/null; then
        exec /usr/bin/python3 "$AUDIT" \
            --driver-pid 8058 --worker-pid 8060 --total-steps 750 \
            --log "$LOG" --out-dir "$OUT" \
            --tdx-private-ip 172.30.25.153 --tdx-key /root/.ssh/a10_to_tdx \
            --gpu-samples 1 --sample-interval 0
    fi
    if grep -qE '^(G2_TRAIN_FAILED|G2_GEN_FAILED|G2_COUNTER_FREEZE_FAILED)$' "$LOG" 2>/dev/null; then
        exec /usr/bin/python3 "$AUDIT" \
            --driver-pid 8058 --worker-pid 8060 --total-steps 750 \
            --log "$LOG" --out-dir "$OUT" \
            --tdx-private-ip 172.30.25.153 --tdx-key /root/.ssh/a10_to_tdx \
            --gpu-samples 1 --sample-interval 0
    fi
    sleep 10
done
