#!/usr/bin/env bash
# Forced command for the A10->TDX production key.
#
# The key remains unable to execute arbitrary shell commands. It may either:
#   1. enter the fixed trusted service; or
#   2. perform the single fixed training-counter freeze used by the durable G2 driver.
set -euo pipefail

SERVICE_PY=/root/miniconda3/envs/tdx310/bin/python
SERVICE=/root/privacy_llm_obfuscation/scripts/tdx_persistent_service.py
SESSION=/tmp/direct_session.json
COUNTERS=/tmp/e2e_batch_counters.json
FREEZE_CMD='cp /tmp/e2e_batch_counters.json /tmp/e2e_training_counters.frozen.json'
PERSIST=/root/privacy_llm_obfuscation/results/generative_lora_counters/e2e_L12_s1234_full
READ_TRAIN_CMD="cat $PERSIST/training_tdx_counters.json"
READ_GEN_CMD="cat $PERSIST/generation_tdx_counters.json"

persist_counters() {
    local suffix=$1
    if [[ -s "$COUNTERS" ]]; then
        mkdir -p "$PERSIST"
        cp -- "$COUNTERS" "$PERSIST/${suffix}.json"
        sha256sum "$PERSIST/${suffix}.json" > "$PERSIST/${suffix}.json.sha256"
        sync
    fi
}

if [[ "${SSH_ORIGINAL_COMMAND:-}" == "$FREEZE_CMD" ]]; then
    test -s "$COUNTERS"
    cp -- "$COUNTERS" /tmp/e2e_training_counters.frozen.json
    persist_counters training_tdx_counters
    exit 0
fi

if [[ "${SSH_ORIGINAL_COMMAND:-}" == "$READ_TRAIN_CMD" ]]; then
    exec /usr/bin/cat -- "$PERSIST/training_tdx_counters.json"
fi

if [[ "${SSH_ORIGINAL_COMMAND:-}" == "$READ_GEN_CMD" ]]; then
    exec /usr/bin/cat -- "$PERSIST/generation_tdx_counters.json"
fi

# All non-whitelisted commands are ignored and enter the same fixed service as before.
set +e
"$SERVICE_PY" "$SERVICE" "$SESSION"
rc=$?
set -e

if [[ -s "$COUNTERS" ]]; then
    kind=$(
        "$SERVICE_PY" - "$COUNTERS" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1]))
c = obj.get("counters", {})
if int(c.get("ce_batch_calls", 0) or 0) > 0:
    print("training_tdx_counters")
elif int(c.get("decode_calls", 0) or 0) > 0:
    print("generation_tdx_counters")
else:
    print("other_tdx_counters")
PY
    )
    persist_counters "$kind"
fi
exit "$rc"
