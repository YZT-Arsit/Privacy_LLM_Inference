#!/bin/bash
set -euo pipefail

ROOT=/Users/Hoshino/Desktop/privacy_llm_obfuscation
KEY="$HOME/Downloads/passkey.pem"
A10=root@8.147.119.67
TDX=root@8.147.112.139
PAIR_ROOT=/root/pllo_pb
TAG=e2e_T2_qkvo_s1234_protected
NEXT=e2e_T4_all7_s1234_protected
LOG="$ROOT/results/aaai_private_base/generative_lora/extension/monitor/t4_scheduler.log"
HEARTBEAT="$ROOT/results/aaai_private_base/generative_lora/extension/monitor/t4_scheduler_heartbeat.json"
LOCAL_RUN="$ROOT/results/aaai_private_base/generative_lora/protected_runs"
LOCAL_GEN="$ROOT/results/aaai_private_base/generative_lora/generations"

mkdir -p "$(dirname "$LOG")" "$LOCAL_RUN" "$LOCAL_GEN"
exec >>"$LOG" 2>&1
echo "$(date -Iseconds) scheduler_start pid=$$"

while true; do
  now=$(date -Iseconds)
  tail_line=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$KEY" "$A10" \
    "tail -1 '$PAIR_ROOT/g2_${TAG}.log' 2>/dev/null" || true)
  printf '{"timestamp":"%s","pid":%d,"waiting_for":"%s","last_log_line":%s}\n' \
    "$now" "$$" "$TAG" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$tail_line")" > "$HEARTBEAT"
  if ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$KEY" "$A10" \
      "grep -q '^G2_ALL_DONE$' '$PAIR_ROOT/g2_${TAG}.log'"; then
    break
  fi
  sleep 30
done

echo "$(date -Iseconds) t2_terminal_marker_seen"
sleep 10
if ssh -o BatchMode=yes -i "$KEY" "$A10" \
    "pgrep -af 'a10_batch_runner.py|protected_generate.py' | grep -v 'pgrep -af'"; then
  echo "T2 terminal marker exists but a GPU worker is still active; refusing T4 launch" >&2
  exit 31
fi

ssh -o BatchMode=yes -i "$KEY" "$A10" "python3 - <<'PY'
import json
run=json.load(open('$PAIR_ROOT/results/aaai_private_base/generative_lora/protected_runs/${TAG}.json'))
rows=[json.loads(x) for x in open('$PAIR_ROOT/results/aaai_private_base/generative_lora/generations/g2_protected_${TAG}.jsonl') if x.strip()]
profile=json.load(open('$PAIR_ROOT/results/aaai_private_base/generative_lora/generations/g2_protected_${TAG}.jsonl.profile.json'))
assert run['steps_run'] == 750 and len(run['trajectory']) == 750
assert all(x['finite'] for x in run['trajectory'])
assert len(rows) == 500 and len({x['sample_id'] for x in rows}) == 500
assert profile['attestation_verified'] is True
print('T2_REMOTE_VALIDATION_OK')
PY"

for suffix in json adapter.pt; do
  scp -i "$KEY" "$A10:$PAIR_ROOT/results/aaai_private_base/generative_lora/protected_runs/${TAG}.${suffix}" "$LOCAL_RUN/"
done
for suffix in jsonl jsonl.profile.json; do
  scp -i "$KEY" "$A10:$PAIR_ROOT/results/aaai_private_base/generative_lora/generations/g2_protected_${TAG}.${suffix}" "$LOCAL_GEN/"
done
scp -i "$KEY" "$A10:$PAIR_ROOT/g2_${TAG}.log" \
  "$ROOT/results/aaai_private_base/generative_lora/extension/monitor/${TAG}.completed.log"
scp -i "$KEY" "$TDX:/tmp/e2e_training_counters.frozen.json" "$LOCAL_RUN/${TAG}.training_tdx_counters.json"
scp -i "$KEY" "$TDX:/tmp/e2e_batch_counters.json" "$LOCAL_RUN/${TAG}.generation_tdx_counters.json"
scp -i "$KEY" "$TDX:/tmp/e2e_attest/session_attestation.json" "$LOCAL_RUN/${TAG}.attestation.json"

python3 - <<PY
import json
from pathlib import Path
root=Path('$ROOT/results/aaai_private_base/generative_lora')
run=json.load(open(root/'protected_runs/${TAG}.json'))
rows=[json.loads(x) for x in open(root/'generations/g2_protected_${TAG}.jsonl') if x.strip()]
att=json.load(open(root/'protected_runs/${TAG}.attestation.json'))
assert run['steps_run'] == 750 and len(run['trajectory']) == 750 and all(x['finite'] for x in run['trajectory'])
assert len(rows) == 500 and len({x['sample_id'] for x in rows}) == 500
assert att['attestation_verified'] is True
print('T2_LOCAL_PRESERVATION_OK')
PY

if ssh -o BatchMode=yes -i "$KEY" "$A10" \
    "test -e '$PAIR_ROOT/results/aaai_private_base/generative_lora/protected_runs/${NEXT}.json' -o -e '$PAIR_ROOT/g2_${NEXT}.log'"; then
  echo "T4 output/log already exists; refusing duplicate launch" >&2
  exit 32
fi
if test -e "$LOCAL_RUN/${NEXT}.json" -o -e "$LOCAL_GEN/g2_protected_${NEXT}.jsonl"; then
  echo "local T4 output already exists; refusing duplicate launch" >&2
  exit 33
fi

CM=/tmp/cm-a10-pair2
rm -f "$CM"
ssh -MNf -S "$CM" -o ControlPersist=12h -o StrictHostKeyChecking=accept-new -i "$KEY" "$A10"
echo "$(date -Iseconds) launching_t4"
cd "$ROOT"
G2_CM_A10="$CM" \
G2_A10_PUB="$A10" \
G2_A10=root@172.30.25.155 \
G2_TDX_PUB="$TDX" \
G2_TDX_PRIV=172.30.25.156 \
G2_SSH_KEY="$KEY" \
python3 scripts/generative_lora/gate_e2e_protected.py \
  --profile L12 --seed 1234 --lr 0.0002 \
  --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max-steps 750 --gen-max 500 --max-new 96 --attest \
  --run-tag "$NEXT" --timeout 30000
echo "$(date -Iseconds) scheduler_complete"
