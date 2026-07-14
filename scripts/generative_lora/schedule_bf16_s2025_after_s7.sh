#!/bin/bash
set -euo pipefail

cd /root/pllo_pb
ROOT=results/aaai_private_base/generative_lora/extension/g1_bf16_matched
PREV="$ROOT/seed7"
NEXT="$ROOT/seed2025"
LOG=/root/g1_bf16_matched_s2025.log

while true; do
  if python3 - "$PREV/terminal_manifest.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
m = json.loads(p.read_text())
ok = (m.get("status") == "complete" and m.get("steps") == 750 and
      m.get("generations") == 500 and m.get("unique_sample_ids") == 500 and
      m.get("all_finite") is True)
raise SystemExit(0 if ok else 2)
PY
  then
    break
  fi
  sleep 20
done

while pgrep -f '[b]f16_matched_lora.py' >/dev/null; do sleep 5; done
test ! -e "$NEXT"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
test "$used" -lt 500

nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python3 scripts/generative_lora/bf16_matched_lora.py train-and-generate \
  --seed 2025 \
  --schedule results/aaai_private_base/generative_lora/data/e2e_schedule_s2025.json \
  --train-data results/aaai_private_base/generative_lora/data/e2e_train_a10.pt \
  --generation-input results/aaai_private_base/generative_lora/data/e2e_test_gen.json \
  --base-checkpoint /root/qwen25_05b --tokenizer /root/genlora_tok \
  --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --rank 8 --alpha 16 --dropout 0 \
  --lr 0.0002 --weight-decay 0.01 \
  --max-steps 750 --batch-size 16 --grad-accum 1 --grad-clip none \
  --generation-dtype fp32 --generation-max 500 --max-new 96 \
  --output-dir results/aaai_private_base/generative_lora/extension/g1_bf16_matched/seed2025' \
  > "$LOG" 2>&1 < /dev/null &
echo "$!" > /root/g1_bf16_matched_s2025.pid
