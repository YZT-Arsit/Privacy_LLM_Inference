#!/usr/bin/env bash
set -euo pipefail

cd /root/pllo_pb
ROOT=results/aaai_private_base/generative_lora
OUT="$ROOT/target_ablation"
ADAPTER="$OUT/qv_s1234_adapter"
META="$OUT/qv_s1234_train.json"
GEN="$OUT/qv_s1234_test.jsonl"

for path in "$ADAPTER" "$META" "$GEN" "$GEN.profile.json" "$OUT/metrics"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite target-ablation artifact: $path" >&2
    exit 9
  fi
done

# Queue behind the fresh-process lifecycle job; never share the single A10.
deadline=$((SECONDS + 3600))
while [[ ! -f "$ROOT/handoff/restart_reproducibility.json" ]]; do
  if (( SECONDS >= deadline )); then
    echo "restart closure did not finish within one hour" >&2
    exit 10
  fi
  sleep 10
done
python3 - <<'PY'
import json
from pathlib import Path
x=json.loads(Path("results/aaai_private_base/generative_lora/handoff/restart_reproducibility.json").read_text())
assert x["fresh_process_ok"]
PY
if pgrep -f 'protected_generate.py|a10_batch_runner.py|a10_mixed_runner.py' >/dev/null; then
  echo "GPU job still active after restart marker" >&2
  exit 11
fi

mkdir -p "$OUT"
python3 scripts/generative_lora/plaintext_lora.py train \
  --ckpt /root/qwen25_05b \
  --targets q_proj,v_proj --rank 8 --alpha 16 --dtype fp32 --seed 1234 \
  --train-data "$ROOT/data/e2e_train_a10.pt" \
  --schedule "$ROOT/data/e2e_schedule_s1234.json" \
  --lr 0.0002 --warmup 0.03 --max-steps 750 \
  --out-adapter "$ADAPTER" --out-meta "$META"

# Training process has exited. Generate the same frozen 500-example test subset.
python3 scripts/generative_lora/plaintext_lora.py generate \
  --ckpt /root/qwen25_05b \
  --targets q_proj,v_proj --rank 8 --alpha 16 --dtype fp32 --seed 1234 \
  --adapter "$ADAPTER" --tok /root/genlora_tok \
  --gen-in "$ROOT/data/e2e_test_gen.json" --gen-out "$GEN" \
  --gen-max 500 --max-new 96 --cell TARGET_ABLATION_QV_s1234 \
  --base-hash 88c142557820ccad

python3 scripts/generative_lora/phase6_metrics.py \
  --gen "ALL7=$ROOT/generations/g1_final_s1234_test.jsonl" "QV=$GEN" \
  --baseline ALL7 --out "$OUT/metrics"

python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root=Path("results/aaai_private_base/generative_lora")
out=root/"target_ablation"
files=[
 out/"qv_s1234_train.json",
 out/"qv_s1234_test.jsonl",
 out/"qv_s1234_test.jsonl.profile.json",
 out/"qv_s1234_adapter"/"adapter_config.json",
 out/"qv_s1234_adapter"/"adapter_model.safetensors",
 out/"metrics"/"aggregate_metrics.csv",
 out/"metrics"/"statistical_summary.json",
]
manifest={
 "schema":"minimal_target_ablation_manifest",
 "seed":1234,
 "comparison":{"T1":"q_proj+v_proj","T4":"all_seven_targets"},
 "all_seven_source":"frozen G1_s1234; not rerun",
 "qv_run":"one permitted frozen-seed ablation",
 "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}
(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
PY

echo MINIMAL_TARGET_ABLATION_COMPLETE
