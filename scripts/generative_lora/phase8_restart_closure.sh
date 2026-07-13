#!/usr/bin/env bash
set -euo pipefail

cd /root/pllo_pb

ROOT=results/aaai_private_base/generative_lora
HANDOFF="$ROOT/handoff"
ADAPTER="$HANDOFF/adapter_package_e2e_L12_s1234_full"
OUT_A="$HANDOFF/restart_A.jsonl"
OUT_B="$HANDOFF/restart_B.jsonl"
REPORT="$HANDOFF/restart_reproducibility.json"
LIFECYCLE="$HANDOFF/lifecycle_gate.json"

for path in "$OUT_A" "$OUT_A.profile.json" "$OUT_B" "$OUT_B.profile.json" "$REPORT" "$LIFECYCLE"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite frozen restart artifact: $path" >&2
    exit 9
  fi
done

if pgrep -f 'a10_batch_runner.py|a10_mixed_runner.py|plaintext_lora.py' >/dev/null; then
  echo "training process still active; refusing fresh-process closure" >&2
  exit 10
fi

export PYTHONPATH=/root/pllo_pb/src
export PB_PKG_DIR=/root/pllo_pb/results/aaai_private_base/private_package/gpu_package
export PB_CKPT_DIR=/root/qwen25_05b
export HF_HUB_OFFLINE=1

COMMON=(
  --session /tmp/e2e_session_a10.json
  --tdx root@172.30.25.153
  --key /root/.ssh/a10_to_tdx
  --service-cmd "/root/miniconda3/envs/tdx310/bin/python /root/privacy_llm_obfuscation/scripts/tdx_persistent_service.py /tmp/direct_session.json"
  --adapter-package "$ADAPTER"
  --expect-base-root bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1
  --expect-model-cfg 479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b
  --expect-targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --expect-rank 8
  --expect-transform-version private_base_fold_v1.0
  --min-optimizer-state-version 1
  --tok /root/genlora_tok
  --gen-in "$ROOT/data/e2e_test_gen.json"
  --gen-max 500
  --max-new 96
  --dtype fp32
  --require-attestation
)

start_a=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 scripts/generative_lora/protected_generate.py "${COMMON[@]}" \
  --gen-out "$OUT_A" --cell G2_fresh_restart_A
end_a=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# The first Python interpreter has exited here. Assert termination before creating
# a second, independently loaded inference process.
if pgrep -f 'protected_generate.py.*restart_A.jsonl' >/dev/null; then
  echo "restart A process did not terminate" >&2
  exit 11
fi

start_b=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 scripts/generative_lora/protected_generate.py "${COMMON[@]}" \
  --gen-out "$OUT_B" --cell G2_fresh_restart_B
end_b=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 - "$start_a" "$end_a" "$start_b" "$end_b" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path("results/aaai_private_base/generative_lora")
handoff = root / "handoff"
a_path = handoff / "restart_A.jsonl"
b_path = handoff / "restart_B.jsonl"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

a = rows(a_path)
b = rows(b_path)
pa = json.loads(Path(str(a_path) + ".profile.json").read_text())
pb = json.loads(Path(str(b_path) + ".profile.json").read_text())
assert len(a) == len(b) == 500
assert [x["sample_id"] for x in a] == [x["sample_id"] for x in b]
token_match = [x["token_ids"] == y["token_ids"] for x, y in zip(a, b)]
text_match = [x["generated_text"] == y["generated_text"] for x, y in zip(a, b)]
report = {
    "schema": "fresh_process_restart_reproducibility",
    "version": "1.0",
    "process_A": {"start_utc": sys.argv[1], "end_utc": sys.argv[2], "profile": str(a_path) + ".profile.json"},
    "process_B": {"start_utc": sys.argv[3], "end_utc": sys.argv[4], "profile": str(b_path) + ".profile.json"},
    "training_processes_absent_before_launch": True,
    "process_A_terminated_before_B": True,
    "rows": len(a),
    "token_agreement": sum(token_match) / len(token_match),
    "generation_agreement": sum(text_match) / len(text_match),
    "output_hash_A": sha(a_path),
    "output_hash_B": sha(b_path),
    "hash_agreement": sha(a_path) == sha(b_path),
    "adapter_hash_A": pa["adapter_hash"],
    "adapter_hash_B": pb["adapter_hash"],
    "adapter_hash_agreement": pa["adapter_hash"] == pb["adapter_hash"],
    "package_hash_A": pa["base_package_root_hash"],
    "package_hash_B": pb["base_package_root_hash"],
    "package_hash_agreement": pa["base_package_root_hash"] == pb["base_package_root_hash"],
    "attestation_A": pa["attestation_verified"],
    "attestation_B": pb["attestation_verified"],
    "adapter_source_A": pa["adapter_source"],
    "adapter_source_B": pb["adapter_source"],
    "plaintext_checkpoint_loaded": False,
    "optimizer_state_loaded": False,
    "training_object_loaded": False,
    "plaintext_adapter_loaded": False,
}
report["fresh_process_ok"] = all([
    report["token_agreement"] == 1.0,
    report["generation_agreement"] == 1.0,
    report["hash_agreement"],
    report["adapter_hash_agreement"],
    report["package_hash_agreement"],
    report["attestation_A"],
    report["attestation_B"],
])
(handoff / "restart_reproducibility.json").write_text(json.dumps(report, indent=2))
lifecycle = {
    "fresh_process_ok": report["fresh_process_ok"],
    "restart_report": str(handoff / "restart_reproducibility.json"),
    "transformed_base_only": True,
    "transformed_adapter_only": True,
    "no_plaintext_checkpoint_or_adapter": True,
    "process_A_terminated_before_B": True,
}
(handoff / "lifecycle_gate.json").write_text(json.dumps(lifecycle, indent=2))
print(json.dumps(report, indent=2))
assert report["fresh_process_ok"]
PY

echo FRESH_PROCESS_RESTART_CLOSURE_PASS
