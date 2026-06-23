#!/usr/bin/env bash
# Stage 8.2 -- top-conference experiment runner (GPU server).
#
# ModelScope cache only (never Hugging Face remote). Compact JSON/MD/CSV only.
# Run from the repo root on the GPU server. Requires:
#   PYTHON           -> python interpreter (default: python)
#   MODELSCOPE_CACHE -> ModelScope cache dir holding the Qwen2.5 checkpoints
#
# Usage:  PYTHON=python MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache \
#         bash scripts/run_topconf_experiments.sh
set -u
PYTHON="${PYTHON:-python}"
CACHE="${MODELSCOPE_CACHE:?set MODELSCOPE_CACHE to your ModelScope cache dir}"
PROMPTS="outputs/real_prompts_stage8_2.jsonl"
MS="--dtype bfloat16 --folding-dtype float32 --folded-weight-runtime-dtype float32 --recovery-dtype float32 --compare-dtype float32"
PROBE="$PYTHON scripts/run_modelscope_real_checkpoint_probe.py --cache-dir $CACHE --device cuda $MS --mask-mode signed_permutation --residual-mask-strategy shared --prompt-file $PROMPTS"
EVALS="$PYTHON scripts/run_stage8_2_topconf_evals.py --cache-dir $CACHE --device cuda $MS --prompt-file $PROMPTS"

echo "### Step 1: targeted tests"
$PYTHON -m pytest tests/test_modelscope_real_checkpoint_probe.py -q || exit 1
$PYTHON -m pytest tests/test_hf_causal_lm_skeleton.py -q || exit 1

echo "### Group A: latency/memory baseline"
$PROBE --model-id Qwen/Qwen2.5-0.5B-Instruct --max-layers 2   --prefill-seq-len 128 --decode-steps 16 \
  --output outputs/eval_latency_0_5b_layers2_realprompts.json
$PROBE --model-id Qwen/Qwen2.5-0.5B-Instruct --max-layers all --prefill-seq-len 128 --decode-steps 16 \
  --output outputs/eval_latency_0_5b_full_layers_realprompts.json
$PROBE --model-id Qwen/Qwen2.5-3B-Instruct   --max-layers 2   --prefill-seq-len 128 --decode-steps 16 \
  --output outputs/eval_latency_3b_layers2_realprompts.json

echo "### Group B: full-layer 0.5B real-prompt probe"
$PROBE --model-id Qwen/Qwen2.5-0.5B-Instruct --max-layers all --prefill-seq-len 128 --decode-steps 16 \
  --negative-control none \
  --output outputs/eval_full_layer_0_5b_realprompts_seq128_decode16_bf16_mixed_safe.json

echo "### Group C+D: output-boundary ablation + leakage/attack metrics (one load)"
$EVALS --model-id Qwen/Qwen2.5-0.5B-Instruct --max-layers 2 --prefill-seq-len 128 --decode-steps 16 \
  --model-tag 0_5b

echo "### Group E: batch-size scaling (1,2,4,8)"
$PYTHON scripts/run_batch_scaling.py --cache-dir "$CACHE" --device cuda $MS \
  --model-id Qwen/Qwen2.5-0.5B-Instruct --max-layers 2 --prefill-seq-len 128 --decode-steps 16 \
  --prompt-file "$PROMPTS" --batch-sizes 1,2,4,8 \
  --output outputs/eval_batch_scaling_0_5b_realprompts.json

echo "### Group F: 3B real-prompt partial-layer validation"
$PROBE --model-id Qwen/Qwen2.5-3B-Instruct --max-layers 2 --prefill-seq-len 128 --decode-steps 16 \
  --negative-control none \
  --output outputs/eval_realprompts_3b_layers2_seq128_decode16_bf16_mixed_safe.json

echo "### Group G: 7B smoke (optional; tolerate failure)"
$PROBE --model-id Qwen/Qwen2.5-7B-Instruct --max-layers 1 --prefill-seq-len 128 --decode-steps 4 \
  --output outputs/eval_7b_smoke_layers1_realprompt_bf16_mixed_safe.json \
  || echo "Group G failed (OOM/disk/download) -- recorded; continuing."

echo "### Summary + size guard"
$PYTHON scripts/summarize_topconf_experiments.py --output-dir outputs
$PYTHON scripts/check_output_sizes.py --output-dir outputs --warn-mb 10 --fail-mb 100

echo "### Archive"
ARCHIVE="stage8_2_topconf_outputs_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" outputs/eval_*.json outputs/audit_*.json \
  outputs/topconf_experiment_summary.* outputs/topconf_experiment_tables.csv \
  outputs/real_prompts_stage8_2.jsonl 2>/dev/null
echo "Wrote archive: $ARCHIVE"
