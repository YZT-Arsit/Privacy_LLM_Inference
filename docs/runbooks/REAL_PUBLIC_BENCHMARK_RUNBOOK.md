# Real public-benchmark runbook (H800 / TDX)

Prepare LOCAL public benchmark files into normalized JSONL, then run cost-
controlled subsets across the deployment backends (plaintext / folded / TDX-lite /
TDX-attested / folded-LoRA) and consolidate utility + LoRA-utility + claims. NO
internet downloads — you supply the dataset files; licenses are your
responsibility. See also
[`PUBLIC_BENCHMARK_PREPARATION.md`](PUBLIC_BENCHMARK_PREPARATION.md) and the
master [`REAL_H800_TDX_FULL_EVAL_RUNBOOK.md`](REAL_H800_TDX_FULL_EVAL_RUNBOOK.md).

```
MODEL=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
PKGS=/root/autodl-tmp/privacy_llm_packages
BASE=$PKGS/qwen7b_folded_full
ART=$PKGS/qwen7b_boundary_artifact_cuda
LORA=$PKGS/qwen7b_lora_folded_synth_r4
URL=http://127.0.0.1:18083 ; PORT=18083 ; MRTD=<expected_mr_td>
```

## 1. Start H800 worker + 2. health + 3. TDX tunnel

Same as the master runbook steps 1–3 (worker with `--folded-package-path $BASE`
[+ `--folded-lora-package-path $LORA`], `curl $URL/health`, SSH `-L` tunnel).

## 6. Prepare public benchmark JSONL files

```
# MMLU (200-500), GSM8K (100-200), BoolQ (200-500), AG News / SST-2 (500-1000)
python scripts/prepare_public_benchmark_jsonl.py --input-path /data/mmlu_test.csv \
  --dataset-name mmlu --split test --max-examples 300 --seed 2035 \
  --output-jsonl outputs/bench/mmlu_300.jsonl \
  --dataset-card-json outputs/bench/mmlu_300_card.json

python scripts/prepare_public_benchmark_jsonl.py --input-path /data/gsm8k_test.jsonl \
  --dataset-name gsm8k --split test --max-examples 150 --seed 2035 \
  --output-jsonl outputs/bench/gsm8k_150.jsonl \
  --dataset-card-json outputs/bench/gsm8k_150_card.json

python scripts/prepare_public_benchmark_jsonl.py --input-path /data/sst2_dev.tsv \
  --dataset-name sst2 --split dev --max-examples 800 --seed 2035 \
  --output-jsonl outputs/bench/sst2_800.jsonl \
  --dataset-card-json outputs/bench/sst2_800_card.json
```

Each `*_card.json` records `input_file_sha256`, `output_file_sha256`,
`sample_count`, `task_type`, `metric`, `sampling_seed`, `license_note`.

## 7. Run public benchmark subsets

Run the SAME dataset on the backends you want to compare (per-backend report):

```
for B in plaintext_local folded_remote tdx_lite_remote tdx_attested_remote; do
  python scripts/run_e9_task_utility_benchmark.py \
    --dataset-jsonl outputs/bench/mmlu_300.jsonl --task-type multiple_choice \
    --backend $B --model-name Qwen2.5-7B-Instruct --model-path $MODEL \
    --gpu-worker-url $URL --embedding-path $ART \
    --attestation-evidence outputs/attestation_evidence.json --expected-mr-td $MRTD \
    --seq-len 512 --max-new-tokens 8 --dtype bfloat16 --device cpu --audit true \
    --output-json outputs/e9_mmlu_${B}.json --output-csv outputs/e9_mmlu_${B}.csv \
    --output-md outputs/e9_mmlu_${B}.md
done
```

LoRA utility (E10) on a task where the adapter helps (e.g. SST-2): run base,
plaintext-LoRA, and folded-LoRA-remote E9 reports, then:

```
python scripts/run_e10_lora_utility_benchmark.py \
  --dataset-name sst2 --task-type classification --metric-name accuracy \
  --base-json outputs/e9_sst2_plaintext_local.json \
  --plaintext-lora-json outputs/e9_sst2_plaintext_lora.json \
  --folded-lora-json outputs/e9_sst2_folded_lora_remote.json \
  --tdx-attested-folded-lora-json outputs/e9_sst2_tdx_attested_lora.json \
  --lora-verify-json outputs/qwen7b_lora_folded_verify.json \
  --no-lora-decode-json outputs/qwen7b_folded_remote_decode.json \
  --lora-decode-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --output-json outputs/e10_lora_utility.json --output-md outputs/e10_lora_utility.md
```

`folded_lora_preserves_gain=True` + `security_ok=True` ⇒ `utility_preserved=True`.

## 8–10. Attestation (if running attested benchmarks)

Same as the master runbook: `check_tdx_measurement_coverage.py` →
`write_tee_boundary_runtime_hash.py` → `generate_tdx_attestation_evidence.py` →
re-run the attested benchmark with `--attestation-evidence` and confirm
`runtime_hash_bound=True`.

## 11–13. Security scan, negative tests, claim validation

```
python scripts/scan_security_transcript.py \
  --transcript-jsonl outputs/transcript_nolora_tdx.jsonl \
  --output-json outputs/security_transcript_scan.json --fail-on-leak true

python scripts/run_security_negative_tests.py \
  --output-json outputs/security_negative_tests.json

python scripts/validate_paper_claims.py \
  --result-json outputs/e9_mmlu_tdx_attested_remote.json \
  --result-json outputs/e10_lora_utility.json \
  --result-json outputs/security_negative_tests.json \
  --required-claims public_benchmark_utility_preserved \
  --output-json outputs/paper_claim_validation.json
```

`public_benchmark_utility_preserved` is only supported when the E9/E10 evidence is
`paper_ready=True` (real model, not dry-run). Synthetic-LoRA evidence does not back
a real-HF-adapter utility claim.

## 14–15. Package + cleanup

Same as the master runbook (`package_final_artifacts.py`; stop worker/tunnel).
