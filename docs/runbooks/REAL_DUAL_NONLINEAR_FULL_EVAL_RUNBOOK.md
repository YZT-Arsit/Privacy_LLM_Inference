# Real dual-nonlinear full evaluation runbook (H800 / TDX)

Run the **complete** experiment suite under **both** nonlinear-layer designs so
the advisor can choose either design later and still have full results. The two
designs are a first-class experimental dimension (`--nonlinear-backend`):

- **`current`** (design A / baseline) — nonlinear islands evaluated inside the
  trusted boundary. Security `established` (the validated path).
- **`trusted_shortcut`** (design B / alternative, alias `amulet_migrated`) — the
  bulk of the nonlinearity migrated onto the untrusted accelerator with a small
  trusted reduction shortcut. Security **not formally claimed** (under
  discussion). Correctness exact.

CRITICAL invariants (enforced by the code):
- The folded package manifest records `nonlinear_backend` + a
  `nonlinear_design_metadata_hash`; verifiers FAIL on a mismatch
  (`--expected-nonlinear-backend`), and a folded-LoRA package must match its
  base package's design.
- The TDX **runtime hash binds the design** (`write_tee_boundary_runtime_hash.py
  --nonlinear-backend …`). `runtime_hash(current) != runtime_hash(trusted_shortcut)`,
  so **attestation evidence for design A cannot be reused for design B** — you
  must regenerate the runtime hash and re-bind a fresh TD Quote per design.
- The claim validator tags evidence by design (`claim[current]` /
  `claim[trusted_shortcut]`); a claim for one design can never be backed by the
  other's evidence.

> Do not start until `scripts/preflight_real_eval.py --nonlinear-backends
> current,trusted_shortcut` passes for the designs you intend to run.

```
MODEL=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
ROOT=/root/autodl-tmp/privacy_llm_packages
OUT=outputs/dual_nonlinear
URL=http://127.0.0.1:18083 ; PORT=18083
DESIGNS="current trusted_shortcut"
```

## 0. Generate the plan + preflight (no server time)

```
python scripts/run_dual_nonlinear_experiment_matrix.py \
  --nonlinear-backends current,trusted_shortcut \
  --model-path $MODEL --model-name Qwen2.5-7B-Instruct \
  --base-output-root $ROOT --outputs-dir $OUT \
  --seq-len 128 --max-new-tokens-list 1,4,8,16 --run-mode plan \
  --include-build true --include-local-probes true --include-remote-decode true \
  --include-tdx-lite true --include-tdx-attested true --include-lora true \
  --include-public-benchmarks true --include-latency true --include-security true \
  --output-json $OUT/dual_matrix_plan.json --output-md $OUT/dual_matrix_plan.md

python scripts/preflight_real_eval.py --backend tdx_attested_remote \
  --nonlinear-backends current,trusted_shortcut --namespace-by-backend \
  --model-path $MODEL --base-folded-package-path $ROOT/qwen7b_folded_full \
  --embedding-artifact-path $ROOT/qwen7b_boundary_artifact \
  --expected-mr-td <MRTD> \
  --output-json $OUT/preflight_matrix.json --output-md $OUT/preflight_matrix.md
```

The plan namespaces every artifact by design
(`qwen7b_folded_full_<design>`, `qwen7b_boundary_artifact_<design>`,
`qwen7b_lora_folded_<design>`, `outputs/<design>/…`). The steps below are the
plan executed per design — loop `for D in $DESIGNS`.

## 1. Build both folded packages

```
for D in $DESIGNS; do
  python scripts/build_qwen7b_folded_package.py --model-path $MODEL \
    --output-dir $ROOT/qwen7b_folded_full_$D --seq-len 128 --num-layers 28 \
    --dtype bfloat16 --nonlinear-backend $D --mask-schedule session \
    --shard-by-layer true --write-manifest true \
    --output-json $OUT/$D/build_$D.json
done
```

## 2. Verify both packages (design recorded + matches)

```
for D in $DESIGNS; do
  python scripts/verify_folded_package.py \
    --package-path $ROOT/qwen7b_folded_full_$D --expected-nonlinear-backend $D \
    --output-json $OUT/$D/verify_$D.json
done
```

## 3. Build both boundary (embedding) artifacts

```
for D in $DESIGNS; do
  python scripts/build_qwen7b_embedding_artifact.py --model-path $MODEL \
    --output-dir $ROOT/qwen7b_boundary_artifact_$D --device cuda --dtype bfloat16
done
```

## 4. Start the H800 worker for a design

The worker loads a folded package; start one per design (or restart between
designs). Run the matching `--folded-package-path` for the design under test:

```
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package \
  --folded-package-path $ROOT/qwen7b_folded_full_$D \
  --folded-lora-package-path $ROOT/qwen7b_lora_folded_$D \
  --device cuda --dtype bfloat16 --listen-port $PORT --audit true
curl $URL/health
```

## 5. Local probes for each design

```
for P in prefill onestep_logits decode; do
  python scripts/run_qwen7b_folded_package_${P/onestep_logits/onestep_logits}_probe.py \
    --package-path $ROOT/qwen7b_folded_full_$D --nonlinear-backend $D \
    --output-json $OUT/$D/local_${P}_$D.json
done
```
(Scripts: `run_qwen7b_folded_package_prefill_probe.py`,
`run_qwen7b_folded_package_onestep_logits_probe.py`,
`run_qwen7b_folded_package_decode_probe.py` — each takes `--nonlinear-backend`.)

## 6. Remote H800 package-backed decode per design

```
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --nonlinear-backend $D --max-new-tokens 16 --audit true \
  --output-json $OUT/$D/remote_decode_$D.json
```

## 7. TDX-lite decode per design

Run the boundary in-process (TDX guest) with NO attestation evidence
(`boundary_mode=lite`):

```
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --boundary-backend process --nonlinear-backend $D --max-new-tokens 16 \
  --audit true --output-json $OUT/$D/tdx_lite_decode_$D.json
```

## 8. Regenerate the TDX runtime hash **per design**

The design is part of the runtime identity — design A and B get different
hashes. Freeze the boundary code, then for each design:

```
python scripts/check_tdx_measurement_coverage.py        # must be OK
python scripts/write_tee_boundary_runtime_hash.py \
  --boundary-backend process --gpu-backend qwen7b \
  --nonlinear-backend $D --expected-mr-td <MRTD> \
  --output $OUT/$D/runtime_hash_$D.txt
```

## 9. Generate the TD Quote + attestation evidence per design

Bind the design's runtime hash into the TD Quote `report_data`, obtain the signed
JWT, and assemble `attestation_evidence_$D.json`:

```
python scripts/generate_tdx_attestation_evidence.py \
  --runtime-hash $(cat $OUT/$D/runtime_hash_$D.txt) \
  --output-json $OUT/$D/attestation_evidence_$D.json
```
Do NOT reuse design A's evidence for design B — the binding will fail
(`runtime_hash_bound=False`).

## 10. TDX-attested decode per design

```
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --boundary-backend process --nonlinear-backend $D \
  --attestation-evidence $OUT/$D/attestation_evidence_$D.json \
  --expected-mr-td <MRTD> --write-runtime-manifest $OUT/$D/runtime_manifest_$D.json \
  --max-new-tokens 16 --audit true \
  --output-json $OUT/$D/tdx_attested_decode_$D.json
```
Confirm `boundary_attested=True` and `runtime_hash_bound=True`.

## 11. LoRA pipeline per design

```
python scripts/build_qwen7b_lora_folded_package.py \
  --base-folded-package-path $ROOT/qwen7b_folded_full_$D \
  --output-dir $ROOT/qwen7b_lora_folded_$D --nonlinear-backend $D \
  --output-json $OUT/$D/lora_build_$D.json
python scripts/verify_qwen7b_lora_folded_package.py \
  --lora-folded-package-path $ROOT/qwen7b_lora_folded_$D \
  --base-folded-package-path $ROOT/qwen7b_folded_full_$D \
  --expected-nonlinear-backend $D --output-json $OUT/$D/lora_verify_$D.json
python scripts/run_qwen7b_lora_folded_local_probe.py --nonlinear-backend $D \
  --output-json $OUT/$D/lora_local_$D.json
python scripts/run_qwen7b_lora_folded_remote_decode_probe.py \
  --gpu-worker-url $URL --nonlinear-backend $D \
  --attestation-evidence $OUT/$D/attestation_evidence_$D.json \
  --expected-mr-td <MRTD> --output-json $OUT/$D/lora_tdx_attested_$D.json
```

## 12. Public benchmarks per design (`--require-real`)

> **Use the REAL staged data, never the tiny unit-test fixtures.** The raw public
> benchmark files are on the server at
> `/root/autodl-tmp/datasets/privacy_llm_benchmarks/raw`. Convert each to a
> normalized JSONL with `scripts/prepare_public_benchmark_jsonl.py` (writes a
> dataset card with input/output sha256), e.g.:
> ```
> RAW=/root/autodl-tmp/datasets/privacy_llm_benchmarks/raw
> python scripts/prepare_public_benchmark_jsonl.py --input-path $RAW/mmlu/test.csv \
>   --dataset-name mmlu --split test --max-examples 300 --seed 2035 \
>   --output-jsonl outputs/bench/mmlu.jsonl --dataset-card-json outputs/bench/mmlu_card.json
> ```
> `tests/fixtures/benchmarks/*` are for unit tests ONLY and must never appear in a
> paper-facing run (the gate/claim validator reject dry-run/fixture evidence).

```
for DS in mmlu gsm8k boolq sst2; do
  for B in plaintext_local tdx_attested_remote; do
    python scripts/run_e9_task_utility_benchmark.py --require-real \
      --dataset-jsonl outputs/bench/${DS}.jsonl --backend $B \
      --nonlinear-backend $D --model-path $MODEL --gpu-worker-url $URL \
      --embedding-path $ROOT/qwen7b_boundary_artifact_$D \
      --attestation-evidence $OUT/$D/attestation_evidence_$D.json \
      --expected-mr-td <MRTD> \
      --output-json $OUT/$D/e9_${DS}_${B}_$D.json
  done
done
```
(Plaintext baseline is design-agnostic but recorded per design for clean pairing.)

## 13. Pairwise + aggregate utility preservation per design

```
for DS in mmlu gsm8k boolq sst2; do
  python scripts/run_e9_pairwise_utility_preservation.py \
    --baseline-json  $OUT/$D/e9_${DS}_plaintext_local_$D.json \
    --candidate-json $OUT/$D/e9_${DS}_tdx_attested_remote_$D.json \
    --dataset $DS --output-json $OUT/$D/e9_${DS}_pairwise_$D.json
done
python scripts/run_e9_pairwise_utility_preservation.py --aggregate \
  --pairwise-json $OUT/$D/e9_mmlu_pairwise_$D.json \
  --pairwise-json $OUT/$D/e9_gsm8k_pairwise_$D.json \
  --pairwise-json $OUT/$D/e9_boolq_pairwise_$D.json \
  --pairwise-json $OUT/$D/e9_sst2_pairwise_$D.json \
  --output-json $OUT/$D/e9_aggregate_$D.json
```
The pairwise/aggregate reports inherit the candidate's `nonlinear_backend`, so
they back `public_benchmark_utility_preserved[$D]`.

## 14. Security transcript scan (per design)

```
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --boundary-backend process --nonlinear-backend $D \
  --record-transcript $OUT/$D/transcript_$D.jsonl --max-new-tokens 8 --audit true \
  --output-json $OUT/$D/transcript_decode_$D.json
python scripts/scan_security_transcript.py \
  --transcript-jsonl $OUT/$D/transcript_$D.jsonl \
  --output-json $OUT/$D/transcript_scan_$D.json --fail-on-leak true
python scripts/run_security_negative_tests.py \
  --output-json $OUT/security_negative_tests.json     # design-agnostic
```

## 15. Latency baselines per design

```
python scripts/run_e12_latency_baselines.py --nonlinear-backend $D \
  --plaintext-h800-json $OUT/$D/remote_decode_$D.json \
  --folded-remote-json  $OUT/$D/remote_decode_$D.json \
  --tdx-attested-json   $OUT/$D/tdx_attested_decode_$D.json \
  --output-json $OUT/$D/latency_$D.json --output-md $OUT/$D/latency_$D.md
```

## 16. Nonlinear design comparison (E15) + ablation (E16)

After BOTH designs are complete:

```
python scripts/run_e15_nonlinear_design_comparison.py \
  --current-json $OUT/current/*.json \
  --trusted-shortcut-json $OUT/trusted_shortcut/*.json \
  --output-json $OUT/e15_comparison.json --output-md docs/paper_draft/evaluation_nonlinear_designs.md
python scripts/run_e16_nonlinear_ablation_report.py \
  --current-json $OUT/current/*.json \
  --trusted-shortcut-json $OUT/trusted_shortcut/*.json \
  --output-json $OUT/e16_ablation.json --output-md $OUT/e16_ablation.md \
  --output-csv $OUT/e16_ablation.csv --output-tex $OUT/e16_ablation.tex
```
E15 emits `recommendation_status=insufficient_evidence` + `missing_evidence`
unless BOTH designs have complete evidence for the compared axis.

## 17. Final claim validator (backend-tagged)

```
python scripts/validate_paper_claims.py \
  $(for D in $DESIGNS; do for f in $OUT/$D/*.json; do echo --result-json $f; done; done) \
  --result-json $OUT/security_negative_tests.json \
  --required-claims \
"no_lora_tdx_attested_remote_package_decode[current],\
no_lora_tdx_attested_remote_package_decode[trusted_shortcut],\
public_benchmark_utility_preserved[current],\
public_benchmark_utility_preserved[trusted_shortcut]" \
  --output-json $OUT/paper_claim_validation.json
```
If you claim only one design, drop the other's required claims AND state in the
paper that the other design was evaluated-but-not-claimed (the validator reports
`nonlinear_designs_evaluated` / `nonlinear_designs_not_evaluated`).

## 18. Final submission gate

```
python scripts/final_submission_gate.py \
  $(for D in $DESIGNS; do for f in $OUT/$D/*.json; do echo --result-json $f; done; done) \
  --result-json $OUT/security_negative_tests.json \
  --nonlinear-backends current,trusted_shortcut \
  --claim-tdx-attested-lora --final-artifact-tar $OUT/final_artifacts.tar.gz \
  --output-json $OUT/final_gate.json --output-md $OUT/final_gate.md
```

## 19. Package artifacts

```
python scripts/package_final_artifacts.py --nonlinear-backends current,trusted_shortcut \
  --output-tar $OUT/final_artifacts.tar.gz --output-json $OUT/final_artifacts.json
```

---
**Limitations carried forward:** `trusted_shortcut` security is not formally
claimed; the dual matrix only proves what its per-design inputs prove (dry-run
inputs are never paper-ready); the runtime hash + TD Quote MUST be regenerated
per design; the masked-remote E9 backend loads a trusted-side tokenizer on the
boundary (heavier than the validated no-LoRA TDX-lite boundary).
