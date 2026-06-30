# Real dual-nonlinear full evaluation runbook (H800 / TDX)

Run the **complete** experiment suite under the **two PAPER-FACING** nonlinear
designs. Both meet the single-TEE-entry / zero-trusted-nonlinear contract; the
design is a first-class experimental dimension (`--nonlinear-backend`):

- **`A_rightmul`** — every nonlinear island (SiLU/SwiGLU, attention softmax,
  RMSNorm/LayerNorm) runs on the untrusted accelerator over the compatible
  right-multiply / permutation-masked state; **zero trusted nonlinear calls**.
  Security `claimed_under_compatible_mask_assumption` (compatible masks
  CHECKED: signed-permutation residual, `Q~K~^T==QK^T`, shared SwiGLU channel
  permutation; arbitrary dense masks are rejected).
- **`amulet_secure_R`** — GELU/SiLU via a dense single-one Kronecker secure-R
  lift (no zero decoys, no visible one-hot selector, secret coordinate +
  shuffles), softmax/RMSNorm directly on the masked state with **no trusted
  reduction shortcut**; **zero trusted nonlinear calls**. Security
  `claimed_under_secure_R_assumption`.

> **Legacy `current` / `trusted_shortcut` are NOT paper-facing** (debug/local
> baselines only). `current` evaluates the nonlinearity in the trusted island and
> `trusted_shortcut` keeps a per-op trusted reduction shortcut — both have
> `nonlinear_trusted_calls > 0` and are REJECTED by the paper-facing build
> (`--paper-facing`), the claim validator (`paper_facing=True`), the submission
> gate, and the end-to-end validator. Do not use them for paper results.

CRITICAL invariants (enforced by the code):
- The folded package manifest records `nonlinear_backend` + a
  `nonlinear_design_metadata_hash`; verifiers FAIL on a mismatch
  (`--expected-nonlinear-backend`), and a folded-LoRA package must match its
  base package's design.
- The TDX **runtime hash binds the design** (`write_tee_boundary_runtime_hash.py
  --nonlinear-backend …`). `runtime_hash(A_rightmul) != runtime_hash(amulet_secure_R)`,
  so **attestation evidence for design A cannot be reused for design B** — you
  must regenerate the runtime hash and re-bind a fresh TD Quote per design.
- The claim validator tags evidence by design (`claim[A_rightmul]` /
  `claim[amulet_secure_R]`); a claim for one design can never be backed by the
  other's evidence.

> Do not start until `scripts/preflight_real_eval.py --nonlinear-backends
> A_rightmul,amulet_secure_R` passes for the designs you intend to run.

```
MODEL=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
ROOT=/root/autodl-tmp/privacy_llm_packages
OUT=outputs/dual_nonlinear
URL=http://127.0.0.1:18083 ; PORT=18083
DESIGNS="A_rightmul amulet_secure_R"
```

## 0. Generate the plan + preflight (no server time)

```
python scripts/run_dual_nonlinear_experiment_matrix.py \
  --nonlinear-backends A_rightmul,amulet_secure_R \
  --model-path $MODEL --model-name Qwen2.5-7B-Instruct \
  --base-output-root $ROOT --outputs-dir $OUT \
  --seq-len 128 --max-new-tokens-list 1,4,8,16 --run-mode plan \
  --include-build true --include-local-probes true --include-remote-decode true \
  --include-tdx-lite true --include-tdx-attested true --include-lora true \
  --include-public-benchmarks true --include-latency true --include-security true \
  --output-json $OUT/dual_matrix_plan.json --output-md $OUT/dual_matrix_plan.md

python scripts/preflight_real_eval.py --backend tdx_attested_remote \
  --nonlinear-backends A_rightmul,amulet_secure_R --namespace-by-backend \
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

`--paper-facing` REQUIRES a paper-facing design (rejects current/trusted_shortcut)
AND asserts Linear-boundary pad coverage on all 8 Linear families from the REAL
shard tensor names (q/k/v/o/gate/up/down/lm_head all have `xpad_tilde`+`cpad_tilde`)
— the build FAILS loudly otherwise. The pad is the main scheme (ON by default).

```
for D in $DESIGNS; do
  python scripts/build_qwen7b_folded_package.py --model-path $MODEL \
    --output-dir $ROOT/qwen7b_folded_full_$D --seq-len 1024 --num-layers 28 \
    --dtype bfloat16 --device cuda --nonlinear-backend $D --paper-facing \
    --mask-schedule session --shard-by-layer true --write-manifest true \
    --output-json $OUT/$D/build_$D.json || { echo "BUILD FAILED for $D"; exit 1; }
done
```

**Validator (fail-stop):** the build with `--paper-facing` already asserts pad
coverage; re-confirm from disk before proceeding:

```
for D in $DESIGNS; do
  python - <<PY || { echo "PAD COVERAGE FAIL $D"; exit 1; }
import sys; sys.path.insert(0, "src")
from pllo.deployment.linear_boundary_pad import assert_paper_facing_pad_coverage
assert_paper_facing_pad_coverage("$ROOT/qwen7b_folded_full_$D")
print("pad coverage OK $D")
PY
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

Pass `--folded-package-path` so the mask seed is synced from the package the
artifact must match, and `--nonlinear-backend` to record the design in
`boundary_meta.json` (provenance only — the design is bound through the folded
manifest + the TDX runtime hash, not the artifact tensors):

```
for D in $DESIGNS; do
  python scripts/build_qwen7b_embedding_artifact.py --model-path $MODEL \
    --folded-package-path $ROOT/qwen7b_folded_full_$D \
    --output-dir $ROOT/qwen7b_boundary_artifact_$D --nonlinear-backend $D \
    --device cuda --dtype bfloat16
done
```

## 4. Start the H800 worker for a design

The worker loads a folded package; start one per design (or restart between
designs). Run the matching `--folded-package-path` for the design under test.
**Pass `--nonlinear-backend $D`** so the untrusted worker actually EXECUTES the
design: `A_rightmul` runs every nonlinear island on the accelerator over the
masked state (zero trusted calls) and stamps measured evidence
(`right_multiply_nonlinear_executed` / `right_multiply_nonlinear_ops_count` /
`trusted_nonlinear_ops_count==0`); `amulet_secure_R` runs the dense single-one
secure-R lift for GELU/SiLU + masked-state softmax/RMSNorm (zero trusted calls)
and stamps `secure_right_multiply_executed` / `secure_R_enabled` /
`zero_decoys==False` / `selector_visible_to_gpu==False`. The client retrieves
these from `/health` and the probes/E3/E9 reports carry them. If you forget it
the worker silently runs `current` and the reports are tag-only / have
`nonlinear_trusted_calls>0` (rejected by the claim validator / gate / E2E).
`ss` is NOT installed on H800 — use `curl /health` (and a portable Python socket
probe), never `ss`:

```
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package --nonlinear-backend $D \
  --folded-package-path $ROOT/qwen7b_folded_full_$D \
  --folded-lora-package-path $ROOT/qwen7b_lora_folded_$D \
  --device cuda --dtype bfloat16 --listen-port $PORT --audit true
# /health now reports nonlinear_backend + nonlinear_execution_evidence (post-run)
curl -fsS $URL/health && echo            # health check (no ss)
python - <<'PY'                          # portable listen probe (no ss)
import socket
s = socket.socket(); s.settimeout(2)
print("18083", "open" if s.connect_ex(("127.0.0.1", 18083)) == 0 else "CLOSED")
PY
```

## 5. Local probes for each design

```
for P in prefill onestep_logits decode; do
  python scripts/run_qwen7b_folded_package_${P}_probe.py \
    --folded-package-path $ROOT/qwen7b_folded_full_$D --model-path $MODEL \
    --nonlinear-backend $D --output-json $OUT/$D/local_${P}_$D.json
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

The quote command MUST include `--nonlinear-backend $D` so the runtime hash /
`report_data` bind the specific design (a real on-TDX quote; never `--simulate`
for paper results):

```
for D in $DESIGNS; do
  python scripts/generate_tdx_attestation_evidence.py \
    --boundary-backend process --gpu-backend qwen7b_folded_package \
    --nonlinear-backend $D \
    --expected-mr-td <MRTD> \
    --quote-command '<tdx_quote_tool ...{report_data_hex}...{quote_out}>' \
    --attest-command '<itrustee/aliyun_attest ...{quote_file}...>' \
    --output-dir $OUT/$D/tdx_$D \
    --output-evidence $OUT/$D/attestation_evidence_$D.json \
    || { echo "QUOTE GEN FAILED $D"; exit 1; }
done
```

The generator verifies and FAILS on: `tee != tdx`, `td_attributes.debug == true`
(unless `--debug-allowed`), missing/!=3-part JWT, `report_data != runtime_hash`,
`mr_td != --expected-mr-td`. The evidence also records `nonlinear_backend` +
`nonlinear_design_metadata_hash` and `runtime_hash_binds_nonlinear_backend=True`.
Do NOT reuse one design's evidence for the other — the binding will fail
(`runtime_hash_bound=False`). Off-TDX `--simulate` evidence is stamped
`paper_facing=false` and is rejected by the E2E validator.

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
  --nonlinear-backends A_rightmul,amulet_secure_R \
  --claim-tdx-attested-lora --final-artifact-tar $OUT/final_artifacts.tar.gz \
  --output-json $OUT/final_gate.json --output-md $OUT/final_gate.md
```

## 19. Package artifacts

```
python scripts/package_final_artifacts.py --nonlinear-backends A_rightmul,amulet_secure_R \
  --output-tar $OUT/final_artifacts.tar.gz --output-json $OUT/final_artifacts.json
```

## 20. End-to-end paper-facing validator (FAIL-STOP, run last)

Reads every JSON under `$OUT` and asserts the strict contract across the
collected evidence: linear pad coverage on all 8 families; nonlinear design
paper-facing + executed (not tag-only) with `nonlinear_trusted_calls==0`; TEE
boundary calls 1/1/0; TDX quote (`tee==tdx`, `debug==false`, 3-part JWT,
`report_data==runtime_hash`, runtime hash binds the nonlinear backend, `mr_td`
matches); H800 worker `/health`, TDX boundary client, and remote-decode
exactness present:

```
python scripts/validate_tee_gpu_e2e.py $OUT --expected-mr-td <MRTD> --require all \
  --output-json $OUT/e2e_validation.json \
  || { echo "E2E PAPER-FACING VALIDATION FAILED"; exit 1; }
```

`passed=true` is required for paper-facing results. Any tag-only design,
`nonlinear_trusted_calls>0`, missing pad coverage, simulated/off-TDX quote, or a
legacy `current`/`trusted_shortcut` design fails this gate.

## 21. Two-machine run (H800 GPU worker + Alibaba TDX boundary client)

Keep the GPU worker on the H800 (untrusted) and the boundary client inside the
Alibaba TDX guest (trusted). **Never put passwords in scripts or logs** — use
SSH keys / an agent, and read any secret from the environment or a key file, not
the command line.

```
# --- H800 (untrusted GPU): start the worker for design $D ---
# (SSH in with a key: `ssh h800-new`; do NOT embed a password)
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package --nonlinear-backend $D \
  --folded-package-path $ROOT/qwen7b_folded_full_$D \
  --device cuda --dtype bfloat16 --listen-port $PORT --audit true
curl -fsS http://127.0.0.1:$PORT/health && echo     # confirm worker health (no ss)

# --- Alibaba TDX guest (trusted boundary client) ---
# Expose the H800 worker to the guest over an SSH tunnel keyed by an SSH key:
#   ssh -i ~/.ssh/<key> -N -L 18083:127.0.0.1:18083 h800-new   # no password on CLI
python scripts/generate_tdx_attestation_evidence.py \
  --boundary-backend process --gpu-backend qwen7b_folded_package \
  --nonlinear-backend $D --expected-mr-td <MRTD> \
  --quote-command '<tdx_quote_tool ...{report_data_hex}...{quote_out}>' \
  --attest-command '<aliyun_attest ...{quote_file}...>' \
  --output-dir $OUT/$D/tdx_$D --output-evidence $OUT/$D/attestation_evidence_$D.json
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url http://127.0.0.1:18083 \
  --boundary-backend process --nonlinear-backend $D \
  --attestation-evidence $OUT/$D/attestation_evidence_$D.json \
  --expected-mr-td <MRTD> --max-new-tokens 16 --audit true \
  --output-json $OUT/$D/tdx_attested_decode_$D.json
```

Then re-run §20 over `$OUT` to confirm the full two-machine evidence
(`worker /health` + TDX boundary client + verified quote + remote exactness +
nonlinear execution evidence + zero trusted nonlinear calls) passes.

---
**Limitations carried forward:** `trusted_shortcut` security is not formally
claimed; the dual matrix only proves what its per-design inputs prove (dry-run
inputs are never paper-ready); the runtime hash + TD Quote MUST be regenerated
per design; the masked-remote E9 backend loads a trusted-side tokenizer on the
boundary (heavier than the validated no-LoRA TDX-lite boundary).
