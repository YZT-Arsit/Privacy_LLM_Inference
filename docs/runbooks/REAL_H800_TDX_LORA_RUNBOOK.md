# Real H800 + TDX private-LoRA runbook

Exact commands to run the **private folded-LoRA** evaluation on the real servers
after a restart, with minimal manual intervention. Mirrors the no-LoRA flow and
reuses the validated scripts. Nothing here weakens the threat model: the untrusted
H800 worker only ever loads the **public base folded package + folded-LoRA
operators**; raw LoRA `A`/`B`, optimizer state, training data, labels, mask
secrets, input ids, and recovered logits stay trusted-side (H800 trusted process
for the reference; the TDX guest for the lite/attested boundary).

> **Nonlinear design dimension.** Add `--nonlinear-backend current|trusted_shortcut`
> to the build/verify/probe/decode/E10 commands below (default `current`); the
> folded-LoRA package must match its base package's design (verifiers enforce
> this via `--expected-nonlinear-backend`). For both designs end-to-end (with
> per-design runtime-hash + TD Quote regeneration), follow
> [`REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md`](REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md)
> §11.

All of this was dry-run validated on CPU (`tests/test_lora_folded_e6.py`,
`tests/test_lora_hf_adapter_fixture.py`, `tests/test_e6_lora_real_pipeline.py`).
Replace the example paths with your real ones.

```
MODEL=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
PKGS=/root/autodl-tmp/privacy_llm_packages
BASE=$PKGS/qwen7b_folded_full                 # public base folded package
ART=$PKGS/qwen7b_boundary_artifact_cuda       # trusted boundary embedding artifact
LORA=$PKGS/qwen7b_lora_folded_synth_r4        # folded-LoRA package (output)
URL=http://127.0.0.1:18083
PORT=18083
TM=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

The base package + embedding artifact are the no-LoRA artifacts (build them with
`build_qwen7b_folded_package.py` / `build_qwen7b_embedding_artifact.py` if they do
not exist yet — see the no-LoRA stage docs).

---

## 0. (Optional) one-command pipeline

The whole of steps 1–5 (+ TDX-lite input prep) in a single command. Use
`--plan-only` first to review every resolved sub-command, then drop it to run.

```
python scripts/run_e6_lora_real_h800_pipeline.py \
  --model-path $MODEL --model-name Qwen2.5-7B-Instruct \
  --base-folded-package-path $BASE --embedding-artifact-path $ART \
  --lora-mode synthetic --lora-rank 4 --lora-alpha 8 --target-modules $TM \
  --output-lora-package $LORA \
  --gpu-worker-url $URL --listen-port $PORT --start-worker true \
  --seq-len 128 --max-new-tokens 4 --dtype bfloat16 --device cuda \
  --audit true \
  --output-json outputs/e6_lora_real_h800_pipeline.json \
  --output-md  outputs/e6_lora_real_h800_pipeline.md
```

For a **real HF/PEFT adapter** use `--lora-mode hf --raw-lora-adapter-path
/path/to/adapter --adapter-format hf_peft` (rank/alpha/target_modules are read
from `adapter_config.json`). The manual steps below do the same thing piecewise.

---

## 1. Start the H800 base + LoRA worker

The untrusted GPU worker holds the base folded package **and** the folded-LoRA
package; it merges `W_tilde += a_tilde @ b_tilde` and runs the existing masked
kernels. It never sees raw A/B or masks.

```
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package \
  --folded-package-path $BASE \
  --folded-lora-package-path $LORA \
  --listen-host 0.0.0.0 --listen-port $PORT \
  --device cuda --dtype bfloat16 --audit true
```

Health check (separate shell): `curl -s $URL/health` should report
`folded_package_loaded`, `lora_enabled=true`, `folded_lora_loaded=true`,
`worker_has_raw_lora=false`. (Build the LoRA package in step 3 first if `$LORA`
does not exist yet — or start the worker after step 3.)

## 2. H800 LoRA reference decode (trusted, local)

Trusted base + raw LoRA vs base-folded + folded-LoRA, on the real model. Echoes
the trusted `input_ids` and the reference token ids for the TDX replay.

```
python scripts/run_qwen7b_lora_folded_local_probe.py \
  --model-path $MODEL --model-name Qwen2.5-7B-Instruct \
  --base-folded-package-path $BASE \
  --target-modules $TM --rank 4 --alpha 8 \
  --max-new-tokens 4 --seq-len 128 --dtype bfloat16 --device cuda \
  --output-json outputs/qwen7b_lora_folded_local_probe.json
```

Expect `allclose=True`, `top1_match=True`, `tokens_exact_match=True`,
`worker_has_raw_lora=False`, `leaked_secret_fields=[]`.

## 3. Build / sync the folded LoRA package

Synthetic adapter (dry-run-equivalent on real masks):

```
python scripts/build_qwen7b_lora_folded_package.py \
  --model-path $MODEL --model-name Qwen2.5-7B-Instruct \
  --base-folded-package-path $BASE \
  --target-modules $TM --rank 4 --alpha 8 \
  --output-dir $LORA --output-json outputs/qwen7b_lora_folded_build.json
```

Real HF/PEFT adapter (`adapter_model.safetensors` + `adapter_config.json`):

```
python scripts/build_qwen7b_lora_folded_package.py \
  --model-path $MODEL --base-folded-package-path $BASE \
  --raw-lora-adapter-path /path/to/adapter --adapter-format hf_peft \
  --target-modules $TM \
  --output-dir $LORA --output-json outputs/qwen7b_lora_folded_build.json
```

Verify (no raw A/B, no optimizer/training/mask secrets, target coverage,
base-manifest compatibility):

```
python scripts/verify_qwen7b_lora_folded_package.py \
  --lora-folded-package-path $LORA --base-folded-package-path $BASE \
  --output-json outputs/qwen7b_lora_folded_verify.json
```

Confirm the LoRA actually changes behaviour vs a no-LoRA decode (guards against a
silently-unmerged adapter):

```
python scripts/validate_lora_effect.py \
  --no-lora-json outputs/qwen7b_folded_full_decode_probe.json \
  --lora-json    outputs/qwen7b_lora_folded_local_probe.json \
  --output-json  outputs/validate_lora_effect.json --require-effect true
```

(Then start/restart the worker from step 1 so it picks up `$LORA`.) Run the H800
**remote** LoRA decode against the worker:

```
python scripts/run_qwen7b_lora_folded_remote_decode_probe.py \
  --gpu-worker-url $URL --embedding-path $ART \
  --input-ids-file outputs/qwen7b_lora_folded_local_probe.json \
  --expected-token-ids-file outputs/qwen7b_lora_folded_local_probe.json \
  --max-new-tokens 4 --seq-len 128 --dtype bfloat16 --device cpu --audit true \
  --output-json outputs/qwen7b_lora_folded_remote_decode_probe.json
```

Expect `lora_enabled=True`, `folded_lora_loaded=True`, `tokens_exact_match=True`,
`worker_has_raw_lora=False`.

## 4. Run TDX-lite LoRA decode

Prepare the TDX replay inputs + a ready-to-run command file from the H800 remote
reference:

```
python scripts/prepare_tdx_lora_lite_inputs.py \
  --reference-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --embedding-path $ART --lora-folded-package-path $LORA \
  --gpu-worker-url $URL --output-dir outputs
```

This writes `outputs/tdx_lora_input_ids.json`,
`outputs/tdx_lora_expected_tokens.json`, `outputs/tdx_lora_replay.json` (+ artifact
hashes), and `outputs/run_tdx_lora_lite_decode.sh`. Copy ONLY the embedding
artifact + those input/expected JSONs to the TDX guest (NOT the full model, NOT
the base folded package, NOT the raw LoRA). On the TDX guest:

```
bash outputs/run_tdx_lora_lite_decode.sh
```

It runs the boundary client with `--skip-reference true`, `--embedding-path`,
`--input-ids-file`, `--expected-token-ids`, and **no** model / base-package / raw-
LoRA paths. Expect `tokens_exact_match=True` against the H800 reference.

## 5. Regenerate the TDX runtime hash

The private LoRA adds NO new boundary file (folding/merging runs on the worker),
but always re-measure after any boundary code change. First confirm the
measurement still covers the boundary import closure:

```
python scripts/check_tdx_measurement_coverage.py --verbose   # must exit 0
```

Then compute the runtime hash to bind into the TD Quote `report_data` (use the
SAME `--boundary-backend` / `--gpu-backend` / `--expected-mr-td` you will use at
verification time):

```
python scripts/write_tee_boundary_runtime_hash.py \
  --boundary-backend process --gpu-backend qwen7b \
  --expected-mr-td <MR_TD> \
  --output outputs/runtime_hash.txt \
  --manifest outputs/runtime_manifest.json
```

## 6. Generate the quote + Alibaba attestation (on the TDX VM)

```
python scripts/generate_tdx_attestation_evidence.py \
  --report-data $(cat outputs/runtime_hash.txt) \
  --output outputs/attestation_evidence.json
```

This drives the TDX guest device / Alibaba attestation client to produce the TD
Quote (with `report_data` = the runtime hash) and the signed JWT, and assembles
`attestation_evidence.json` (`tee`, `mr_td`, `report_data`, `jwt`,
`td_attributes.debug`).

## 7. Run the TDX-attested LoRA decode

Same lite decode as step 4 but with the evidence + expected mr_td, so the binding
is verified. The folded-LoRA metadata is covered by the attested run.

```
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --embedding-path $ART --skip-reference true \
  --input-ids-file outputs/tdx_lora_input_ids.json \
  --expected-token-ids $(python -c "import json;print(','.join(map(str,json.load(open('outputs/tdx_lora_expected_tokens.json'))['expected_token_ids'])))") \
  --seq-len 128 --max-new-tokens 4 --dtype bfloat16 --device cpu --audit true \
  --boundary-backend process --gpu-backend qwen7b --expected-mr-td <MR_TD> \
  --attestation-evidence outputs/attestation_evidence.json \
  --output-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json
```

Expect `tokens_exact_match=True`, `boundary_attested=True`,
`runtime_hash_bound=True`, `binding_mismatch_reason=null`. If
`runtime_hash_bound=False`, the boundary code/metadata changed since binding —
redo steps 5–7.

Then consolidate the four E8 tables:

```
python scripts/run_e8_lora_final_report.py \
  --local-json   outputs/qwen7b_lora_folded_local_probe.json \
  --remote-json  outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --attested-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json \
  --lora-build-json  outputs/qwen7b_lora_folded_build.json \
  --lora-verify-json outputs/qwen7b_lora_folded_verify.json \
  --base-decode-json outputs/qwen7b_folded_full_decode_probe.json \
  --training-json    outputs/lora_private_training_probe.json \
  --output-json outputs/e8_lora_final_report.json \
  --output-md   outputs/e8_lora_final_report.md
```

## 8. Package artifacts

```
python scripts/package_final_artifacts.py \
  --outputs-dir outputs \
  --tee-artifacts-dir /root/privacy_llm_tee_artifacts \
  --base-folded-package-path $BASE \
  --embedding-artifact-path $ART \
  --lora-folded-package-path $LORA \
  --extra-file outputs/runtime_hash.txt \
  --output-tar /root/privacy_llm_final_artifacts.tar.gz
```

Missing optional inputs are recorded in the tar's `MANIFEST.json` and skipped.

## 9. Cleanup

```
# stop the worker
pkill -f "run_tee_gpu_protocol_demo.py --mode gpu_worker_server" || true
# (optional) drop intermediate pipeline JSONs
rm -rf outputs/e6_pipeline
# (optional) free GPU package copies you no longer need
# rm -rf $LORA            # folded-LoRA package (rebuildable from the adapter)
nvidia-smi               # confirm no lingering processes hold GPU memory
```

Do **not** delete the base folded package / embedding artifact unless you intend
to rebuild them — they are the expensive one-time trusted-setup outputs.

## Full-eval integration (security, benchmark utility, claims)

These steps wire the LoRA run into the paper-facing evaluation. See the master
[`REAL_H800_TDX_FULL_EVAL_RUNBOOK.md`](REAL_H800_TDX_FULL_EVAL_RUNBOOK.md) for the
shared no-LoRA steps and [`REAL_PUBLIC_BENCHMARK_RUNBOOK.md`](REAL_PUBLIC_BENCHMARK_RUNBOOK.md)
for dataset prep.

```
# (a) record + scan a LoRA TDX transcript for leaks
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --embedding-path $ART --skip-reference true \
  --input-ids-file outputs/tdx_lora_input_ids.json \
  --expected-token-ids "$(python -c "import json;print(','.join(map(str,json.load(open('outputs/tdx_lora_expected_tokens.json'))['expected_token_ids'])))")" \
  --seq-len 128 --max-new-tokens 4 --dtype bfloat16 --device cpu --audit true \
  --record-transcript outputs/transcript_lora_tdx.jsonl \
  --output-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json
python scripts/scan_security_transcript.py \
  --transcript-jsonl outputs/transcript_lora_tdx.jsonl \
  --output-json outputs/security_transcript_scan_lora.json --fail-on-leak true

# (b) confirm the LoRA actually changes behaviour vs no-LoRA
python scripts/validate_lora_effect.py \
  --no-lora-json outputs/qwen7b_folded_remote_decode.json \
  --lora-json    outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --output-json  outputs/validate_lora_effect.json --require-effect true

# (c) LoRA utility preservation on a real task (E10) -- see benchmark runbook
python scripts/run_e10_lora_utility_benchmark.py \
  --dataset-name sst2 --task-type classification --metric-name accuracy \
  --base-json outputs/e9_sst2_base.json \
  --plaintext-lora-json outputs/e9_sst2_plaintext_lora.json \
  --folded-lora-json outputs/e9_sst2_folded_lora_remote.json \
  --tdx-attested-folded-lora-json outputs/e9_sst2_tdx_attested_lora.json \
  --lora-verify-json outputs/qwen7b_lora_folded_verify.json \
  --no-lora-decode-json outputs/qwen7b_folded_remote_decode.json \
  --lora-decode-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --output-json outputs/e10_lora_utility.json

# (d) security negative tests (all 14 must be caught)
python scripts/run_security_negative_tests.py \
  --output-json outputs/security_negative_tests.json

# (e) claim validation -- a synthetic-LoRA dry-run can ONLY support
#     folded_lora_dry_run_validated; a real attested LoRA run supports
#     folded_lora_tdx_attested_validated / real_lora_tdx_attested.
python scripts/validate_paper_claims.py \
  --result-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json \
  --result-json outputs/e10_lora_utility.json \
  --result-json outputs/security_negative_tests.json \
  --required-claims folded_lora_h800_real_validated \
  --output-json outputs/paper_claim_validation_lora.json
```

For a real HF adapter, build the package with `--raw-lora-adapter-path <dir>
--adapter-format hf_peft` (step 3) so claims about real-adapter utility are not
flagged as synthetic by the validator.
