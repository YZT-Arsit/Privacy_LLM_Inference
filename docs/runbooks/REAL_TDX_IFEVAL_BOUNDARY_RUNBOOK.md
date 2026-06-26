# Real TDX boundary-client IFEval runbook (folded_remote)

The paper topology for the IFEval generation benchmark is:

```
  trusted TDX guest (boundary client)            untrusted H800 (GPU worker)
  ----------------------------------             ---------------------------
  tokenizer / generation config                  qwen7b_folded_package
  embedding artifact (LiteBoundary)   --masked-->  folded prefill / decode
  mask / recover / sample / EOS         embeddings  (tee_used_on_gpu=false,
  per-step obfuscation schedule       <--masked--   resident_folded_weights=true)
                                        logits
  NEVER loads the full 7B weights
```

**Do NOT run the final benchmark on the H800 itself.** The H800 keeps ONLY the
worker; the benchmark is launched from the TDX guest as a `folded_remote`
boundary client (`--tdx-boundary-client`), which loads only the tokenizer/config
+ the embedding artifact — never the full 7B model.

## Hosts

```
TDX guest:  root@39.96.90.40   repo /root/Privacy_LLM_Inference   conda env tdx310
H800:       ssh alias from TDX = h800-new   worker http://127.0.0.1:18082 on H800
```

## 0. Verify the H800 worker is up (from the TDX guest)

`ss` is NOT installed on H800 — use curl. The worker must already be the resident
folded package worker.

```
ssh h800-new "curl -s http://127.0.0.1:18082/health"
# expect: gpu_backend=qwen7b_folded_package, tee_used_on_gpu=false,
#         resident_folded_weights=true
```

If the benchmark runs on the TDX guest, point `--gpu-worker-url` at the H800
worker reachable from the guest (e.g. an SSH tunnel: from the guest run
`ssh -N -L 18082:127.0.0.1:18082 h800-new &`, then use
`http://127.0.0.1:18082`).

## 1. Paths (TDX guest)

```
REPO=/root/Privacy_LLM_Inference
export PYTHONPATH=$REPO/src:$PYTHONPATH
MODEL=/root/.../Qwen2___5-7B-Instruct          # tokenizer + generation_config only
ART=/root/.../qwen7b_boundary_artifact_current # LiteBoundary embedding artifact
IN=/root/.../ifeval_prompts.jsonl
URL=http://127.0.0.1:18082                      # H800 worker (tunnelled if needed)
OUT=outputs/ifeval ; mkdir -p $OUT
```

## 2. TDX smoke (2 examples) — proves the boundary client reaches the worker

```
python scripts/run_ifeval_generation.py \
  --input-jsonl $IN --backend folded_remote \
  --tdx-boundary-client --trusted-runtime real_tdx \
  --model-path $MODEL --gpu-worker-url $URL --embedding-path $ART \
  --seq-len 1024 --max-new-tokens 256 --dtype bfloat16 --device cuda --audit \
  --nonlinear-backend current --use-chat-template \
  --align-generation-config --repetition-penalty 1.05 \
  --precompute-obfuscation-schedule --schedule-max-steps 1024 \
  --schedule-proof-mode online_deterministic --schedule-precompute-device cpu \
  --progress --progress-every 1 \
  --max-examples 2 \
  --attestation-evidence-json /root/.../tdx_attestation_evidence.json \
  --output-response-jsonl $OUT/ifeval_tdx_smoke2_responses.jsonl \
  --output-report-json    $OUT/ifeval_tdx_smoke2_generation.json
```

Within seconds you should see, flushed live (works under `PYTHONUNBUFFERED=1`):

```
[ifeval] example 1/2 id=... phase=schedule_start elapsed=...s
[ifeval] example 1/2 id=... phase=generate_start elapsed=...s
[ifeval] example 1/2 id=... phase=done tokens=... finish_reason=... elapsed=...s avg_s_per_example=... eta=...s
```

and `ifeval_tdx_smoke2_responses.jsonl` gains one line per example **immediately**
(no more 0-byte logs). The H800 worker logs should show `/init` `/prefill`
`/decode` requests arriving.

## 3. TDX full run (541 examples)

Identical, drop `--max-examples 2`:

```
python scripts/run_ifeval_generation.py \
  --input-jsonl $IN --backend folded_remote \
  --tdx-boundary-client --trusted-runtime real_tdx \
  --model-path $MODEL --gpu-worker-url $URL --embedding-path $ART \
  --seq-len 1024 --max-new-tokens 256 --dtype bfloat16 --device cuda --audit \
  --nonlinear-backend current --use-chat-template \
  --align-generation-config --repetition-penalty 1.05 \
  --precompute-obfuscation-schedule --schedule-max-steps 1024 \
  --schedule-proof-mode online_deterministic --schedule-precompute-device cpu \
  --progress --progress-every 1 \
  --attestation-evidence-json /root/.../tdx_attestation_evidence.json \
  --deployment-truth-json     /root/.../deployment_truth.json \
  --tdx-measurement-log       /root/.../tdx_measurement.log \
  --output-response-jsonl $OUT/ifeval_tdx_541_responses.jsonl \
  --output-report-json    $OUT/ifeval_tdx_541_generation.json
```

The per-example progress + ETA tells you the run is alive; if it dies, the
streamed `..._responses.jsonl` shows exactly how far it got (resume / progress).

## 4. Why no CUDA secret-tensor precompute (the old hang)

The previous hang was `--precompute-obfuscation-schedule` materializing
541×1024 per-step **secret tensors** before any GPU call. Those tensors are only
ever *consumed for freshness accounting* — the decode path never reads them — so
the default is now `--schedule-proof-mode online_deterministic`: build only the
per-step slot metadata (fast, no torch tensors), derive the per-step obfuscation
domain deterministically at consume time, and **prove full coverage** instead of
pre-materializing secrets. The old command still works; it just takes this fast
path now. Heavy secret-tensor precompute is opt-in only:
`--schedule-proof-mode precompute_secret_tensors --enable-secret-tensor-precompute
--allow-secret-persist` (and `--schedule-precompute-device` stays `cpu`; `cuda`
is refused). Paper claims need the **coverage + non-leakage** audit, not secret
tensors on the GPU.

## 5. Report fields that support the proof

Security / topology (TDX boundary client):

- `tee_mode=real_tdx`, `trusted_runtime=tdx_guest`, `tdx_boundary_client=true`
- `full_model_weights_loaded_in_trusted_runtime=false` (TDX never holds the 7B)
- `h800_worker_url`, `h800_worker_tee_used_on_gpu=false`
- `audit_passed=true`, `worker_has_mask_secrets=false`, `leaked_secret_fields=[]`,
  `gpu_visible_plaintext_fields=[]`
- `tdx_claim_ready` — **true only** on a real (non-dry) client run with
  attestation evidence attached and the worker NOT in the TEE; else false.

Schedule full-coverage proof (no GPU secrets):

- `schedule_proof_mode=online_deterministic`
- `schedule_slots_required_total == schedule_slots_consumed_total`
- `schedule_full_coverage_verified=true` (every example:
  `slots_consumed == generated_tokens`)
- `schedule_secret_leaked_to_gpu=false`, `schedule_materialized_on_gpu=false`,
  `online_remask_still_performed=true`, `schedule_used_for_metadata_only=true`
- `schedule_coverage_per_example[*]` — per-example
  `{example_id, schedule_seed_commitment, generated_tokens, slots_required,
  slots_consumed, slots_consumed_matches_generated_tokens}`

Progress / streaming:

- `progress_streaming_enabled=true`, `responses_streamed=true`

Quality alignment (already shipped): `prompt_format=chat`,
`generation_processors_applied=true`, `repetition_penalty=1.05`,
`stop_on_eos=true`, `finish_reason_per_example`, `generated_tokens_per_example`.
