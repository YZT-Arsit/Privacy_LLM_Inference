# Gate 1 — Trusted Training Protocol: local implementation + contract tests

**run_id:** `real_qwen_tdx_20260710T065808Z` · **status: PASS (CPU contract, service-validation only)** · not committed.

> SERVICE VALIDATION, NOT A REAL EXPERIMENT. All Gate-1 tests run on CPU float64 with
> synthetic protocol tensors, and the service runs in `cpu_contract` mode
> (`tee_type="cpu_contract_test"`, `guest_verified=false`, `attestation_verified=false`).
> This is explicitly **not** a real TDX result and **not** a real Qwen result — those are
> Gate 2 (real guest) and Gate 3 (real model).

## What was built (new files)
- `src/pllo/experiments/real_tdx_training_service.py` — trusted service: `init_session`,
  `/train/logits_loss`, `/train/packed_update`, in-guest CE + unmask + real AdamW + remask,
  fail-closed validation, `serve_http()` (deploys unchanged in the guest).
- `src/pllo/experiments/real_tdx_training_client.py` — GPU-side client: two boundaries,
  nonce/manifest/run_id/step_id state machine, in-process + HTTP transport (dtype-agnostic
  torch.save+base64 tensor codec, handles bf16).
- `src/pllo/experiments/real_tdx_attestation.py` — runtime-hash binding (reuses
  `pllo.protocol.attestation.compute_runtime_hash`), fail-closed `real_tdx` vs marked
  `cpu_contract` mode.
- `scripts/start_tdx_trusted_service.sh` — Gate-2 entrypoint (refuses to start without
  `/dev/tdx_guest`; `require_real_tdx=1`, no CPU fallback).
- Tests: `tests/test_real_qwen_protocol_contracts.py`, `test_real_tdx_fail_closed.py`,
  `test_packed_manifest_validation.py`.

## Evidence — 18/18 tests pass
Log: `results/real_qwen_tdx_audit/logs/gate1_protocol_contract_tests.log`.

**Protocol correctness (contract):**
- `test_packed_update_matches_plaintext_adamw_trajectory` — 6-step, 2-layer (q_proj+gate_proj)
  loop through the service reproduces a PLAINTEXT AdamW trajectory EXACTLY: per-step plaintext
  `A`/`B`, `DeltaW=AB`, and Adam `m`/`v` all match to `<1e-9`. In-TDX-side unmask + AdamW +
  remask is exactly plaintext AdamW on recovered gradients.
- `test_ce_loss_and_logit_gradient_correct` — service CE == plaintext CE (`<1e-9`); returned
  masked logit gradient un-masks to the true `dlogits` (`<1e-9`).
- `test_recovered_masked_adapters_match_plaintext` — updated MASKED adapters returned to the
  GPU un-mask to the trusted plaintext adapters (`<1e-9`).
- `test_http_transport_round_trip` — the same service+client work over localhost HTTP
  (Gate-2 transport validated end-to-end for `init`/`logits_loss`/`packed_update`).

**Fail-closed (no silent fallback, no dummy AdamW):**
- bad nonce → reject; replayed/stale nonce → reject (challenge-response, service rotates the
  nonce each response); run_id mismatch → reject; config_digest mismatch → reject.
- `require_real_tdx=1` + no `/dev/tdx_guest` (CPU host) → `AttestationFailClosed` at init.
- empty/short packed manifest → reject AND plaintext adapters remain byte-identical (no
  partial/dummy update).
- client surfaces service `FailClosed` as `TrainingProtocolError` (GPU side also fail-closed).

**Manifest / shape / dtype validation:**
- packed manifest must cover EXACTLY the registered layers (missing/extra layer → reject);
- per-layer gradA/gradB shape mismatch → reject; dtype-in-manifest mismatch → reject;
- masked_logits shape mismatch → reject.

## Interface fields present on every request
`run_id`, `step_id`, `nonce` (rotating), `manifest` (per-tensor shape+dtype), `config_digest`.
Init also carries `lora_manifest`, `vocab_size`. Responses carry `verification` flags
(`tee_type`, `guest_verified`, `attestation_verified`, `runtime_hash_bound`,
`trusted_loss_executed`, `trusted_adamw_executed`, `packed_gradient_unmask_executed`,
`adapter_remask_executed`, `plaintext_adapter_exported=false`, `plaintext_label_exported=false`)
and `attestation` (the binding dict).

## Masking relations verified (fp64 exact)
- adapters: `A_t = N_in^-1 A R`, `B_t = R^-1 B N_out`
- masked LoRA grads (GPU, masked-domain only, upstream `U_t = U N_out^-T`):
  `gA_t = N_in^T gradA R^-T`, `gB_t = R^T gradB N_out^-T`
- recovery (in-TDX): `gradA = N_in^-T gA_t R^T`, `gradB = R^-T gB_t N_out^T`
- logits: `masked_logits = logits N_head` → recover `logits = masked_logits N_head^-1`;
  logit grad returned as `dlogits M_head`.

## Blocking issues / not-yet-done
- **Gate 2 (deploy in real guest) NOT run** — needs the guest reachable and the service
  started with `require_real_tdx=1`, plus real quote binding of the runtime hash. GPU→TDX
  tunnel to be established.
- **Gate 3 (real Qwen2.5-0.5B single step) NOT run** — blocked on (a) 0.5B download (GPU SSH
  gateway was throttling) and (b) wiring the real Qwen masked forward/backward to produce the
  masked LoRA grads that this protocol consumes.
- The contract uses synthetic tensors; the masked forward/backward here is the analytic
  relation, exact in fp64. bf16 numerical behavior on real activations is a Gate-3 question.

## Honest labels
`uses_real_gpu=false` (Gate 1 is CPU), `uses_real_tee=false` (cpu_contract mode). No CPU
simulation is presented as TDX; the mode is stamped on every response. Nothing committed.
