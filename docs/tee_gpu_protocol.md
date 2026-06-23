# TEE ↔ Untrusted-GPU-Worker Protocol (Stage 8.5)

This document describes the message protocol + reference orchestration added in
`src/pllo/protocol/`, which drives the thin trusted boundary runtime
(`src/pllo/tee/`, Stage 8.3) against an **untrusted GPU worker** that runs the
masked model. It builds directly on the boundary design in
[`tee_boundary_design.md`](tee_boundary_design.md); the threat model and mask
math are unchanged.

> **No semantic, cryptographic, or formal security is claimed.** Same caveats as
> Stage 8.3: orthogonal signed-permutation residual mask + positive vocab
> scaling; attention structure, sequence length, batch size, and step count
> remain visible to the untrusted side. This stage adds the *wire protocol* and
> an *audit* of exactly what crosses the trust boundary.

## 1. Domains

| Domain | Process | Holds / does |
|---|---|---|
| **Trusted boundary** | orchestrator + boundary runtime (optionally a separate `spawn` process, the `process` backend) | raw prompt, tokenization, `input_ids`, mask handles, offline weight fold, embedding+masking, logit recovery, greedy selection, generated token ids, remasking each new token |
| **Untrusted GPU worker** | a separate `spawn` process | receives only masked embeddings + public metadata + the folded LM head; runs the masked decoder + masked KV cache; returns only masked logits |

The model — Qwen, decoder blocks, attention, MLP, KV cache, LM head — is **never**
placed inside the TEE. `tee_used_on_gpu` is always `False`.

## 2. Messages (`tee_gpu_messages.py`)

GPU-channel messages (the only objects allowed to cross to the untrusted side):

- `BoundaryInitRequest` → public hyper-params + the folded head
  `W_tilde = N⁻¹ · W · M_vocab` (a transformed artifact, not a recoverable
  secret); `BoundaryInitResponse` (carries `tee_used_on_gpu=False`).
- `MaskedPrefillRequest` (masked prompt embeddings `[B,T,H]` + public positions)
  → `MaskedPrefillResponse` (masked logits `[B,V]` + public KV length).
- `MaskedDecodeRequest` (one masked token embedding `[B,1,H]` + public position)
  → `MaskedDecodeResponse` (masked logits + public KV length).

Trusted-only (never on the GPU channel): `RecoveredTokenResponse` — the recovered
next token returned to the client.

`ProtocolTrace` records the *exact* objects sent to / received from the worker,
plus boundary/GPU call counts and byte accounting — it is the audit subject.

## 3. Decode flow

1. **Offline (trusted):** derive handles, fold `W_tilde`, hand the worker only
   `W_tilde` + public metadata via `BoundaryInitRequest`.
2. **Prefill (trusted→GPU→trusted):** trusted `embed_and_mask(input_ids)` →
   `MaskedPrefillRequest` → worker returns masked logits → trusted
   `recover_logits` → `sample` (greedy).
3. **Decode loop:** trusted embeds+**remasks** the new token →
   `MaskedDecodeRequest` → worker returns masked logits → trusted recover +
   sample. The worker stores only a **masked** KV cache; token ids never cross.

With the identity-decoder `mock` backend the recovered greedy tokens match the
trusted plaintext reference exactly, so the round trip is verifiable on CPU with
numpy only. The `qwen7b` backend plugs the same protocol into the real masked
pipeline on the GPU server (CUDA + checkpoint); locally it runs as an init-only
probe.

## 4. Security audit (`security_audit.py`)

Run against the recorded `ProtocolTrace` (structural **and** value checks):

- `assert_no_gpu_visible_plaintext` — no raw prompt, `input_ids`, generated token
  ids, recovered logits, or tokenizer output crossed (by field name *and* by
  comparing the actual plaintext values against every GPU message).
- `assert_no_mask_secret_leak` — no perm / inverse / signs / scale / seed /
  `MaskHandles` crossed (the folded head is allowed — it is not equal to any raw
  secret array).
- `assert_wrong_mask_recovery_fails` — the correct mask recovers the plaintext
  within tolerance; a wrong mask flips the argmax token and diverges by a
  non-trivial fraction of the logit scale.

Each returns a findings list (empty == clean) and raises by default.

## 5. Demo & reproduction

```bash
# end-to-end (numpy only): two processes, full audit
python scripts/run_tee_gpu_protocol_demo.py --mode local_two_process \
    --boundary-backend process --gpu-backend mock --max-new-tokens 8 \
    --audit true --output-json outputs/tee_gpu_protocol.json \
    --output-md outputs/tee_gpu_protocol.md

# qwen7b: init-only protocol/audit probe locally (full decode on the GPU server)
python scripts/run_tee_gpu_protocol_demo.py --gpu-backend qwen7b \
    --boundary-backend process --audit true

# tests
python -m pytest tests/test_tee_gpu_protocol_security.py -q
```

Reported fields: `gpu_visible_plaintext_fields`, `leaked_secret_fields`,
`boundary_calls`, `trusted_bytes`, `gpu_bytes`, `max_new_tokens`,
`boundary_backend`, `gpu_backend`, `tee_used_on_gpu`, `audit_passed`.

## 6. Boundary attestation + runtime-hash binding (`attestation.py`)

The trusted boundary is meant to run inside an Intel TDX guest on Alibaba Cloud.
Attestation has been **validated end-to-end** on the VM: a TD Quote is generated,
the Alibaba Cloud Attestation API returns a signed 3-part JWT, `tee == tdx`,
`tdx.td_attributes.debug == false`, `mr_td` is reported, and the **runtime-hash
binding** verifies (`expected_runtime_hash == report_data`).

`attestation.py` reproduces the verification (it does **not** generate a quote —
that is the deployment's attestation client):

- `compute_runtime_hash(components)` — deterministic 64-byte SHA-512 over the
  boundary's *public* identity/config (component name + version + boundary
  config). No secrets, no prompt. This is what the boundary binds into the
  quote's 64-byte `report_data`.
- `verify_evidence(evidence, runtime_hash, expected_mr_td=...)` — checks
  `tee==tdx`, `debug==false`, a 3-part signed JWT is present, `report_data`
  equals the runtime hash, and (optionally) `mr_td` matches. Returns
  `AttestationEvidence`; `verified` is the conjunction.
- `attest_boundary(...)` — with `--attestation-evidence <json>` (produced on the
  VM) it verifies the real binding; on the TDX guest without evidence it reports
  `tdx` available; off-TDX it degrades to `simulated` while still exposing the
  runtime hash the boundary *would* bind, so deployment is testable in CI.

The demo adds an `attestation` block + `boundary_tee_type`, `boundary_attested`,
`runtime_hash`, `runtime_hash_bound`, `mr_td`. `tee_used_on_gpu` stays `False`
regardless — attestation covers the *boundary*, never the GPU worker.

> The JWT signature / certificate chain is verified by the remote attestation
> service; this module verifies the tee/debug/binding/`mr_td` *claims*. We do not
> re-verify the signature locally.
