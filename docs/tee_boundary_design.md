# TEE Boundary Design (Stage 8.3)

This document describes the trusted-execution-environment (TEE) **boundary
runtime** added in `src/pllo/tee/`. The runtime is deliberately *thin*: only
the privacy-critical boundary operations run inside the TEE. The transformer,
the masked decoder blocks, the masked KV cache, the masked LM head, and every
large matrix multiplication stay **outside** the TEE on the untrusted host/GPU.

It depends on **numpy only** — no torch, no transformers, no GPU — so it can be
compiled into a small trusted image (e.g. an Intel TDX guest) and unit-tested
without any ML stack.

> **No semantic, cryptographic, or formal security is claimed.** The masks are
> orthogonal (signed-permutation residual mask) + positive vocab scaling, which
> is weaker than dense masking. Attention scores/probabilities and the sequence
> length remain visible to the untrusted side. This is an architecture +
> leakage-accounting prototype, not a hardened system.

## 1. Threat model

- **Trusted (inside TEE):** the client's `input_ids`, the embedding table, the
  mask seeds/handles, the recovered plaintext logits, and the sampled token.
  The adversary cannot read TEE memory (assumed; on TDX this rests on the
  hardware + a verified attestation, see §5).
- **Untrusted (host + GPU):** can observe everything it processes — masked
  embeddings, masked hidden states, masked KV cache, the folded LM head, and
  masked logits. It can also observe **metadata it is not meant to hide**:
  sequence length, batch size, number of decode steps, and attention score
  structure.
- **Goal:** the untrusted side performs the heavy compute without ever seeing
  plaintext embeddings, plaintext logits, plaintext token ids, or the mask
  secrets. Breaking the mask (wrong seed) must destroy the result.
- **Out of scope:** side channels (timing/power/cache), traffic analysis on
  sequence length, semantic inference from preserved geometry, a malicious TEE
  image, and a compromised attestation chain.

## 2. Trusted / untrusted split

| Step | Side | Operation |
|---|---|---|
| 1 | **TEE** | receive plaintext `input_ids` from the client |
| 2 | **TEE** | manage mask seeds / mask handles (`setup_masks`) |
| 3 | **TEE** | trusted embedding lookup + masking boundary (`embed_and_mask`) → release **masked embeddings only** |
| 4 | host/GPU | masked decoder blocks, masked KV cache, masked LM head, all GEMMs |
| 5 | **TEE** | recover masked logits (`recover_logits`) |
| 6 | **TEE** | trusted greedy sampling / argmax (`sample`) → return **next token only** |

The masking algebra (mirrors Stage 8.2): an orthogonal **signed-permutation**
residual mask `N` (`x̃ = x[perm]·signs`) and a vocab logit mask
`M = P·D` (permutation + positive diagonal scale). With orthogonal `N`,
`rmsnorm_core` commutes with the mask and `M`/`N` fold offline into the LM head
and decoder weights, so the GPU computes on masked tensors throughout and the
TEE only does cheap boundary work:

```
TEE:  x̃ = (E[ids]) @ N                      # embed_and_mask
GPU:  L̃ = MaskedDecoder(x̃) @ W̃             # W̃ = N⁻¹ · W · M  (folded offline)
TEE:  L = L̃ @ M⁻¹ ;  token = argmax(L)      # recover_logits + sample
```

`MaskHandles` (the mask secrets) never leave the TEE. In the **process
backend** this is enforced concretely: handles live in the worker process and
`setup_masks()` returns `None` to the untrusted parent.

**What the TEE stores.** Only (a) the mask secrets/handles and (b) the
embedding table, which is intrinsic to the embedding boundary (TEE step 3:
`input_ids → embedding-like vectors → masked hidden states`). The TEE stores
**no decoder/attention/MLP/KV/LM-head weights** and runs **no** decoder,
attention, MLP, or LM-head GEMM — those are the untrusted "full model weights"
and computation. This is asserted by tests: an AST scan rejects any
decoder/attention/MLP/transformer/LM-head/model class or function in
`src/pllo/tee/*`, and a runtime-attribute scan confirms the live runtime holds
only `config`, dtype, mask handles, and the embedding table. The untrusted GPU
computation is represented purely by **synthetic `masked_logits` generated
outside the runtime** (microbench) or by a folded head run **outside** the TEE
(demo); the runtime never executes model layers.

## 3. Why the full transformer is NOT inside the TEE

- **Performance:** TEEs have limited memory and no GPU; running multi-billion-
  parameter attention/MLP/LM-head GEMMs inside a TDX guest would be orders of
  magnitude slower and often infeasible. The boundary work here is `O(B·T·H)`
  gather + sign-flip and `O(B·V)` permutation/scale — tiny next to the decoder's
  `O(B·T·H²)` + `O(B·T·H·V)`.
- **TCB size:** keeping torch/transformers/CUDA out of the trusted image keeps
  the trusted computing base small and auditable (numpy-only, a few hundred
  lines). A smaller TCB is easier to attest and reason about.
- **Separation of concerns:** privacy comes from the *boundary* (what crosses
  the trust line), not from hiding the arithmetic. The masked decoder can run on
  commodity untrusted GPUs unchanged.

The microbenchmark and `scripts/run_tee_end_to_end_demo.py` use a **stub**
"decoder" (identity hidden state) precisely because the real decoder is, by
design, untrusted and out of this module.

## 4. Backends

- **simulated** (`simulated_runtime.py`): in-process numpy reference; the
  ground truth all other backends must match.
- **process** (`process_runtime.py`): the simulated runtime running in a
  separate Python process (multiprocessing `spawn` + `Pipe`). Models the real
  deployment where the trusted runtime is an isolated domain driven by message
  passing, and measures the IPC cost of the boundary. Results are numerically
  identical to the simulated backend (same code, same seed-derived masks).

## 5. Connection to Alibaba Cloud Intel TDX

This runtime is the software that would execute inside a confidential VM. The
TDX platform was validated separately on Alibaba Cloud:

- `/dev/tdx_guest` exists in the guest;
- **TDREPORT** generation succeeds;
- TD attributes include **NO_DEBUG** and **SEPT_VE_DISABLE**;
- remote **Quote** generation / verification is **pending vendor-specific
  platform-side QGS (Quote Generation Service) / evidence support**.

`attest()` performs a best-effort probe (`probe_tdx`): it detects the guest
device and reports the documented attributes, but it **does not generate or
verify a Quote**. Its `AttestationReport` therefore carries:

- `tee_type = "intel_tdx"` iff `/dev/tdx_guest` is present (else `"simulated"`);
- `tdreport_available` gated on device presence (capability, not a generation);
- `quote_available = False`, `quote_status = "pending_vendor_qgs_evidence"`.

On developer machines (no TDX device) the report cleanly degrades to
`tee_type = "simulated"`, so the code path and tests run anywhere.

## 6. How to interpret these results in the paper

State plainly what is and is not demonstrated.

**Supported claims**
- A working **trusted/untrusted boundary** runtime with a small numpy-only TCB
  (no torch/transformers/GPU in the TEE), exercised by an end-to-end demo and a
  microbenchmark.
- The trusted boundary cost is small and well-characterised: `setup`,
  `embed_and_mask`, `recover_logits`, `sample` latencies plus the bytes crossing
  the trust line, for realistic `hidden_size`/`vocab_size`.
- **Correctness** of the masking boundary: deterministic under a fixed seed;
  recovered greedy tokens match the plaintext reference; a **wrong mask
  numerically destroys** recovery (negative control).
- **Process isolation** reproduces the in-process reference exactly, and mask
  handles never cross to the untrusted side.
- Validated TDX execution at the platform level: guest device + TDREPORT +
  NO_DEBUG/SEPT_VE_DISABLE.

**Claims that must be downgraded / avoided**
- **No** end-to-end remote attestation: Quote generation/verification is
  pending vendor QGS/evidence support; do not claim a verified attestation
  chain.
- **No** production-ready TEE, **no** semantic/cryptographic/formal security.
- The runtime does **not** hide sequence length, batch size, decode-step count,
  or attention scores.
- The masks are orthogonal + positive scaling: they hide coordinate/index
  identities but **preserve norms and relative geometry** (see Stage 8.2
  hidden-structure leakage accounting). This is explicit leakage accounting, not
  a guarantee.
- Microbenchmark latencies measure the **boundary only** (the untrusted decoder
  is stubbed); they are not full-model generation latencies.

In short: this stage demonstrates a *deployable boundary architecture* with a
small trusted base and verified correctness/negative-controls, on a platform
where TDX execution is confirmed but full remote attestation is still pending.
