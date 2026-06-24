# TDX-Attested Boundary Protocol — Experiment Summary

Status of the Stage 8.5 trusted-boundary ↔ untrusted-GPU-worker protocol under a
**real Intel TDX guest on Alibaba Cloud**. This summarises what has been validated
end-to-end and states precisely what has **not** yet been done.

See [`tee_gpu_protocol.md`](tee_gpu_protocol.md) for the protocol/audit design and
[`tee_boundary_design.md`](tee_boundary_design.md) for the boundary threat model.

> **Scope of claims.** The attested run below uses the **mock** GPU backend (a
> numpy identity decoder), so the round trip is exact and CPU-reproducible. The
> full Qwen2.5-7B masked execution is a **separate, untrusted-GPU** evaluation on
> H800 (`tee_used=False`); it did **not** run inside TDX, and a cross-machine
> TDX-boundary + H800-worker end-to-end has **not** been completed yet.

## 1. TDX quote generation + Alibaba Cloud attestation

Verified on the Alibaba Cloud TDX VM:

- A **TD Quote was generated** for the trusted-boundary guest.
- The **Alibaba Cloud Attestation API returned a signed JWT** (3 parts:
  header.payload.signature).
- JWT claims: `tee = tdx`, `tdx.td_attributes.debug = false`.
- `mr_td = e0199499baacb2e4f4bc73046f25bedf674d42defbe4e854242bd6554a9d155edf7f3bff8e6202e63ed230e59ab2568a`.

The JWT signature / certificate chain is verified by the Alibaba Cloud attestation
service; our module verifies the `tee` / `debug` / binding / `mr_td` **claims**.

## 2. Runtime-hash binding through `report_data`

The boundary binds a **runtime hash** into the TD Quote's 64-byte `report_data`.
The runtime hash is SHA-512 over the trusted-boundary *manifest*
(`build_trusted_boundary_manifest`): SHA-256 digests of the measured boundary
source files + public runtime identity (`protocol_version`, `boundary_backend`,
`allowed_gpu_backend`, `expected_mr_td`, Python/package versions), with the raw
prompt, `input_ids`, generated tokens, recovered logits, mask secrets, and model
weights explicitly **excluded**.

- `report_data` **equals** the runtime hash (binding verified).
- `runtime_hash = de3fa0bc972fdf418e6877b507195c3844877857c1038950e6d325e0614a01ed8aabc88cf5f29c6157a3236be6a5a89a9850f4e8b8341c8558c7e90abc15dffe`
- `runtime_hash_bound = True`

The hash is read off with `scripts/write_tee_boundary_runtime_hash.py` (or
`run_tee_gpu_protocol_demo.py --print-runtime-hash-only`) **after** the code is
frozen, bound into the quote, then recomputed and checked at verification time
with the identical recipe — so the quote is never bound to a stale hash.

## 3. Attested protocol demo result

`run_tee_gpu_protocol_demo.py` (`mode=local_two_process`, boundary in a
TDX-attested process, **mock** GPU worker in a separate untrusted process):

| field | value |
|---|---|
| `audit_passed` | **True** |
| `boundary_tee_type` | `tdx` |
| `boundary_attested` | **True** |
| `runtime_hash_bound` | **True** |
| `tee_used_on_gpu` | **False** |
| `boundary_calls` | `{'embed_and_mask': 4, 'recover_logits': 4, 'sample': 4}` |
| `trusted_bytes` | 32,128 |
| `gpu_bytes` | 1,063,680 |
| `mr_td` | `e0199499…ab2568a` |

The boundary performed the trusted work (embedding+masking, logit recovery,
greedy selection, remasking each next-token embedding); the GPU worker only
received masked tensors + public metadata and returned masked logits.

## 4. Security audit result

Audited against the *exact* messages recorded as crossing to the untrusted GPU
worker:

- `gpu_visible_plaintext_fields = []` — no raw prompt, `input_ids`, generated
  token ids, recovered logits, or tokenizer output crossed to the GPU.
- `leaked_secret_fields = []` — no mask perm / signs / scale / seed /
  `MaskHandles` crossed.
- `tee_used_on_gpu = False` — the model side is never a TEE.
- Wrong-mask control: recovering the masked logits with the wrong mask flips the
  argmax and diverges by a large fraction of the logit scale (recovery fails).

## 5. Limitations (explicit)

1. **Mock GPU backend.** The attested demo above uses the numpy identity-decoder
   mock backend. It validates the *boundary protocol, masking, recovery, and
   audit* — not a transformer.
2. **Qwen2.5-7B is evaluated separately, on the untrusted GPU.** Full-layer
   masked Qwen2.5-7B execution (28/28 layers, teacher-forced HF-prefix top-1
   agreement = 1.0) was measured on **H800 with `tee_used=False`**. It is an
   untrusted-GPU evaluation.
3. **Qwen2.5-7B did NOT run inside TDX.** No transformer, decoder, attention,
   MLP, KV cache, or LM head runs in the TEE — by design.
4. **Cross-machine transport implemented; full TDX+H800+Qwen run NOT completed.**
   The cross-machine protocol modes now exist (`gpu_worker_server` /
   `boundary_client` over stdlib HTTP) and are **validated over localhost with the
   mock backend** (two processes, `audit_passed=True`, no plaintext/secret on the
   wire, `tee_used_on_gpu=false`, forbidden-field rejection enforced server-side).
   A combined deployment with the trusted boundary inside the Alibaba Cloud TDX
   guest driving an H800 GPU worker running **masked Qwen2.5-7B** over this
   protocol has **not** been run yet. The pieces are validated independently:
   (a) the attested boundary + mock worker, (b) the cross-machine HTTP transport
   (mock, localhost), and (c) the H800 masked-Qwen evaluation (`tee_used=False`).

## 6. What would close the gap

The transport is now in place (`--mode gpu_worker_server` on the H800,
`--mode boundary_client --gpu-worker-url ...` in the TDX guest; see
[`tee_gpu_protocol.md`](tee_gpu_protocol.md) §8). To close the remaining gap:

1. Start `--mode gpu_worker_server --gpu-backend qwen7b` on the H800 and bridge
   the masked Qwen2.5-7B pipeline so the worker consumes only masked protocol
   messages (the current `qwen7b` backend serves `/health` + `/init` with
   `tee_used_on_gpu=false`; masked prefill/decode over the protocol is the
   outstanding piece).
2. Run `--mode boundary_client` inside the TDX guest with valid attestation
   evidence, and confirm on that combined run: `runtime_hash_bound=True`,
   `boundary_attested=True`, `gpu_visible_plaintext_fields=[]`,
   `leaked_secret_fields=[]`, `tee_used_on_gpu=False`.

Until that combined run is executed, no cross-machine TDX+H800 end-to-end with
Qwen2.5-7B is claimed.
