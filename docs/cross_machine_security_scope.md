# Cross-machine security scope (do not conflate)

The paper makes **three separable claims**. They are produced by different runs
and must never be presented as one number. Every cross-machine report stamps
`compute_correctness_source`, `security_boundary_source`, `connectivity_note`,
and `limitations` so a reader can always tell them apart.

## 1. Compute correctness + scaling — standalone H800 (E1/E2)

`scripts/run_qwen7b_e1_nolora_generation.py` and
`scripts/run_qwen7b_e2_token_scaling.py` validate **full Qwen2.5-7B masked
compute correctness, generation behaviour, and token scaling** on the H800.

- This is the **main compute result**. Fixed config: bs=1, seq_len=128,
  max_new_tokens=64 (E1) and the {1,8,16,32,64} sweep (E2), 28/28 layers, bf16.
- Headline correctness for bf16 is **teacher-forced top-1 agreement** and
  **plain-vs-masked token match**, plus logits errors and top-k overlap. (Long
  free-running greedy may diverge after a near-tie; that is expected, not a
  defect.)
- `tee_used_on_gpu=false`: the model never enters a TEE.

## 2. Attested trusted boundary driving a remote untrusted GPU worker — TDX+H800

`scripts/run_tee_gpu_protocol_demo.py` (`--mode gpu_worker_server` on H800,
`--mode boundary_client` on the TDX VM) validates the **attested trusted
boundary driving a remote untrusted GPU worker**.

- **`--gpu-backend mock` runs end-to-end cross-machine** (`cross_machine_compute
  = end_to_end`): real TDX attestation, real wire-field rejection, masked-tensor
  -only traffic, full security audit, recovered tokens match the trusted
  plaintext reference over HTTP. The decoder is an identity model, so this
  proves the *protocol + boundary + audit*, not Qwen math.
- **`--gpu-backend qwen7b` runs an init/attestation/audit probe**
  (`cross_machine_compute = probe_only`): the real `/health` + `/init`
  handshake to the H800 worker, server-reported `tee_used_on_gpu=false`, and an
  audit of the init traffic (only masked/public metadata crosses). The full
  Qwen2.5-7B masked prefill/decode is the standalone E1/E2 result (claim 1).

### Why qwen7b cross-machine is a probe, not end-to-end

A **private** cross-machine masked Qwen decode requires the untrusted worker to
hold the **folded layer weights** (`W_tilde = N^{-1} W M`) while **never** seeing
the masks `N`, `M`. Producing the masked logits from `h_tilde` + folded weights
alone is mask-free, but folding needs the masks, so the trusted boundary must
fold offline and **ship the folded weights** to the worker. For Qwen2.5-7B that
is ~14 GB (bf16) of folded tensors — impractical over the stdlib JSON HTTP
transport. The alternative (worker holds the plaintext model + reconstructs the
masks from a shared seed) is functionally runnable but **not private** — the
worker could unmask — so it is deliberately **not** offered as a security run.

Therefore the masked compute is validated standalone (claim 1) and the attested
boundary is validated with the mock end-to-end run + the qwen7b init probe +
TDX attestation (claim 2). This split is reported explicitly in every run; it is
the sanctioned fallback when full cross-machine 64-token qwen7b is too heavy.

## 3. Connectivity is not a contribution

Cross-machine connectivity is the **deployment setting** for the attested
protocol, not a contribution. Networking working (or not) says nothing about the
privacy guarantee; it only determines whether claim 2 runs remote vs local.

## Discipline reminders

- Do **not** present 4-token smoke tests as paper results — debugging only. The
  main cross-machine target is `max_new_tokens=64`.
- Do **not** claim `tee_used_on_gpu=true` anywhere: the model/decoder/attention/
  MLP/KV cache/LM head always run outside any TEE.
- If full cross-machine 64-token qwen7b is blocked (heavy or networking), report
  the limitation explicitly and keep standalone H800 E1/E2 as the compute result
  plus TDX local attestation/protocol as the security-boundary result.

## Forbidden wire fields (worker rejects with HTTP 400)

The GPU worker server (`pllo.protocol.remote.FORBIDDEN_WIRE_FIELDS`) rejects any
request body containing, anywhere: `raw_prompt`, `prompt`, `input_ids`,
`generated_token_ids`, `recovered_logits`, `mask_secret`, `tokenizer_output`,
`labels`, `train_examples`, `tokenized_examples`, `plain_hidden`, `lora_a`,
`lora_b`, `delta_w`, `lora_grad_a`, `lora_grad_b`, `optimizer_state`,
`adapter_update` (plus the other mask-secret / training-stage aliases).
