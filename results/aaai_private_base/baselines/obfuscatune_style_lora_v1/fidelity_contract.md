# OBFUSCATUNE_STYLE_LORA_BASELINE fidelity contract

Status: **FIDELITY_GATE_BLOCKED — FORMAL GPU EXECUTION PROHIBITED**

Paper-facing label, if and only if the missing implementation is later completed:
`OBFUSCATUNE_STYLE_LORA_BASELINE`. This is not an official reproduction.

## Mechanism and scope

The intended source mechanism is the linear-obfuscation split from ObfuscaTune
(arXiv:2407.02960): high-parameter linear work is transformed for an untrusted
accelerator, while de-obfuscation, nonlinearities, normalization, residual
bookkeeping, and other sensitive operations remain in a trusted runtime. The
repository implements a Qwen inference simulator for this mechanism, but it does
not implement Qwen LoRA finetuning or a real A10-to-TDX ObfuscaTune training
boundary.

| Component | Classification | Contract |
|---|---|---|
| Dense linear input/output transformation | FAITHFUL | Orthogonal transform, condition number 1; algebraic simulator tests pass. |
| Qwen RMSNorm | ADAPTED_WITH_JUSTIFICATION | Runs on de-obfuscated plaintext in the simulated trusted boundary; fp32 variance then cast-back matches Qwen2 RMSNorm. |
| Qwen SwiGLU | ADAPTED_WITH_JUSTIFICATION | Gate/up projections use the linear transformation; SiLU and the elementwise product remain trusted/plaintext; down projection is output-transformed. |
| Qwen RoPE | ADAPTED_WITH_JUSTIFICATION | Applied to plaintext Q/K in the trusted domain using the HF Qwen2 helper. |
| Grouped-query attention | ADAPTED_WITH_JUSTIFICATION | `repeat_kv` follows RoPE; cache stores pre-repeat K/V heads. |
| Qwen prefill/decode inference | ADAPTED_WITH_JUSTIFICATION | Simulator implementation and CPU equivalence tests exist. |
| KV-cache protection | NOT_IMPLEMENTED | The baseline cache is explicitly plaintext/intermediate-exposed. |
| Qwen LoRA target handling | NOT_IMPLEMENTED | No protected Qwen LoRA training path exists for q/k/v/o/gate/up/down. |
| Optimizer placement | NOT_IMPLEMENTED | No baseline-faithful optimizer placement across a real TDX/A10 split exists. |
| Trusted/untrusted training partition | NOT_IMPLEMENTED | Current Qwen path is an in-process simulator, not a real isolated TDX service. |
| Adapter transform/export | NOT_IMPLEMENTED | No transformed adapter package, integrity manifest, or baseline-supported export exists. |
| Post-training inference handoff | NOT_IMPLEMENTED | No baseline-produced protected adapter can be loaded by a fresh baseline inference process. |
| Real TDX attestation/authentication | NOT_IMPLEMENTED | Existing ObfuscaTune code records assumed authentication only; it is not bound to the repository's real TDX attestation path. |
| Plaintext embedding/final LM head in trusted runtime | ADAPTED_WITH_JUSTIFICATION | Present in the simulator; no real TDX deployment exists. |
| Official-paper GPT-2/nanoGPT protocol | NOT_APPLICABLE | This requested comparison adapts the method to Qwen2.5-0.5B and E2E NLG. |

## Security-view difference from our method

The intended baseline protects proprietary model weights and private data against
an honest-but-curious accelerator, but exposes plaintext Q/K/V, attention-related
intermediates, MLP gate/up intermediates, and plaintext KV cache under the current
Qwen mapping. Our protected G2 cell instead evaluates a public-base setting with
masked LoRA/hidden/logit state and a different trusted/untrusted partition. These
are different security views; only explicitly matched utility and measured system
fields may be compared.

## Gate decision

Launching the existing simulator as a real ObfuscaTune-style LoRA baseline would
misrepresent the baseline because forward/backward, optimizer update, checkpoint,
adapter export, fresh-process load, and real TDX execution are absent. The smoke
and formal commands are therefore intentionally **undefined**. Seed 1234, 7, and
2025 must not be launched until a versioned implementation closes these items and
adds positive and fail-closed tests.

