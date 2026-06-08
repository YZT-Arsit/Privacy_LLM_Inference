# Decoder-only Component Coverage Audit

_Stage 7.8d: paper-appendix-ready table of which decoder-only components are covered by the masked low-interaction protocol._

## Summary

| Category | Count |
|---|---|
| Covered in main protocol | 10 |
| Partially covered / extension | 6 |
| Not covered (future work) | 9 |

## A. Covered in Main Protocol

| Component | Status | Reason | Evidence | Remaining Blocker |
|---|---|---|---|---|
| `RMSNorm` | **supported** | operator-compatible orthogonal: RMSNormCore(H Q) = RMSNormCore(H) Q | `outputs/modern_decoder_low_interaction_correctness.json` | none for algebraic claim |
| `SwiGLU` | **supported** | paired permutation P shared by gate / up branches | `outputs/modern_decoder_low_interaction_correctness.json` | none for algebraic claim |
| `standard 1D RoPE` | **supported** | rotate-half RoPE commutes with RoPE-plane block-diagonal mask | `outputs/modern_decoder_rope_safe_low_interaction.json` | none for algebraic claim |
| `GQA / MQA` | **supported** | N_Q[h] = N_K[h//group_size]^{-T} -> Q_tilde K_tilde^T = Q K^T | `outputs/modern_decoder_rope_safe_low_interaction.json` | none for algebraic claim |
| `causal attention` | **supported** | QK score invariant + causal mask in softmax | `outputs/attention_privacy_modes.json` | n/a |
| `KV cache` | **supported** | K_tilde = K @ N_K, V_tilde = V @ N_V per session | `outputs/modern_decoder_low_interaction_correctness.json` | real serving runtime |
| `paged KV abstraction` | **supported** | block-table indexing preserves the masked KV invariant | `outputs/paged_kv_abstraction.json` | real GPU paged-attention kernel |
| `LM head` | **supported** | padded masked logits with trusted recovery | `outputs/lm_head_scalability.json` | n/a (see LM-head scalability) |
| `LoRA inference` | **supported** | padded LoRA boundary identity (Stage 7.7b) | `outputs/lora_protocol_integration.json` | LoRA training (backward) |
| `generation processors inside TEE` | **supported** | main theorem (Stage 7.8c): logit processors are exact on recovered logits | `outputs/generation_processor_coverage.json` | output-length side-channel hiding |

## B. Partially Covered / Extension

| Component | Status | Reason | Evidence | Remaining Blocker |
|---|---|---|---|---|
| `sliding window attention` | **supported** | Stage 7.8a: KV window invariant + score invariant under window cut-off | `outputs/sliding_window_attention.json` | real FlashAttention / sliding-window CUDA kernel |
| `LayerNorm (non-LLaMA path)` | **audit_only** | theory mirrors RMSNorm-compatible orthogonal mask but is not the main Llama/Qwen path | `n/a` | extra constraint that Q preserves mean direction; not exercised here |
| `GELU MLP (non-SwiGLU path)` | **audit_only** | permutation island theory exists from earlier stages but not the main SwiGLU path | `outputs/nonlinear_island_experiments.json` | not exercised under the latest low-interaction wrapper |
| `prefix cache (cross-session sharing)` | **audit_only** | private mode only; cross-session sharing flagged as leakage surface | `outputs/paged_kv_abstraction.json` | explicit public-prefix flag + threat-model declaration |
| `beam search` | **audit_only** | main theorem applies; not implemented end-to-end here | `outputs/generation_processor_coverage.json` | TEE-resident beam manager |
| `quantization (fp16 / bf16 / int8 / int4)` | **partially_supported** | Stage 7.8b: simulated only; well-conditioned masks recommended for low precision | `outputs/precision_quantization_stability.json` | real GPU fp16 / bf16 / int8 / int4 kernels |

## C. Not Covered (Future Work)

| Component | Status | Reason | Evidence | Remaining Blocker |
|---|---|---|---|---|
| `M-RoPE / multimodal positional encoding` | **unsupported** | M-RoPE mixes multiple position axes; one block-diagonal rotation may not commute across all axes | `n/a` | extend RoPE-plane analysis to multiple position dimensions |
| `MoE router / expert dispatch` | **unsupported** | router output reveals expert selection; routing decisions cross the masked invariant | `n/a` | trusted routing or masked expert dispatch |
| `Multi-Head Latent Attention` | **unsupported** | latent compression changes the (Q, K, V) algebra; not covered by current QK invariant | `n/a` | MLA-specific invariant derivation |
| `speculative decoding` | **unsupported** | draft / target verification protocol crosses TEE boundary; not analysed | `n/a` | speculative-decode threat model + masked draft model |
| `real vLLM / FlashAttention backend` | **unsupported** | CPU local emulation only; no real GPU kernel | `n/a` | real GPU + serving runtime |
| `real GPU / TEE hardware side channels` | **unsupported** | no real hardware platform available | `n/a` | real hardware + side-channel platform |
| `full active malicious security` | **unsupported** | Stage 7.7e is probabilistic spot-check only; no verifiable computation | `outputs/integrity_spotcheck.json` | cryptographic verifiable computation |
| `LoRA training (backward)` | **unsupported** | Stage 7.7b is forward only | `n/a` | backward path through padded boundary; gradient masks |
| `full Qwen / LLaMA deployment` | **unsupported** | synthetic tiny modern decoder only; no real model loader | `n/a` | real HF / safetensors loader + GPU |

## Limitations

- CPU local emulation only.
- No real TEE / GPU deployment.
- No hardware side-channel evaluation.
- No formal cryptographic / semantic / differential-privacy security.
- No full Qwen / LLaMA deployment unless a real wrapper exists.

## Paper-Safe Wording

> We provide a coverage table for common decoder-only components. Supported components carry algebraic evidence under CPU local emulation; partially supported components have audit-only or simulation-only evidence; unsupported components are listed as future work with explicit remaining blockers.

## Unsafe Wording to Avoid

- M-RoPE supported.
- MoE supported.
- Speculative decoding supported.
- Real quantized model deployment.
- Real vLLM serving support.
- Hardware side channels evaluated.

