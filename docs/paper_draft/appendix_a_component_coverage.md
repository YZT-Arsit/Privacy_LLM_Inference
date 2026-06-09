# Appendix A — Decoder-only Component Coverage

This appendix consolidates the component-coverage classification from `outputs/decoder_component_coverage_audit.json` into a paper-ready table. **Unsupported components remain unsupported.** Any attempt to phrase an unsupported component as supported (e.g. "M-RoPE supported", "MoE supported", "speculative decoding supported", "real vLLM serving support") is in the explicit unsafe-wording list of the per-claim audit (`outputs/paper_claims_audit_v2.json`).

## A.1. Components covered in the main protocol

| Component | Status | Reason | Required invariant | Leakage surface | Evidence artifact | Remaining blocker |
|---|---|---|---|---|---|---|
| RMSNorm | supported | operator-compatible orthogonal: `rmsnorm_core(H Q) = rmsnorm_core(H) Q` | `Q` orthogonal; `gamma` folded into next linear | per-row L2 norms preserved (RMSNorm correctness requirement) | `outputs/modern_decoder_low_interaction_correctness.json` | none for algebraic claim |
| SwiGLU | supported | shared paired permutation `P` across gate / up branches; `(U P) ⊙ silu(G P) = (U ⊙ silu(G)) P` | `P` is a permutation matrix | permutation seed observable but uniform across batch | `outputs/modern_decoder_low_interaction_correctness.json` | none |
| standard 1D RoPE | supported | rotate-half RoPE commutes with block-diagonal 2D rotation in the same RoPE plane: `RoPE(X B) = RoPE(X) B` | `B` block-diagonal with one 2D rotation per `(j, j + d_h/2)` pair | per-RoPE-pair 2D norms preserved | `outputs/modern_decoder_rope_safe_low_interaction.json` | none |
| GQA / MQA causal attention | supported | per-Q-head `N_Q[h] = N_K[h // group_size]^T = B_K[group(h)]^T` → `B_Q B_K^T = I` → `Q_tilde K_tilde^T = Q K^T` | `B_Q B_K^T = I` per Q head | score matrix visible in `exact_visible_attention`; hidden in `trusted_softmax_attention` | `outputs/attention_privacy_modes.json` | n/a |
| causal softmax | supported | composes with QK invariant; `softmax(S_tilde + causal mask) = softmax(S + causal mask)` | identity score recovery | attention map in exact-visible mode; hidden in trusted-softmax | `outputs/attention_privacy_modes.json` | n/a |
| KV cache (standard) | supported | per-(session, layer, KV head) `K_tilde = K N_K, V_tilde = V N_V`; append commutes with right-multiplication | per-session orthogonal `N_K, N_V` | cache *length* visible; cache *values* not | `outputs/modern_decoder_low_interaction_correctness.json` | real serving runtime |
| paged KV abstraction | supported | block-table indexing preserves the per-block masked invariant; full-cache invariant via `gather_full_tilde` | mask shared across all blocks of a session per (layer, head) | block-table observable; timing not evaluated | `outputs/paged_kv_abstraction.json` | real GPU paged-attention kernel |
| LM head (dense, permutation, block) | supported | `z_tilde = z N_vocab`, `z = z_tilde N_vocab^{-1}` | `N_vocab` invertible (orthogonal / permutation / block-diagonal) | dense not scalable; permutation preserves logit multiset; block reveals block membership | `outputs/lm_head_scalability.json` | scalable LM head choice required for real `V` |
| LoRA inference | supported | padded LoRA boundary `A_tilde = M^{-1} A R, B_tilde = R^{-1} B N_out` | per-site `N_out`: `B_Q, B_K, N_V, Q_l, P` | padded rank `r_pad` observable; true rank `r` hidden by zero pad | `outputs/lora_protocol_integration.json` | LoRA training (backward) |
| trusted-side generation processors | supported | Lemma 9: logit processors are exact under recovered logits | `z_recovered = z_plain` at machine precision | output length / stop timing observable unless padded | `outputs/generation_processor_coverage.json` | output-length side-channel hiding |

## A.2. Components partially covered / extensions

| Component | Status | Reason | Required invariant | Leakage surface | Evidence artifact | Remaining blocker |
|---|---|---|---|---|---|---|
| sliding window attention | supported | Stage 7.8a: per-rolling-window `K_tilde = K_plain N_K` invariant preserved; full-vs-sliding equality when `w >= s_total` | window size `w` public; per-(layer, head) mask shared across rolling window | window-size policy public; timing not evaluated | `outputs/sliding_window_attention.json` | real FlashAttention / sliding-window CUDA kernel |
| LayerNorm (non-LLaMA path) | audit_only | theory mirrors RMSNorm-compatible orthogonal mask; mean preservation requires extra constraint | `Q` orthogonal AND row-mean-preserving | row mean preserved | n/a (audit only) | extra constraint on `Q` to preserve mean direction; not exercised here |
| GELU MLP (non-SwiGLU path) | audit_only | permutation island theory from earlier stages; GELU is element-wise and permutation-equivariant | shared permutation across input axis | permutation seed observable | `outputs/nonlinear_island_experiments.json` | not exercised under the latest low-interaction wrapper |
| prefix cache (cross-session sharing) | audit_only | private mode is on by default; cross-session sharing must be flagged as leakage | n/a (off by default) | if enabled, shared prefix `K_tilde, V_tilde` rows leak across sessions | `outputs/paged_kv_abstraction.json` | explicit `public_prefix` flag + threat-model declaration |
| beam search | audit_only | Lemma 9 covers deterministic per-step argmax over expanded `(prefix, candidate)` pairs | recovered logits at machine precision | beam width / state observable on output channel | `outputs/generation_processor_coverage.json` | TEE-resident beam manager |
| quantisation (fp16 / bf16 / int8 / int4) | partially_supported | Stage 7.8b: simulated only; well-conditioned masks recommended for low precision | mask family well-conditioned (cond ≈ 1) | per-channel quantisation scale observable | `outputs/precision_quantization_stability.json` | real GPU fp16 / bf16 / int8 / int4 kernels |

## A.3. Components not covered (future work)

These components remain *unsupported* in this paper. They are listed here to make the scope explicit; they are *not* contributions.

| Component | Status | Reason | Required invariant | Leakage surface | Evidence artifact | Remaining blocker |
|---|---|---|---|---|---|---|
| M-RoPE / multimodal positional encoding | unsupported | M-RoPE mixes multiple position axes; one block-diagonal rotation may not commute across all axes | future work | n/a | n/a | extend RoPE-plane analysis to multiple position dimensions |
| MoE router / expert dispatch | unsupported | router output reveals expert selection; routing decisions cross the masked invariant | future work | n/a | n/a | trusted routing or masked expert dispatch |
| Multi-Head Latent Attention (MLA) | unsupported | latent compression changes the (Q, K, V) algebra; current QK invariant does not cover MLA | future work | n/a | n/a | MLA-specific invariant derivation |
| speculative decoding | unsupported | draft / target verification crosses the TEE boundary; not analysed | future work | n/a | n/a | speculative-decode threat model + masked draft model |
| real vLLM / FlashAttention backend | unsupported | CPU local emulation only; no real GPU kernel | n/a | n/a | n/a | real GPU + serving runtime |
| real GPU / TEE hardware side channels | unsupported | no real hardware platform available | n/a | n/a | n/a | real hardware + side-channel platform |
| full active malicious security | unsupported | Stage 7.7e is probabilistic spot-check only; no verifiable computation | n/a | n/a | `outputs/integrity_spotcheck.json` (proxy only) | cryptographic verifiable computation |
| LoRA training (backward) | unsupported | Stage 7.7b is forward only | n/a | n/a | n/a | backward path through padded boundary; gradient masks |
| full Qwen / LLaMA deployment | unsupported | synthetic tiny modern decoder only; no real model loader | n/a | n/a | n/a | real HF / safetensors loader + GPU |

## A.4. Reading this appendix

The classifier is *dynamic* and is regenerated from `outputs/decoder_component_coverage_audit.json` on every aggregator run. Components whose evidence artifact does *not* exist are downgraded automatically (e.g. sliding window would drop from `supported` to `audit_only` if `outputs/sliding_window_attention.json` were deleted). The pytest suite enforces that no row of section A.3 is promoted to A.1 or A.2 without the corresponding evidence (`test_unsupported_marked_unsupported`, `test_no_unsupported_component_marked_supported`).
