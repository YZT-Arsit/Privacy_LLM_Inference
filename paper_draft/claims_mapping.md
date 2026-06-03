# Claims Mapping

This document pins every paper-body claim to the authoritative claims audit in `paper_results/markdown/paper_claims_audit.md`. Every claim has four fields: `status` (`supported` / `proxy_supported` / `unsupported`); the `evidence_artifacts`; the `paper_safe_wording` to use verbatim or paraphrase; and the `unsafe_wording_to_avoid`. The wording columns are quoted verbatim from the audit; do not edit them when copying claims into the paper body.

## Supported claims (8)

### S1. GPT-2 model-level masked execution preserves output equality
- **Status.** `supported`.
- **Evidence.** `outputs/gpt2_model_correctness.json`, `outputs/gpt2_generation_correctness.json`, `outputs/workload_profile.json`.
- **Safe wording.** *"We empirically verify that the GPT-2 model-level masked wrapper reproduces the plain reference output token-for-token in our tested configurations."*
- **Unsafe wording.** *"Masked execution provably preserves output equality."*

### S2. Modern decoder-only masked execution preserves output equality
- **Status.** `supported`.
- **Evidence.** `outputs/modern_decoder_model_wrapper_smoke.json`, `outputs/modern_decoder_block_wrapper_smoke.json`, `outputs/modern_decoder_probe.json`.
- **Safe wording.** *"We empirically verify that the modern decoder-only model-level masked wrapper reproduces the plain reference output token-for-token in our tested configurations."*
- **Unsafe wording.** *"Masked execution provably preserves output equality for all modern decoder models."*

### S3. Compatible nonlinear islands preserve correctness across tested architectures
- **Status.** `supported`.
- **Evidence.** `outputs/nonlinear_island_experiments.json`, `outputs/cross_architecture_summary.json`, `outputs/cross_architecture_compatible_island_smoke.json`.
- **Safe wording.** *"Compatible mask families (mean-preserving orthogonal for LayerNorm, channel permutation for GELU/ReLU, paired permutation for SwiGLU) preserve output correctness in our tests; we do not claim they extend beyond the tested mask families."*
- **Unsafe wording.** *"Compatible mask families preserve correctness universally."*

### S4. KV cache append invariant holds for tested decoder-only paths
- **Status.** `supported`.
- **Evidence.** `outputs/kv_cache_correctness.json`, `outputs/gpt2_cache_correctness.json`, `outputs/modern_decoder_model_wrapper_smoke.json`.
- **Safe wording.** *"We verify the KV cache append invariant matches the plain reference in our tested decoder-only generation paths."*
- **Unsafe wording.** *"Our KV cache implementation is provably secure."*

### S5. LoRA masked forward / backward / rank-padded primitives match plain rank-r reference
- **Status.** `supported`.
- **Evidence.** `outputs/lora_training_experiments.json`, `outputs/lora_backward_experiments.json`, `outputs/lora_rank_padding_experiments.json`.
- **Safe wording.** *"We empirically verify that the masked LoRA forward / backward / rank-padded primitives match the plain rank-r reference to float64 precision on synthetic single-linear tests."*
- **Unsafe wording.** *"Our LoRA implementation provably matches the plain reference."*

### S6. Multi-layer rank-padded LoRA training reproduces the plain reference
- **Status.** `supported`.
- **Evidence.** `outputs/multilayer_lora_training_experiments.json`.
- **Safe wording.** *"We empirically verify that a synthetic multi-layer rank-padded LoRA training step matches the plain reference on every per-module slice."*
- **Unsafe wording.** *"Our multi-layer LoRA training scheme is identical to plain LoRA for any model."*

### S7. Stronger dummy distributions preserve the rank-padded LoRA invariant and training correctness
- **Status.** `supported`.
- **Evidence.** `outputs/lora_stronger_dummy_experiments.json`.
- **Safe wording.** *"Every tested stronger-dummy strategy preserves the rank-padded LoRA invariant `A_pad B_pad = A_real B_real` (either exactly or via a tracked trusted-side correction) and matches plain training to float64 precision in our tests."*
- **Unsafe wording.** *"Our stronger dummy distributions provably hide the true LoRA rank."*

### S8. Constant-time training proxy reduces timing classifier near random chance
- **Status.** `supported`.
- **Evidence.** `outputs/lora_training_timing_proxy.json`.
- **Safe wording.** *"Under the cost-model proxy, equalizing per-step latency to the upper bucket reduces the worst-case timing classifier accuracy to near random chance in our tests."*
- **Unsafe wording.** *"Constant-time training closes timing side channels."*

## Proxy-supported claims (5)

### P1. Activation-recovery / linkability proxy bounded under full mitigation bundle
- **Status.** `proxy_supported`.
- **Evidence.** `outputs/adaptive_island_attacks.json`, `outputs/real_activation_attacks.json`, `outputs/real_token_activation_attacks.json`, `outputs/stronger_attackers.json`.
- **Safe wording.** *"Under our adaptive proxy attackers (ridge, small MLP, signature / Sinkhorn permutation recovery, linkability), the full mitigation bundle keeps the worst-case attacker close to random chance in our tested configurations."*
- **Unsafe wording.** *"Our mitigation bundle is secure against arbitrary adversaries."*

### P2. Cost-model timing side-channel proxy bounded under proxy_equalized
- **Status.** `proxy_supported`.
- **Evidence.** `outputs/stronger_attackers.json`, `outputs/lora_training_timing_proxy.json`.
- **Safe wording.** *"Under our cost-model timing proxy, equalizing latency to the upper bucket reduces classifier accuracy near random chance."*
- **Unsafe wording.** *"Constant-time mode prevents timing side-channel attacks."*

### P3. LoRA adapter / gradient leakage proxy bounded under fresh masks + pad
- **Status.** `proxy_supported`.
- **Evidence.** `outputs/lora_security_proxy.json`, `outputs/lora_gradient_security_proxy.json`.
- **Safe wording.** *"Under our LoRA adapter / gradient leakage proxy attackers, fresh masks + pad bring the linkability AUC close to 0.5 (random chance) in our tests."*
- **Unsafe wording.** *"Our LoRA path is secure against adapter / gradient extraction."*

### P4. Rank padding + stronger dummies reduce proxy spectral inference of true_rank
- **Status.** `proxy_supported`.
- **Evidence.** `outputs/lora_rank_security_proxy.json`, `outputs/lora_stronger_dummy_security_proxy.json`.
- **Safe wording.** *"Rank padding hides true_rank from tensor shape; under our spectral-cliff / energy / elbow / ensemble proxy detectors, stronger dummy strategies fail to recover true_rank reliably in our tests (`needs_more_evaluation`)."*
- **Unsafe wording.** *"Our rank-padding scheme cryptographically hides the LoRA rank."*

### P5. Cross-layer linkage low under fresh masks + paired cancellation
- **Status.** `proxy_supported`.
- **Evidence.** `outputs/multilayer_lora_security_proxy.json`, `outputs/lora_stronger_dummy_security_proxy.json`.
- **Safe wording.** *"Under our cross-layer linkability proxy, fresh masks per module keep the AUC near 0.5 across the tested multi-layer configuration."*
- **Unsafe wording.** *"Cross-layer linkability is impossible."*

## Unsupported claims (8) — must NOT appear as positive claims in the paper body

### U1. Formal / cryptographic / semantic security
- **Status.** `unsupported`.
- **Safe wording.** *"We make no formal / cryptographic / semantic security claims."*
- **Unsafe wording.** *"Our scheme provides formal / cryptographic / semantic security."*

### U2. Real TEE wall-time
- **Status.** `unsupported`.
- **Safe wording.** *"We report measured local-emulation latencies; we do not claim real TEE wall-time."*
- **Unsafe wording.** *"Our wrapper achieves X ms latency on real TEE."*

### U3. Hardware side-channel security
- **Status.** `unsupported`.
- **Safe wording.** *"We do not evaluate hardware side-channels (cache / power / EM)."*
- **Unsafe wording.** *"Our system is resistant to hardware side-channel attacks."*

### U4. Full Qwen / TinyLlama / LLaMA LoRA fine-tuning
- **Status.** `unsupported`.
- **Safe wording.** *"Our LoRA experiments use synthetic single-linear and multi-layer tiles; we do not run full Qwen / TinyLlama / LLaMA LoRA fine-tuning."*
- **Unsafe wording.** *"Our LoRA path supports full Qwen / TinyLlama / LLaMA fine-tuning."*

### U5. PEFT / DeepSpeed / vLLM / FlashAttention compatibility
- **Status.** `unsupported`.
- **Safe wording.** *"We do not integrate PEFT / DeepSpeed / vLLM / FlashAttention; our LoRA primitives are stand-alone functional API."*
- **Unsafe wording.** *"Our scheme drops into PEFT / DeepSpeed / vLLM / FlashAttention."*

### U6. padded_rank is hidden from the GPU
- **Status.** `unsupported`.
- **Safe wording.** *"Stage 7.2 / 7.3 / 7.4 hide true_rank from tensor shape; padded_rank itself remains visible to the GPU."*
- **Unsafe wording.** *"Our rank padding hides the LoRA rank."*

### U7. Loss / optimizer is fully outsourced to the GPU
- **Status.** `unsupported`.
- **Safe wording.** *"Loss + optimizer remain trusted-side; only forward / backward matmuls cross the boundary."*
- **Unsafe wording.** *"Loss / optimizer run on the untrusted GPU."*

### U8. Protection against a compromised TEE
- **Status.** `unsupported`.
- **Safe wording.** *"We assume the TEE is honest; a compromised TEE breaks every guarantee we evaluate."*
- **Unsafe wording.** *"Our scheme protects against a compromised TEE."*

## Body sentence → claim mapping

Every body section pins back to the audit as follows:

- Abstract: S1, S2, S3, S4, S5, S6, S7, S8; P1, P2, P3, P4, P5; explicit disclaimers U1, U2, U4, U5, U6, U7.
- Introduction 1.5 Contributions: C1 → S1, S2, S4; C2 → S3; C3 → S5, S6, S7; C4 → cites the audit + summary directly.
- Background 2.5 Trust assumptions: U8 (compromised TEE out of scope).
- System and Threat Model 3.2: U1, U2, U3, U4, U5, U6, U7, U8 all listed verbatim under "out of scope".
- Design 5.7–5.9: U6 (padded rank visible), U7 (loss/optimizer trusted).
- Correctness 6: only supported claims (S1–S8) carry a "theorem" label; nothing in this section uses the word "secure".
- Security 7: only proxy_supported claims (P1–P5) carry positive statements; every paragraph is hedged with "in our tested configurations".
- Evaluation 8: each RQ result rows are tagged with their underlying claim ID in the row notes.
- Limitations 9: items 1–15 are the textual expansion of U1–U8 plus per-stage limitations from `paper_results/markdown/limitations_summary.md`.
- Related Work 10: positioning uses "proxy-evaluated", never "secure".
- Conclusion 11: closes with the same 8-item disclaimer block.

## How to check a new sentence before pasting it into the paper body

If a candidate sentence contains any of: `secure`, `provably`, `cryptographic`, `semantic security`, `prevents all leakage`, `guarantees`, `real TEE wall-time`, `hides padded rank`, `Qwen fine-tune`, `LLaMA fine-tune`, `PEFT`, `DeepSpeed`, `vLLM`, `FlashAttention`, or `fully outsourced loss / optimizer` — that sentence requires a corresponding `unsupported` audit row. Re-read U1–U8 and either re-word or move to Limitations.
