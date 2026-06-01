# Stage 7.0 — LoRA Security Proxy

## 1. Experiment Scope

- d_in=32, d_out=16, rank=4, alpha=1.0.
- num_trials=32, membership_trials_per_sample=8, dtype=float64.
- strategies: ['unmasked_adapter_baseline', 'fixed_masks_fixed_u', 'fresh_u_only', 'fresh_masks_fresh_u', 'fresh_masks_fresh_u_with_pad']
- scope: single linear + LoRA, tiny dimensions, synthetic adapter

## 2. Threat Model

- Passive GPU observer of the masked transcript (X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde) under one masked LoRA linear.
- Knows the model architecture, the LoRA rank dimension, and the masking scheme; does NOT know N_in, N_out, U, T, the plaintext A / B, or the private (X, Y_target).
- No hardware side-channel (cache / power / EM) and no active boundary attack.
- This is a *proxy*: ranks strategies under three sub-attacks. It does NOT prove security.

## 3. Adapter Extraction Proxy

| strategy | dW_recovery_rel_l2 | adapter_A_rel_l2 | adapter_B_rel_l2 | rank_signature_A | rank_visible | subspace_sim_A | sv_sim_A |
|----------|---------------------|------------------|------------------|------------------|--------------|----------------|----------|
| unmasked_adapter_baseline | 0.000 | 0.000 | 0.000 | 4/4 | True | 1.000 | 1.000 |
| fixed_masks_fixed_u | 1.458 | 1.441 | 1.454 | 4/4 | True | 0.256 | 1.000 |
| fresh_u_only | 1.458 | 1.422 | 1.407 | 4/4 | True | 0.256 | 1.000 |
| fresh_masks_fresh_u | 1.421 | 1.414 | 1.401 | 4/4 | True | 0.310 | 1.000 |
| fresh_masks_fresh_u_with_pad | 1.411 | 1.410 | 1.398 | 4/4 | True | 0.279 | 1.000 |

- **unmasked_adapter_baseline**: Adapter is fully exposed; ΔW = A B reconstructible exactly.
- **fixed_masks_fixed_u**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive extraction unreliable.
- **fresh_u_only**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive extraction unreliable.
- **fresh_masks_fresh_u**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive extraction unreliable.
- **fresh_masks_fresh_u_with_pad**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive extraction unreliable.

## 4. Gradient Leakage Accounting

Representative strategy: **fresh_masks_fresh_u_with_pad**.

| variable | visibility | plaintext | leakage_risk | mitigation | stage_7_0_status |
|----------|------------|-----------|--------------|------------|------------------|
| private_input_X | trusted | True | low | Right-mask + optional input pad before crossing the boundary. | covered |
| private_target_Y | trusted | True | low | Stays trusted; loss computed inside the trusted side. | covered |
| adapter_A | trusted | True | low | A_tilde = N_in^{-1} A U; fresh U per call. | covered |
| adapter_B | trusted | True | low | B_tilde = U^{-1} B N_out; fresh U per call. | covered |
| grad_A | trusted | True | low | Backward remains trusted in Stage 7.0; gradient never crosses the boundary. | trusted_backward_prototype |
| grad_B | trusted | True | low | Backward remains trusted in Stage 7.0; gradient never crosses the boundary. | trusted_backward_prototype |
| optimizer_state (SGD momentum / AdamW m, v) | trusted | True | low | Trusted-only; never exported to JSON/CSV/Markdown. | covered |
| base_weight_W | public | True | low | Public model weight; ΔW = A B does NOT merge into W. | covered |
| X_tilde | gpu | False | low | Right-mask + optional pad; fresh N_in per call. | covered |
| A_tilde / B_tilde | gpu | False | medium | U-mask in rank space; fresh U recommended. | covered |
| Y_tilde | gpu | False | low | Right-mask via N_out; recovered only on trusted side. | covered |

Per-strategy variation lives in the JSON / CSV; the GPU-visibility contract is the same across strategies, only the *risk level* differs.

## 5. Membership-Style Linkability Proxy

| strategy | same_sample_dist | different_sample_dist | AUC_proxy | linkability_rank | risk_level |
|----------|-------------------|------------------------|-----------|--------------------|------------|
| unmasked_adapter_baseline | 0.000 | 22.804 | 1.000 | 1.000 | high |
| fixed_masks_fixed_u | 0.000 | 22.804 | 1.000 | 1.000 | high |
| fresh_u_only | 0.000 | 22.804 | 1.000 | 1.000 | high |
| fresh_masks_fresh_u | 22.925 | 23.086 | 0.555 | 0.109 | low |
| fresh_masks_fresh_u_with_pad | 32.149 | 32.319 | 0.537 | 0.074 | low |

## 6. Interpretation

- **linkability_summary**: fresh masks reduce membership-style linkability (Δ AUC = +0.463 vs fixed_masks_fixed_u).
- **rank_visibility_note**: LoRA rank r is visible from the shape of A_tilde / B_tilde under all current strategies; rank padding is NOT implemented in Stage 7.0.
- **trusted_backward_status**: training backward remains trusted in Stage 7.0 prototype
- **merge_adapter_into_w**: False

## 7. Limitations

- These are proxy attacks, not formal security proofs.
- LoRA rank ``r`` remains visible from the shape of A_tilde / B_tilde unless explicit rank padding is implemented (NOT in Stage 7.0).
- Optimizer state (SGD momentum / AdamW moments) remains trusted-only in Stage 7.0 and is never exported.
- No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.
- No full private fine-tuning workload is evaluated; this is a single-linear, tiny-dimension proxy.
- Adversary model is a passive GPU observer of (X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde) transcripts plus the dimensions; active boundary-attack and adaptive-attack proxies are deferred to later stages.
- Hardware side-channel attacks (cache / power / EM) are NOT evaluated.
- Adapter is NEVER merged into the public base weight W (constraint 7).

## 8. Next Stage Plan

- Stage 7.1 — masked backward / gradient-side obfuscation.
- Stage 7.2 — rank padding to hide r.
- Stage 7.3 — multi-layer LoRA + cross-layer adapter linkability.
