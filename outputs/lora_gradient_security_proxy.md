# Stage 7.1 — LoRA Gradient Security Proxy

## 1. Experiment Scope

- d_in=32, d_out=16, rank=4, alpha=1.0.
- num_trials=32, membership_trials_per_sample=8, dtype=float64.
- strategies: ['unmasked_gradient_baseline', 'fixed_masks_fixed_u', 'fresh_u_only', 'fresh_masks_fresh_u', 'fresh_masks_fresh_u_with_pad']
- scope: single linear + LoRA, tiny dimensions, synthetic adapter and synthetic upstream gradient

## 2. Threat Model

- Passive GPU observer of the masked forward + backward transcript: (X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde, G_tilde, grad_A_tilde, grad_B_tilde).
- Knows the model architecture, the LoRA rank dimension, and the masking scheme; does NOT know N_in, N_out, U, T, plaintext A / B / G / grad_A / grad_B, optimizer state, or the private (X, Y_target).
- No hardware side-channel (cache / power / EM) and no active boundary attack.
- This is a *proxy*: ranks strategies under three sub-attacks. It does NOT prove security.

## 3. Gradient Extraction Proxy

| strategy | grad_A_rel_l2 | grad_B_rel_l2 | grad_subspace_A | gradient_norm_sim_A | sv_sim_A | rank_signature_A | rank_visible |
|----------|---------------|---------------|------------------|---------------------|----------|------------------|--------------|
| unmasked_gradient_baseline | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 4/4 | True |
| fixed_masks_fixed_u | 1.420 | 1.400 | 0.287 | 1.000 | 1.000 | 4/4 | True |
| fresh_u_only | 1.403 | 1.422 | 0.288 | 1.000 | 1.000 | 4/4 | True |
| fresh_masks_fresh_u | 1.414 | 1.425 | 0.285 | 1.000 | 1.000 | 4/4 | True |
| fresh_masks_fresh_u_with_pad | 1.759 | 1.767 | 0.281 | 0.694 | 0.996 | 4/4 | True |

- **unmasked_gradient_baseline**: Gradient is fully exposed; attacker recovers grad_A / grad_B exactly.
- **fixed_masks_fixed_u**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive gradient extraction unreliable.
- **fresh_u_only**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive gradient extraction unreliable.
- **fresh_masks_fresh_u**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive gradient extraction unreliable.
- **fresh_masks_fresh_u_with_pad**: Recovery error is large (rel L2 > 0.5) and subspace similarity is low; under this proxy, mask makes naive gradient extraction unreliable.

## 4. Gradient Membership-Style Linkability

| strategy | same_sample_grad_dist | different_sample_grad_dist | AUC_proxy | gradient_linkability_rank | risk_level |
|----------|------------------------|-----------------------------|-----------|----------------------------|------------|
| unmasked_gradient_baseline | 0.000 | 0.772 | 1.000 | 1.000 | high |
| fixed_masks_fixed_u | 0.000 | 0.772 | 1.000 | 1.000 | high |
| fresh_u_only | 0.844 | 0.855 | 0.541 | 0.082 | low |
| fresh_masks_fresh_u | 0.846 | 0.854 | 0.522 | 0.045 | low |
| fresh_masks_fresh_u_with_pad | 1.199 | 1.212 | 0.533 | 0.066 | low |

## 5. Gradient Leakage Accounting

Representative strategy: **fresh_masks_fresh_u_with_pad**.

| variable | visibility | plaintext | leakage_risk | mitigation | stage_7_1_status |
|----------|------------|-----------|--------------|------------|------------------|
| private_input_X | trusted | True | low | Right-mask + optional input pad before backward. | covered |
| private_target_Y | trusted | True | low | Stays trusted; loss G = dL/dY computed inside trusted side. | covered |
| adapter_A / adapter_B | trusted | True | low | Adapter masked by A_tilde / B_tilde; fresh U per call. | covered |
| grad_A / grad_B (plain) | trusted | True | low | Recovered only on trusted side after multiplying by N_in^{-T} / U^{-T} / N_out^T. | trusted |
| G (plain upstream gradient) | trusted | True | low | Computed from trusted Y_recovered and Y_target. | trusted_loss |
| G_tilde (masked upstream gradient) | gpu | False | low | G_tilde = G N_out^{-T}; fresh N_out per call. | covered |
| grad_A_tilde / grad_B_tilde | gpu | False | low | Masked by U / N_in / N_out; fresh U per call. | covered |
| X_tilde A_tilde (intermediate) | gpu | False | low | Lives in the rank space of U; recovered grad_A / grad_B never surfaced on GPU. | covered |
| optimizer_state (SGD momentum / AdamW m, v) | trusted | True | low | Trusted-only; never exported to JSON/CSV/Markdown. | trusted_optimizer |
| masks N_in / N_out / U / pad T_in | trusted | True | low | Sampled inside trusted side; never exported. | covered |
| X_tilde / Y_tilde (forward, inherited from Stage 7.0) | gpu | False | low | Right-mask N_in / N_out; fresh per call recommended. | covered |

Per-strategy variation lives in the JSON / CSV; the GPU-visibility contract is the same across strategies, only the *risk level* differs.

## 6. Interpretation

- **linkability_summary**: fresh masks reduce gradient-side membership linkability (Δ AUC = +0.478 vs fixed_masks_fixed_u).
- **rank_visibility_note**: LoRA rank r is visible from the shape of grad_A_tilde (d_in × r) and grad_B_tilde (r × d_out). Rank padding is NOT implemented in Stage 7.1.
- **loss_status**: trusted_loss
- **optimizer_status**: trusted_optimizer
- **merge_adapter_into_w**: False

## 7. Limitations

- These are gradient-side proxy attacks, not formal security proofs.
- Gradient tensors may leak rank (`grad_A_tilde` is (d_in × r); `grad_B_tilde` is (r × d_out)). Rank padding is NOT implemented in Stage 7.1 (deferred to Stage 7.2).
- Optimizer state remains trusted-only in Stage 7.1; SGD momentum / AdamW (m, v) are never exposed to the GPU.
- Loss / upstream gradient computation remains trusted-only; only G_tilde crosses the boundary.
- No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.
- No full model LoRA fine-tuning is evaluated; this is a single-linear, tiny-dimension gradient proxy.
- Adversary model is a passive GPU observer of (X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde, G_tilde, grad_A_tilde, grad_B_tilde) transcripts plus the dimensions; active / adaptive attackers and hardware side-channels are NOT evaluated.
- Adapter is NEVER merged into the public base weight W.

## 8. Next Stage Plan

- Stage 7.2 — rank padding to hide r from gradient shape.
- Stage 7.3 — multi-layer + cross-layer adapter / gradient linkability proxy.
