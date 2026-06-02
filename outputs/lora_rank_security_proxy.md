# Stage 7.2 — LoRA Rank Security Proxy

## 1. Experiment Scope

- d_in=32, d_out=16, true_ranks=(2, 4, 8), padded_rank=16, alpha=1.0.
- num_trials=32, membership_trials_per_sample=8, dummy_strategy='paired_cancellation_dummy', use_pad=True, dtype=float64.
- scope: single linear + LoRA, tiny dimensions, synthetic adapter + synthetic upstream gradient, rank padding via dummy_strategy

## 2. Threat Model

- Passive GPU observer of the rank-padded transcript: (X_tilde, W_tilde, A_pad_tilde, B_pad_tilde, Y_tilde, G_tilde, grad_A_pad_tilde, grad_B_pad_tilde).
- Knows model architecture, padded_rank dimension, and the masking scheme; does NOT know N_in, N_out, U_pad, T, plaintext A / B / A_pad / B_pad / G / grad_A / grad_B, optimizer state, true_rank, or the private (X, Y_target).
- No hardware side-channel (cache / power / EM) and no active boundary attack.
- This is a *proxy*: ranks dummy strategies and padding levels under four sub-attacks. It does NOT prove security.

## 3. Shape-Level Rank Leakage

| strategy | true_rank | exposed_rank | true_rank_hidden_from_shape |
|----------|-----------|--------------|-----------------------------|
| no_padding | 2 | 2 | False |
| no_padding | 4 | 4 | False |
| no_padding | 8 | 8 | False |
| rank_padding | 2 | 16 | True |
| rank_padding | 4 | 16 | True |
| rank_padding | 8 | 16 | True |

- Without padding, the rank dimension of A_tilde / B_tilde / grad_A_tilde / grad_B_tilde equals the true LoRA rank r. With rank padding, the GPU-visible rank dimension is the padded rank r_pad and r is hidden from shape.

## 4. Spectral Rank Inference Proxy

| true_rank | inferred_no_padding | inferred_rank_padding_A_tilde | inferred_rank_padding_B_tilde | accuracy | risk |
|-----------|----------------------|-------------------------------|-------------------------------|----------|------|
| 2 | 1.00 | 9.00 | 9.00 | 0.00 | needs_more_evaluation |
| 4 | 2.28 | 10.00 | 10.00 | 0.00 | needs_more_evaluation |
| 8 | 6.09 | 12.00 | 12.00 | 0.00 | needs_more_evaluation |

- **true_rank=2**: Spectral attacker recovers true rank in fewer than 20% of trials. The inferred rank may still be an upper bound constrained by paired-cancellation structure; this is a proxy result, not a formal claim.
- **true_rank=4**: Spectral attacker recovers true rank in fewer than 20% of trials. The inferred rank may still be an upper bound constrained by paired-cancellation structure; this is a proxy result, not a formal claim.
- **true_rank=8**: Spectral attacker recovers true rank in fewer than 20% of trials. The inferred rank may still be an upper bound constrained by paired-cancellation structure; this is a proxy result, not a formal claim.

- Spectral rank inference threshold: largest σ_{i+1}/σ_i drop in the singular-value spectrum. The inferred rank is the index just before the cliff. For zero_dummy this aligns with the true rank exactly (B_pad has zero rows). For paired_cancellation_dummy the inferred rank is bounded by true_rank + ⌊(r_pad - r) / 2⌋ + leftover_zero.

## 5. Gradient Rank Inference Proxy

| true_rank | inferred_grad_A | inferred_grad_B | accuracy | risk |
|-----------|-----------------|-----------------|----------|------|
| 2 | 8.00 | 8.00 | 0.00 | needs_more_evaluation |
| 4 | 8.00 | 8.00 | 0.00 | needs_more_evaluation |
| 8 | 8.00 | 8.00 | 1.00 | high |

- Gradient-side spectral rank inference uses the same SVD-cliff detector but on grad_A_tilde_pad / grad_B_tilde_pad. Because gradients depend on (X, A, B, G), the spectrum may differ from the static adapter spectrum and warrants a separate section.

## 6. Membership / Linkability Proxy

| true_rank | same_sample_dist | different_sample_dist | AUC_proxy | linkability_rank | risk_level |
|-----------|-------------------|------------------------|-----------|--------------------|------------|
| 2 | 353.388 | 354.555 | 0.541 | 0.081 | low |
| 4 | 167.755 | 166.912 | 0.487 | 0.025 | low |
| 8 | 70.473 | 70.656 | 0.505 | 0.009 | low |

## 7. Interpretation

- **shape_level_summary**: Rank padding hides true_rank from tensor shape; padded_rank remains visible.
- **spectral_inference_summary**: Across true_ranks=[2, 4, 8] with dummy_strategy='paired_cancellation_dummy', the spectral rank inference risk is **needs_more_evaluation**.
- **gradient_inference_summary**: Gradient-side spectral inference risk under the same dummy strategy is **high**.
- **padded_rank_visibility_note**: Padded rank r_pad is still visible from tensor shape; hiding it is out of Stage 7.2 scope.
- **merge_adapter_into_w**: False

## 8. Limitations

- These are rank-leakage proxy attacks, not formal security proofs.
- True rank is hidden only from shape-level leakage when rank padding is enabled. Padded rank r_pad remains visible.
- Spectral / statistical rank inference may still be possible depending on dummy strategy. zero_dummy explicitly leaks true rank via SVD of B_pad.
- paired_cancellation_dummy reduces obvious zero-norm leakage but the spectral upper bound `true_rank + ⌊(r_pad - r) / 2⌋` still narrows the attacker's range; this is reported as needs_more_evaluation, not low.
- No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.
- Optimizer state remains trusted-only and is sized to true_rank.
- Hardware side-channels (cache / power / EM) are NOT evaluated.
- No full model LoRA fine-tuning is evaluated; this is a single-linear, tiny-dimension proxy.
- Adapter is NEVER merged into the public base weight W.

## 9. Next Stage Plan

- Stage 7.3 — multi-layer LoRA + LoRA training timing-side proxy.
- Stage 7.x — stronger dummy distributions that resist spectral and gradient rank inference.
- Stage 7.x — explore hiding padded_rank itself (e.g. tiled padding across multiple linears).
