# Stage 7.4 — Stronger Dummy LoRA Security Proxy

## 1. Experiment Scope

- d_in=32, d_out=16, true_ranks=(2, 4, 8), padded_rank=16, alpha=1.0.
- num_trials=24, num_lora_modules_for_linkage=4, use_pad=True, dtype=float64.
- dummy_strategies=('zero_dummy', 'paired_cancellation_dummy', 'gaussian_matched_dummy', 'spectrum_matched_dummy', 'noise_injected_cancellation_dummy', 'orthogonalized_cancellation_dummy', 'mixed_dummy_ensemble'), dummy_scale=1.0, noise_scale=0.001, spectrum_match_strength=1.0.
- scope: stronger-dummy LoRA security proxy: spectral inference, gradient inference, dummy strategy classification, cross-layer linkage

## 2. Threat Model

- Passive GPU observer of the rank-padded transcript: (X_tilde, W_tilde, A_pad_tilde, B_pad_tilde, Y_tilde, G_tilde, grad_A_pad_tilde, grad_B_pad_tilde).
- Knows model architecture, padded_rank, masking scheme; does NOT know N_in, N_out, U_pad, T, plaintext A / B, true_rank, dummy strategy choice, private (X, Y_target), optimizer state.
- No hardware side-channel and no active boundary attack.
- The dummy-strategy classifier is a generous attacker that sees per-bucket means; this is an upper bound, not a black-box attacker.

## 3. Spectral Rank Inference

| strategy | true_rank | cliff_acc | energy_acc | elbow_acc | ensemble_acc | risk |
|----------|-----------|------------|-------------|------------|---------------|------|
| zero_dummy | 2 | 1.00 | 1.00 | 1.00 | 1.00 | high |
| zero_dummy | 4 | 1.00 | 1.00 | 1.00 | 1.00 | high |
| zero_dummy | 8 | 1.00 | 0.88 | 1.00 | 1.00 | high |
| paired_cancellation_dummy | 2 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| paired_cancellation_dummy | 4 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| paired_cancellation_dummy | 8 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| gaussian_matched_dummy | 2 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| gaussian_matched_dummy | 4 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| gaussian_matched_dummy | 8 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| spectrum_matched_dummy | 2 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| spectrum_matched_dummy | 4 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| spectrum_matched_dummy | 8 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| noise_injected_cancellation_dummy | 2 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| noise_injected_cancellation_dummy | 4 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| noise_injected_cancellation_dummy | 8 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| orthogonalized_cancellation_dummy | 2 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| orthogonalized_cancellation_dummy | 4 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| orthogonalized_cancellation_dummy | 8 | 0.00 | 0.17 | 0.00 | 0.00 | needs_more_evaluation |
| mixed_dummy_ensemble | 2 | 0.00 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| mixed_dummy_ensemble | 4 | 0.00 | 0.42 | 0.00 | 0.00 | needs_more_evaluation |
| mixed_dummy_ensemble | 8 | 0.00 | 0.17 | 0.00 | 0.00 | needs_more_evaluation |

- Spectral rank inference with three detectors (SVD-cliff, 99%-energy, log-elbow) and a majority ensemble. Higher accuracy ⇒ true_rank more readable from the visible spectrum. Conservative verdicts per requirement 12.

## 4. Gradient Rank Inference

| strategy | true_rank | grad_A_acc | grad_B_acc | ensemble | risk |
|----------|-----------|-------------|-------------|----------|------|
| zero_dummy | 2 | 1.00 | 0.00 | 0.00 | high |
| zero_dummy | 4 | 1.00 | 0.00 | 0.00 | high |
| zero_dummy | 8 | 1.00 | 1.00 | 1.00 | high |
| paired_cancellation_dummy | 2 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| paired_cancellation_dummy | 4 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| paired_cancellation_dummy | 8 | 1.00 | 1.00 | 1.00 | high |
| gaussian_matched_dummy | 2 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| gaussian_matched_dummy | 4 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| gaussian_matched_dummy | 8 | 1.00 | 1.00 | 1.00 | high |
| spectrum_matched_dummy | 2 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| spectrum_matched_dummy | 4 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| spectrum_matched_dummy | 8 | 1.00 | 1.00 | 1.00 | high |
| noise_injected_cancellation_dummy | 2 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| noise_injected_cancellation_dummy | 4 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| noise_injected_cancellation_dummy | 8 | 1.00 | 1.00 | 1.00 | high |
| orthogonalized_cancellation_dummy | 2 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| orthogonalized_cancellation_dummy | 4 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| orthogonalized_cancellation_dummy | 8 | 1.00 | 1.00 | 1.00 | high |
| mixed_dummy_ensemble | 2 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| mixed_dummy_ensemble | 4 | 0.00 | 0.00 | 0.00 | needs_more_evaluation |
| mixed_dummy_ensemble | 8 | 1.00 | 1.00 | 1.00 | high |

- Gradient-side spectral rank inference. Gradients depend on (X, A, B, G); their spectrum may differ from the static adapter spectrum. Conservative verdicts per requirement 12.

## 5. Dummy Strategy Classification

- strategy_classification_accuracy: 0.476
- random_chance_baseline: 0.143
- risk_level: **medium**
- interpretation: Nearest-bucket-mean classifier on top-k normalised singular values of A_pad_tilde / B_pad_tilde. Higher accuracy ⇒ the visible spectrum carries enough signal to discriminate the trusted-side dummy strategy choice.

## 6. Cross-Layer Linkage

| strategy | layer_linkability_auc | retrieval_top1 | same_module_sim | different_module_sim | risk |
|----------|------------------------|-----------------|------------------|----------------------|------|
| zero_dummy | 0.481 | 0.115 | 0.804 | 0.749 | low |
| paired_cancellation_dummy | 0.536 | 0.172 | 1.057 | 1.102 | low |
| gaussian_matched_dummy | 0.499 | 0.167 | 0.433 | 0.444 | low |
| spectrum_matched_dummy | 0.540 | 0.161 | 0.360 | 0.407 | low |
| noise_injected_cancellation_dummy | 0.465 | 0.115 | 1.081 | 1.034 | low |
| orthogonalized_cancellation_dummy | 0.459 | 0.104 | 1.166 | 1.041 | low |
| mixed_dummy_ensemble | 0.519 | 0.156 | 3.575 | 3.756 | low |

- Cross-layer linkage proxy under fresh masks per module. AUC ≈ 0.5 ⇒ same-module / different-module distance distributions overlap; retrieval_top1 near 1/(num_modules - 1) ⇒ no systematic linkage. Stronger dummies aim to bring both statistics closer to the random baseline.

## 7. Interpretation

- **spectral_summary**: Worst spectral rank inference risk across strategies: **high**.
- **gradient_summary**: Worst gradient rank inference risk across strategies: **high**.
- **dummy_strategy_classification_summary**: Dummy strategy classifier accuracy: 0.476 (chance 0.143), risk **medium**.
- **cross_layer_linkage_summary**: Worst cross-layer linkage risk across strategies: **low**.
- **true_rank_shape_hidden_when_padded**: True
- **padded_rank_visibility_note**: padded_rank itself remains visible from tensor shape; hiding it is out of Stage 7.4 scope.
- **merge_adapter_into_w**: False

## 8. Limitations

- These are stronger-dummy proxy attacks, not formal security proofs.
- Padded rank r_pad remains visible from tensor shape — only true_rank is hidden.
- Spectral hardening does not imply cryptographic hiding.
- Dummy strategy classification uses a generous bucket-mean attacker model.
- No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.
- Optimizer state remains trusted-only and is sized to true_rank.
- Hardware side-channels (cache / power / EM) are NOT evaluated.
- No full Qwen / TinyLlama / LLaMA LoRA fine-tuning is evaluated; this is a single-linear + cross-layer proxy.
- Adapter is NEVER merged into the public base weight W.

## 9. Next Stage Plan

- Stage 7.5 — paper artifact consolidation + projected vs measured runtime alignment.
- Stage 7.x — heterogeneous padded_rank across modules / layers to hide padded_rank itself.
