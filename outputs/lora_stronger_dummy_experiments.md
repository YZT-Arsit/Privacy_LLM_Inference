# Stage 7.4 — Stronger Dummy Distributions / Spectral-Rank Hardening

## 1. Experiment Scope

- Single LoRA-augmented linear with d_in=32, d_out=16.
- true_rank=4, padded_rank=16, alpha=1.0, batch_size=4.
- num_steps=5, optimizer=sgd, lr=0.01, use_pad=True, fresh_u_per_step=True, dtype=float64.
- dummy_scale=1.0, noise_scale=0.001, spectrum_match_strength=1.0.
- Synthetic private data; no network access; no PEFT integration.

## 2. Stronger Dummy Strategy Design

- All cancellation strategies preserve A_pad B_pad = A_real B_real exactly.
- noise_injected_cancellation_dummy tracks a small trusted-side correction = A_pad[:, r:] B_pad[r:, :] that the harness subtracts from the recovered output via (alpha / true_rank) X @ correction.
- spectrum_matched_dummy cycles singular values from the empirical A_real / B_real spectrum.
- gaussian_matched_dummy samples R / S from a Gaussian matched to per-column statistics of A_real / B_real.
- orthogonalized_cancellation_dummy projects R / S orthogonal to the column / row span of A_real / B_real.
- mixed_dummy_ensemble samples a per-pair strategy from the four cancellation strategies above.
- Spectral hardening does not imply cryptographic hiding.

- supported_strategies: ['zero_dummy', 'paired_cancellation_dummy', 'gaussian_matched_dummy', 'spectrum_matched_dummy', 'noise_injected_cancellation_dummy', 'orthogonalized_cancellation_dummy', 'mixed_dummy_ensemble']
- evaluated_strategies: ['zero_dummy', 'paired_cancellation_dummy', 'gaussian_matched_dummy', 'spectrum_matched_dummy', 'noise_injected_cancellation_dummy', 'orthogonalized_cancellation_dummy', 'mixed_dummy_ensemble']

## 3. Forward Correctness

| strategy | max_loss_diff | max_forward_err | max_dummy_contribution_norm | max_correction_norm | allclose |
|----------|---------------|------------------|------------------------------|----------------------|----------|
| zero_dummy | 5.684e-14 | 2.842e-14 | 0.000e+00 | 0.000e+00 | True |
| paired_cancellation_dummy | 6.395e-14 | 2.487e-14 | 2.910e-15 | 0.000e+00 | True |
| gaussian_matched_dummy | 1.421e-14 | 2.665e-14 | 5.405e-18 | 0.000e+00 | True |
| spectrum_matched_dummy | 2.842e-14 | 3.020e-14 | 7.059e-18 | 0.000e+00 | True |
| noise_injected_cancellation_dummy | 2.842e-14 | 1.954e-14 | 1.135e-01 | 1.135e-01 | True |
| orthogonalized_cancellation_dummy | 3.553e-14 | 2.220e-14 | 2.394e-15 | 0.000e+00 | True |
| mixed_dummy_ensemble | 2.842e-14 | 2.132e-14 | 1.672e-15 | 0.000e+00 | True |

## 4. Backward Correctness

| strategy | max_grad_A_real_err | max_grad_B_real_err | max_update_A_err | max_update_B_err |
|----------|----------------------|----------------------|------------------|------------------|
| zero_dummy | 1.110e-16 | 2.776e-15 | 2.776e-17 | 4.857e-17 |
| paired_cancellation_dummy | 2.528e-15 | 2.831e-15 | 5.551e-17 | 6.592e-17 |
| gaussian_matched_dummy | 1.110e-16 | 1.887e-15 | 5.551e-17 | 3.469e-17 |
| spectrum_matched_dummy | 1.943e-16 | 2.193e-15 | 5.551e-17 | 2.429e-17 |
| noise_injected_cancellation_dummy | 2.543e-15 | 3.136e-15 | 5.551e-17 | 4.857e-17 |
| orthogonalized_cancellation_dummy | 7.529e-16 | 2.220e-15 | 5.551e-17 | 4.337e-17 |
| mixed_dummy_ensemble | 7.754e-16 | 1.721e-15 | 5.551e-17 | 3.469e-17 |

## 5. Optimizer Handling

| strategy | trainable_a | trainable_b | optimizer_state_a | optimizer_state_b | dummy_in_state | dummy_updated |
|----------|-------------|-------------|--------------------|-------------------|----------------|----------------|
| zero_dummy | [32, 4] | [4, 16] | None | None | False | False |
| paired_cancellation_dummy | [32, 4] | [4, 16] | None | None | False | False |
| gaussian_matched_dummy | [32, 4] | [4, 16] | None | None | False | False |
| spectrum_matched_dummy | [32, 4] | [4, 16] | None | None | False | False |
| noise_injected_cancellation_dummy | [32, 4] | [4, 16] | None | None | False | False |
| orthogonalized_cancellation_dummy | [32, 4] | [4, 16] | None | None | False | False |
| mixed_dummy_ensemble | [32, 4] | [4, 16] | None | None | False | False |

## 6. Dummy Contribution and Correction

- Cancellation strategies maintain `A_pad B_pad = A_real B_real` exactly to float64 precision.
- `noise_injected_cancellation_dummy` carries a small trusted-side correction term that the harness subtracts via `(alpha / true_rank) X @ correction` from the recovered output.

- `noise_injected_cancellation_dummy`: max_dummy_contribution_norm = 1.135e-01, max_correction_norm = 1.135e-01; trusted-side correction is applied each step.

## 7. Comparison with Stage 7.2 / 7.3

- Stage 7.2 `paired_cancellation_dummy` keeps `dummy_contribution_norm = 0` exactly. The five Stage 7.4 stronger strategies preserve this property EXCEPT `noise_injected_cancellation_dummy`, which carries a tracked trusted-side correction.
- Per-step `max_loss_diff` / `max_grad_*_real_err` / `max_update_*_err` remain at float64 floor (≤ 1e-13) for every strategy — Stage 7.0 / 7.1 / 7.2 / 7.3 correctness regressions are checked separately by the existing test suites.
- Stage 7.2's `lora_rank_padding_status = "implemented"` / `lora_hidden_rank_status = "padded-rank-prototype"` are preserved. Stage 7.4 adds `lora_stronger_dummy_status = "implemented"` / `lora_spectral_rank_hardening_status = "proxy-evaluated"` as additive metadata.

## 8. Limitations

- Stronger dummy distributions are proxy-evaluated, not formal.
- Padded rank remains visible unless heterogeneous padded_rank is separately enabled.
- Spectral hardening does not imply cryptographic hiding.
- Optimizer state remains trusted-only and is sized to true_rank for every LoRA module.
- No real TEE training is evaluated; security_profile stays 'proxy-evaluated, not formal'.
- No full Qwen / TinyLlama / LLaMA LoRA fine-tuning is evaluated; this is a single-linear probe.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- Adapter is NEVER merged into the public base weight W.
- Reports publish summary metrics + fingerprints only; private data, raw adapters, raw gradients, optimizer state, and dense masks are never emitted.

## 9. Next Stage Plan

- Stage 7.5 — paper artifact consolidation + projected vs measured runtime alignment.
- Stage 7.x — heterogeneous padded_rank across modules / layers to hide padded_rank itself.
