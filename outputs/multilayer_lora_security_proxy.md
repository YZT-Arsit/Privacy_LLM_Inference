# Stage 7.3 — Multi-Layer LoRA Cross-Layer Security Proxy

## 1. Experiment Scope

- num_layers=2, hidden_size=32, intermediate_size=64.
- true_ranks=(2, 4), padded_rank=8, alpha=1.0, num_trials=32.
- dummy_strategy='paired_cancellation_dummy', use_pad=True, dtype=float64.
- scope: multi-layer tiny LoRA model (q/k/v/o/gate/up/down per layer), synthetic adapters + synthetic upstream gradients, rank padding optional per strategy

## 2. Threat Model

- Passive GPU observer across multiple LoRA-augmented linears in a multi-layer block stack.
- Sees per-module ``A_pad_tilde`` / ``B_pad_tilde`` / ``grad_A_pad_tilde`` / ``grad_B_pad_tilde``.
- Knows model architecture, padded_rank dimension, masking scheme; does NOT know per-module N_in / N_out / U_pad / T, plaintext A / B / A_pad / B_pad / G / grad_A / grad_B, optimizer state, true_rank, or private (X, Y_target).
- No hardware side-channel and no active boundary attack.
- This is a *proxy*: ranks per-module strategies; no formal claim.

## 3. Cross-Layer Adapter Linkage

| strategy | layer_linkability_auc | retrieval_top1 | same_module_sim | different_module_sim | risk |
|----------|-----------------------|-----------------|------------------|----------------------|------|
| fixed_masks_shared_u | 0.468 | 0.304 | 0.142 | 0.134 | high |
| independent_u_per_layer | 0.502 | 0.321 | 0.134 | 0.131 | medium |
| fresh_masks_independent_u | 0.514 | 0.315 | 0.133 | 0.135 | medium |
| rank_padding_full_bundle | 0.629 | 0.324 | 1.223 | 2.425 | medium |

- Cross-layer adapter linkage proxy. AUC > 0.5 means the attacker's same-module distance distribution is smaller than the different-module distribution, so transcript fingerprints correlate with module identity across layers. fixed_masks_shared_u is the worst case; rank_padding_full_bundle is the most defensive setting tested here, but no result below is formally secure.

## 4. Heterogeneous True Rank with Shared Padded Rank

| layer | module | true_rank | padded_rank | visible_rank | shape_hidden_rate | spectral_acc | gradient_acc | risk |
|-------|--------|-----------|-------------|---------------|--------------------|--------------|--------------|------|
| 0 | q_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 0 | k_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 0 | v_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 0 | o_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 0 | gate_proj | 4 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 0 | up_proj | 4 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 0 | down_proj | 4 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | q_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | k_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | v_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | o_proj | 2 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | gate_proj | 4 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | up_proj | 4 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |
| 1 | down_proj | 4 | 8 | 8 | 1.00 | 0.00 | 0.00 | needs_more_evaluation |

- Heterogeneous true_rank with a shared padded_rank. Shape-level leakage is fully closed in all rows when padded_rank is the same across modules (true_rank_shape_hidden_rate == 1.0). Spectral / gradient inference is reported per module under dummy_strategy='paired_cancellation_dummy'; paired_cancellation yields needs_more_evaluation, zero_dummy is high.

## 5. Multi-Step Membership Linkability

| module | same_sample_dist | different_sample_dist | AUC_proxy | linkability_rank | risk |
|--------|-------------------|------------------------|-----------|--------------------|------|
| q_proj | 323.505 | 323.284 | 0.498 | 0.004 | low |
| k_proj | 302.910 | 303.899 | 0.509 | 0.018 | low |
| v_proj | 330.102 | 328.947 | 0.487 | 0.026 | low |
| o_proj | 296.516 | 296.325 | 0.508 | 0.017 | low |
| gate_proj | 462.207 | 463.049 | 0.505 | 0.010 | low |
| up_proj | 448.889 | 448.267 | 0.497 | 0.007 | low |
| down_proj | 449.780 | 450.031 | 0.501 | 0.001 | low |

- mean_membership_auc_proxy: 0.501
- adapter_update_linkability: 0.001
- Per-module multi-step membership linkability proxy. Under fresh masks per call + paired_cancellation_dummy, the transcript distance distribution should be roughly the same for same-sample replays and different-sample comparisons, so AUC ≈ 0.5 and linkability_rank ≈ 0.

## 6. Interpretation

- **shape_level_summary**: true_rank is hidden from tensor shape across all modules when padded_rank is shared; padded_rank itself remains visible.
- **cross_layer_linkage_summary**: Cross-layer linkage risk under multi-layer strategies is **high** (worst-case).
- **heterogeneous_rank_summary**: Heterogeneous true_rank with shared padded_rank inference risk under dummy_strategy='paired_cancellation_dummy' is **needs_more_evaluation**.
- **true_rank_shape_hidden_rate**: 1.0
- **padded_rank_visibility_note**: padded_rank is still visible across modules; hiding it is out of Stage 7.3 scope.
- **merge_adapter_into_w**: False

## 7. Limitations

- These are cross-layer leakage proxy attacks, not formal security proofs.
- True rank is hidden from shape-level leakage only when padded_rank is shared across modules.
- padded_rank itself remains visible from tensor shape.
- Spectral / gradient rank inference may still narrow the attacker's range, especially under zero_dummy.
- Cross-layer linkability under fixed_masks_shared_u is reported high by construction; the no-mitigation baseline IS leaky.
- No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.
- Optimizer state remains trusted-only and is sized to true_rank for every module.
- Hardware side-channels (cache / power / EM) are NOT evaluated.
- No full Qwen / TinyLlama / LLaMA fine-tuning is evaluated; this is a multi-linear, tiny-dimension proxy.
- Adapter is NEVER merged into the public base weight W.

## 8. Next Stage Plan

- Stage 7.4 — stronger dummy distributions / spectral-rank hardening.
- Stage 7.x — heterogeneous padded_rank across modules to hide r_pad itself.
- Stage 7.x — gradient-side noise to weaken cross-layer linkability under shared-U setups.
