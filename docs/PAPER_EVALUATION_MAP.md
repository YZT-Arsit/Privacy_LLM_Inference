# Paper Evaluation Map

This document maps each theorem and claim from
`docs/PAPER_THEORY_OUTLINE.md` to the artifacts that exercise it,
and grades the support level using a fixed five-tier vocabulary.

## Grading vocabulary

| grade | meaning |
|---|---|
| `algebraically_proven` | The claim is a direct algebraic identity under the stated assumptions. The artifact verifies the identity numerically at machine precision; the proof itself is in `docs/PAPER_THEORY_OUTLINE.md`. |
| `experimentally_validated` | The claim is verified by a measured experiment over a defined input distribution (synthetic and/or real-token), with conservative thresholds and labels. |
| `proxy_evaluated` | The claim is supported only by a named-attacker proxy evaluation. No formal / cryptographic / semantic security is claimed. |
| `cost_proxy_only` | The claim is a *cost* comparison (bandwidth, table size, microbenchmark). No security primitive is implemented. |
| `unsupported` | The claim is not supported by this artifact. The artifact records the limitation explicitly (e.g., `unsupported`, `not_implemented`, or an explicit raise). |

No grade in this table implies formal, cryptographic, or semantic
security. Where a claim has multiple aspects (e.g., correctness +
known-leakage), each aspect is graded separately.


## Claim → artifact map

| ID | Claim | Primary artifact(s) | Secondary artifact(s) | Grade(s) |
|---|---|---|---|---|
| T1 | Right-action mask correctness of linear layers | `outputs/ours_runtime_api_validation.json` (linear with pad row) | `paper_results/markdown/correctness_summary.md` | `algebraically_proven` |
| T2 | Boundary pad cancellation | `outputs/ours_runtime_api_validation.json` (linear with pad compensation row) | `outputs/lora_protocol_integration.json` | `algebraically_proven` |
| T3 | Attention QK score invariance under per-head right-action masks | `outputs/attention_experiments.json` | `outputs/cross_attention_experiments.json`, `outputs/encoder_attention_experiments.json` | `algebraically_proven` |
| T4 | KV-cache append invariance (per-call mask binding) | `outputs/ours_runtime_api_validation.json` (prefill / decode_step / greedy-generation rows) | `outputs/modern_decoder_model_wrapper_smoke.json`, `outputs/modern_decoder_generation_correctness.json` | `algebraically_proven` |
| T5 | RMSNorm commutes with orthogonal residual mask | `outputs/modern_decoder_model_wrapper_smoke.json` | `paper_results/markdown/correctness_summary.md` | `algebraically_proven` |
| T6 | Post-RoPE per-head masking preserves T3 | `outputs/modern_decoder_model_wrapper_smoke.json` | `outputs/attention_experiments.json` (modern-decoder probes) | `algebraically_proven` |
| T7 | Grouped-query attention with per-K/V mask tying | `outputs/modern_decoder_model_wrapper_smoke.json` | `outputs/attention_experiments.json` (GQA probes) | `algebraically_proven` |
| T8 | SwiGLU compatible-island correctness (per call) | `outputs/modern_decoder_model_wrapper_smoke.json` (SwiGLU rows) | `outputs/ours_runtime_api_validation.json` (nonlinear island row) | `algebraically_proven` |
| T8a | Compatible island preserves permutation-invariant statistics (known leakage) | `outputs/permutation_invariant_leakage.json` | `outputs/real_activation_attacks.json`, `outputs/real_token_activation_attacks.json` | `experimentally_validated` |
| T9 | Permutation-invariant statistics theorem | `outputs/permutation_invariant_leakage.json` | `outputs/real_activation_attacks.json`, `outputs/real_token_activation_attacks.json` | `algebraically_proven` + `experimentally_validated` |
| T9-attacker | Named-attacker risk under fresh-perm bundle (linear inverter, MLP inverter, signature-matching permutation recovery, linkability) | `outputs/real_activation_attacks.json`, `outputs/real_token_activation_attacks.json` | `outputs/stronger_attackers.json` | `proxy_evaluated` |
| T9-fixed-perm-ablation | Fixed-perm baseline shows attacker DOES recover (sanity reference) | `outputs/real_activation_attacks.json` (`fixed_permutation_debug` bundle) | `outputs/stronger_attackers.json` | `experimentally_validated` |
| T10 | Lookup table size scales as `2^(2b)`; cost baseline only | `outputs/lookup_nonlinear_cost_proxy.json` | `outputs/lookup_nonlinear_cost_proxy.csv`, `outputs/lookup_nonlinear_cost_proxy.md` | `cost_proxy_only` |
| T11 | Rank-space-mixed masked LoRA forward correctness | `outputs/masked_gradient_lora_training.json` (`forward_max_abs_err`) | `outputs/lora_training_inference_lifecycle.json` (`masked_forward` rows) | `algebraically_proven` |
| T12 | Gradient relations `grad_A_tilde = N_x^T grad_A M`, `grad_B_tilde = M^T grad_B N_y`, and masked-loss equality | `outputs/masked_gradient_lora_training.json` (grad relation + loss rows) | `outputs/lora_training_inference_lifecycle.json` (`masked_backward` rows) | `algebraically_proven` |
| T13 | Masked-SGD / masked-momentum-SGD equivalence to plaintext under recovery | `outputs/masked_gradient_lora_training.json` (SGD / momentum recovery rows) | `outputs/lora_training_inference_lifecycle.json` (`masked_sgd`, `masked_momentum_sgd` rows) | `algebraically_proven` |
| T13-leakage | GPU-visible LoRA gradients: true-rank inference, real-vs-dummy subspace, cross-step linkability | `outputs/masked_gradient_lora_security_proxy.json` | `outputs/lora_stronger_dummy_security_proxy.json` | `proxy_evaluated` |
| T14 | Cancellation-padded rank: `A_pad B_pad = A_real B_real` at init | `outputs/masked_gradient_lora_training.json` (`initial_dummy_contribution_norm`) | `outputs/lora_stronger_dummy_experiments.json` | `algebraically_proven` |
| T15 | AdamW dense-mask exactness is unsupported (explicit raise) | `outputs/masked_gradient_lora_training.json` (`adamw_dense_mask_unsupported.status = "explicitly_raised_as_designed"`) | `outputs/lora_training_inference_lifecycle.json` (`masked_adamw_unsupported` row) | `algebraically_proven` (limitation) + `experimentally_validated` (gate) |
| C-PaperSafe | Paper-safe wording: no formal / cryptographic / semantic security | `outputs/stage_7_6_claims_consistency.json` (`passes_consistency_check = True`, `total_unsafe_wording_present = 0`) | `outputs/paper_claims_audit_v2.json`, `paper_results/markdown/paper_claims_audit.md` | `experimentally_validated` (lexical scan) |


## Per-artifact summary

### attention_experiments.{json,csv,md}
- Supports T3 (QK score invariance, per-head right-action masks).
- Modern-decoder probes also support T6 (post-RoPE masking) and T7 (GQA tying).
- Grade contribution: `algebraically_proven` for T3/T6/T7.

### modern_decoder_model_wrapper_smoke.{json,md}
- Supports T4 (KV-cache append invariance through prefill + decode_step + greedy generation), T5 (RMSNorm commutation), T6 (post-RoPE masking), T7 (GQA tying), T8 (SwiGLU compatible-island correctness).
- Grade contribution: `algebraically_proven` for T4–T8.

### real_activation_attacks.{json,csv,md}
- Supports T8a / T9 known-leakage corollary by running the Stage 5.4 named attackers (ridge linear inverter, small MLP inverter, signature-matching + Sinkhorn permutation recovery, per-row linkability proxy) on real activations under both Stage 5.3e bundles plus a `fixed_permutation_debug` sanity reference.
- Risk thresholds: `linear_rel_l2 < 0.2` ⇒ high, `0.2 ≤ . < 0.6` ⇒ medium, otherwise low.
- Grade contribution: `proxy_evaluated` (named attackers); `experimentally_validated` (fixed-perm ablation showing attacker recovery).

### real_token_activation_attacks.{json,csv,md}
- Same attacker stack as `real_activation_attacks`, applied to real-token-driven activations (synthetic-by-default; tokenizer / real model loading opt-in).
- Grade contribution: `proxy_evaluated` (under real-token traces).

### stronger_attackers.{json,csv,md}
- Stronger / adaptive attacker variants (extended ridge with larger features, larger MLP inverters, adaptive permutation recovery probes).
- Grade contribution: `proxy_evaluated`.

### permutation_invariant_leakage.{json,csv,md}
- Supports T9 (permutation-invariant statistics theorem) and T8a (known leakage of compatible island).
- Per `(scope, tensor, bundle)`: row-wise norm preservation (l1/l2/linf), extrema preservation, sorted multiset preservation, quantile preservation, scope classification on visible-only per-row features.
- Conservative risk: `statistical_leakage_detected_high` only when `l2_corr > 0.999` AND `sorted_mse_mean < 1e-6` AND `sorted_l2_rel_mean < 1e-3`.
- Records `formal_security_claim = False`.
- Grade contribution: `algebraically_proven` (the theorem) + `experimentally_validated` (the measurement).

### lookup_nonlinear_cost_proxy.{json,csv,md}
- Supports T10 (lookup table size + bandwidth scaling; CPU microbenchmark).
- Compares `compatible_swiglu_island_current`, `compatible_swiglu_full_bundle`, `lookup_swiglu_proxy_{4,6,8}bit`.
- Records `formal_security_claim = False`, `cryptographic_lookup_implemented = False`, `recommended_use = "cost-baseline-and-future-work-motivation"`.
- Lookup-proxy rows publish `implemented_security = "none_cost_proxy_only"` and `security_potential = "stronger_value_hiding_if_combined_with_secure_lookup_protocol"`.
- Grade contribution: `cost_proxy_only`.

### masked_gradient_lora_training.{json,csv,md}
- Supports T11 (forward correctness), T12 (gradient relations + loss equality), T13 (masked SGD / momentum SGD equivalence), T14 (cancellation padding), and T15 (AdamW gate).
- Numerical envelope: `forward_max_abs_err <= 4.66e-15`, `loss_abs_err <= 1.11e-16`, grad relations `<= 2.52e-15`, SGD/momentum recovery `<= 1.78e-15`, `initial_dummy_contribution_norm = 0` (paired cancellation).
- AdamW gate: `adamw_dense_mask_unsupported.status = "explicitly_raised_as_designed"`.
- Grade contribution: `algebraically_proven` for T11–T14; `algebraically_proven` (limitation) + `experimentally_validated` (gate) for T15.

### masked_gradient_lora_security_proxy.{json,csv,md}
- Supports T13-leakage: GPU-visible parameters / gradients are probed by (i) true-rank inference from `A_tilde / B_tilde / grad_A_tilde / grad_B_tilde` spectra, (ii) real-vs-dummy subspace separation via largest-spectral-gap analysis, (iii) cross-step linkability under fresh-masks-per-step vs fixed-masks baseline.
- Conservative labels: `low_proxy_risk` / `medium_proxy_risk` / `high_proxy_risk` / `needs_more_evaluation`.
- Records `formal_security_claim = False`.
- Grade contribution: `proxy_evaluated`.

### paper_claims_audit (.md / .json from paper_results/, paper_claims_audit_v2.{json,md} from outputs/, and stage_7_6_claims_consistency.{json,csv,md})
- Supports C-PaperSafe: enforces that the artifact does not claim formal, cryptographic, or semantic security.
- Stage 7.6 claims-consistency scan: `total_unsafe_wording_present = 0`, `passes_consistency_check = True`.
- Tracks `formal security`, `cryptographically secure`, `semantic security`, `AdamW supported`, `plaintext gradients hidden by proof`, `optimizer fully outsourced`, `LoRA rank is hidden`.
- Grade contribution: `experimentally_validated` (lexical scan).


## Cross-cutting honesty constraints

- No claim in this map is marked higher than the artifact supports.
- `algebraically_proven` is reserved for direct algebraic identities verified at machine precision; it is *not* a security claim.
- `experimentally_validated` is reserved for measured behaviour over a defined input distribution; it is *not* a security claim.
- `proxy_evaluated` is reserved for named-attacker probes; it is *not* a security claim.
- `cost_proxy_only` is reserved for table-size / bandwidth / microbenchmark cost comparisons; it is *not* a security claim.
- `unsupported` is used wherever the artifact explicitly declines to support a claim (e.g., dense masked AdamW exactness).

`formal_security_claim`: `False` in every artifact that records the field.
