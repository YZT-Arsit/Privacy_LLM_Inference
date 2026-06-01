# Adaptive Permutation / Linkability Attacks — Stage 5.4

## Experiment Scope

- hidden_size: 64; num_train_samples: 512; num_test_samples: 256
- num_sessions: 16; samples_per_session: 32; permutation_pool_size: 4
- attacker_steps: 200; attacker_lr: 0.01; mlp_hidden_size: 128
- soft_assignment_iters: 50; soft_assignment_temperature: 0.05
- dtype: float32; device: cpu; seed: 2026

## Threat Model

Adaptive-proxy adversary: observes the GPU-visible nonlinear-island tensor across many sessions and has a labelled (visible, plaintext) training pool. May fit a ridge-regularised linear inverter, a small MLP inverter, and a Sinkhorn-style soft-assignment over per-channel signatures. Does NOT see trusted-side tensors, does NOT use side channels, does NOT query an actual deployed LLM, and does NOT have formal-security guarantees.

## Structured Synthetic Activation Distribution

- channel_mean_range: [-2.0, 2.0]
- channel_scale_range: [0.5, 3.0]
- channel_skew_profile: 0.5 * cos(linspace(0, pi, hidden))
- distribution_summary: `X[:, j] = mean_j + scale_j * (noise + skew_j * (noise**2 - 1))`

## Learned Linear Inverter

| strategy | relative_l2_error | cosine_similarity | mse |
|---|---|---|---|
| `fixed_permutation` | 0.0000 | 1.0000 | 2.5233e-12 |
| `fresh_permutation_per_session` | 1.1134 | 0.0899 | 7.5130e+00 |
| `permutation_pool` | 0.7545 | 0.6624 | 3.4404e+00 |
| `dense_sandwich` | 1.1550 | -0.0086 | 8.3240e+00 |
| `boundary_pad_only_boundary_view` | 1.0959 | 0.0583 | 7.4294e+00 |
| `boundary_pad_only_activation_view` | 0.0000 | 1.0000 | 2.7464e-12 |
| `fresh_perm_plus_sandwich_plus_pad` | 1.1347 | -0.0148 | 7.7025e+00 |

Weakest mitigation under linear inverter: `fixed_permutation`.

## Small MLP Inverter

| strategy | relative_l2_error | cosine_similarity | final_train_loss | mlp_improves_over_linear |
|---|---|---|---|---|
| `fixed_permutation` | 0.0963 | 0.9955 | 2.7274e-02 | False |
| `fresh_permutation_per_session` | 1.1923 | 0.2383 | 1.4388e+00 | False |
| `permutation_pool` | 0.6599 | 0.7669 | 7.8340e-01 | True |
| `dense_sandwich` | 1.2396 | 0.1820 | 1.4791e+00 | False |
| `boundary_pad_only_boundary_view` | 1.1809 | 0.2335 | 1.5188e+00 | False |
| `boundary_pad_only_activation_view` | 0.1162 | 0.9934 | 1.7073e-02 | False |
| `fresh_perm_plus_sandwich_plus_pad` | 1.2090 | 0.2008 | 1.5136e+00 | False |

## Adaptive Permutation Recovery

### Signature matching (Stage 5.2b naive nearest-neighbour proxy)

| strategy | top1_recovery_rate | top5_recovery_rate | mean_correct_rank |
|---|---|---|---|
| `fixed_permutation` | 0.1719 | 0.7031 | 3.66 |
| `fresh_permutation_per_session` | 0.2656 | 0.7812 | 3.42 |
| `permutation_pool` | 0.2188 | 0.6719 | 3.78 |
| `dense_sandwich` | 0.0000 | 0.0938 | 32.89 |
| `fresh_perm_plus_sandwich_plus_pad` | 0.0156 | 0.0781 | 31.12 |

### Soft assignment (Sinkhorn-style log-domain normalisation)

| strategy | top1_recovery_rate | top5_recovery_rate | mean_correct_rank |
|---|---|---|---|
| `fixed_permutation` | 0.1562 | 0.6250 | 4.55 |
| `fresh_permutation_per_session` | 0.2344 | 0.7188 | 3.36 |
| `permutation_pool` | 0.1406 | 0.6719 | 4.31 |
| `dense_sandwich` | 0.0312 | 0.0938 | 32.48 |
| `fresh_perm_plus_sandwich_plus_pad` | 0.0469 | 0.0938 | 28.95 |

Random chance top1 = 0.0156 (1 / hidden_size).

## Mitigation Decision Table

| strategy | best_linear_rel_l2 | best_mlp_rel_l2 | best_perm_top1 | risk_level | default_on_recommendation |
|---|---|---|---|---|---|
| `fixed_permutation` | 0.0000 | 0.0963 | 0.1719 | high | `unsafe_default_on` |
| `fresh_permutation_per_session` | 1.1134 | 1.1923 | 0.2656 | medium | `needs_more_evaluation` |
| `permutation_pool` | 0.7545 | 0.6599 | 0.2188 | medium | `needs_more_evaluation` |
| `dense_sandwich` | 1.1550 | 1.2396 | 0.0312 | low | `acceptable_with_mitigation` |
| `boundary_pad_only_boundary_view` | 1.0959 | 1.1809 | n/a | low | `acceptable_with_mitigation` |
| `boundary_pad_only_activation_view` | 0.0000 | 0.1162 | 0.1719 | high | `unsafe_default_on` |
| `fresh_perm_plus_sandwich_plus_pad` | 1.1347 | 1.2090 | 0.0469 | low | `acceptable_with_mitigation` |

Recommended default-on candidate: `fresh_permutation + dense_sandwich + pad at Linear boundaries`.

Safe-to-default-on only means "within the tested adaptive proxy attackers (ridge linear inverter, small MLP inverter, Sinkhorn-style permutation recovery)". This is NOT a formal security claim and NOT a TEE measurement.

### Required mitigations per strategy

- **`fixed_permutation`**:
  - do not deploy without per-session fresh permutation
  - must add a dense sandwich at Linear boundaries
  - must pad at Linear boundaries
- **`fresh_permutation_per_session`**:
  - must combine with pad at Linear boundaries
  - should add a dense sandwich on at least one side
  - short island lifetime + per-session rotation
- **`permutation_pool`**:
  - pool size must be large; treat as fixed_permutation for small pools
  - rotate the pool frequently
  - must combine with pad and dense sandwich
- **`dense_sandwich`**:
  - still requires fresh permutation per session
  - still requires pad at Linear boundaries
  - not formally secure; default-on only under tested adaptive proxy
- **`boundary_pad_only_boundary_view`**:
  - pad protects the boundary view only
  - must combine with dense sandwich or fresh permutation for activation view
- **`boundary_pad_only_activation_view`**:
  - boundary pad does NOT protect this view
  - must replace with fresh permutation + dense sandwich
- **`fresh_perm_plus_sandwich_plus_pad`**:
  - fresh permutation per session is mandatory
  - dense sandwich on both sides of the permutation island is mandatory
  - pad must remain at Linear boundaries only — never pushed through the activation
  - remains gated behind the ``nonlinear_mode`` feature flag

## Comparison with Stage 5.2b Naive Proxy

| strategy | naive_signature_matching_top1 | adaptive_soft_assignment_top1 | absolute_uplift |
|---|---|---|---|
| `fixed_permutation` | 0.1719 | 0.1562 | -0.0156 |
| `fresh_permutation_per_session` | 0.2656 | 0.2344 | -0.0312 |
| `permutation_pool` | 0.2188 | 0.1406 | -0.0781 |
| `dense_sandwich` | 0.0000 | 0.0312 | +0.0312 |
| `fresh_perm_plus_sandwich_plus_pad` | 0.0156 | 0.0469 | +0.0312 |

Stage 5.4 reproduces Stage 5.2b's signature-matching proxy on the same data and compares it against the Sinkhorn-style soft-assignment adaptive attacker. Larger uplift means the adaptive attacker is strictly stronger on that strategy.

## Limitations

- These are adaptive/proxy attacks, not formal security proofs.
- The attacks use synthetic structured channel data, not full real-model activation traces.
- No adaptive black-box querying of a deployed LLM is implemented.
- No side-channel attack is implemented.
- No real TEE isolation is evaluated.
- Dense sandwiching reduces tested recovery but does not imply semantic security.
- Default-on recommendations are conditional on the tested threat model only.
- These are adaptive/proxy attacks, not formal security proofs.
- Dense sandwiching reduces tested recovery but does not imply semantic security.
- Default-on recommendations are conditional on the tested threat model only.

## Next Stage Plan

- Stage 6.4 — Qwen / TinyLlama migration. Reuse the Stage 5.3a / 5.3b / 5.3c wrapper / probe pattern; behind a feature flag, default `trusted`.
- Stage 5.3d (deferred) — Full BERT / T5 obfuscated wrappers; only landed once an adaptive attacker against the chosen mitigation bundle (fresh permutation + dense sandwich + pad at Linear boundaries) is bounded below the agreed acceptance budget.
