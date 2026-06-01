# Stronger Attackers (Stage 5.6)

## Experiment Scope

Stage 5.6 ships three proxy attackers that DO NOT require paired plaintext / visible internal supervision: (1) a **black-box query attacker** that sees only the generated token sequence + per-step logits summaries — no internal traces; (2) a **model-based timing / boundary-call side-channel proxy** driven by Stage 5.2c op-count formulas + the Stage 5.2c cost model + Gaussian noise — NOT a real TEE wall-time measurement; and (3) an **inter-block residual masking gap analysis** that quantifies the Stage 6.4c structural plain-boundary issue surfaced by Stage 5.5b and runs a single-transition math probe showing the orthogonal-mask fix is numerically correct.

## Threat Model

- **Black-box attacker** sees: generated tokens, per-step top-1 / top-5 token IDs, entropy, max logit, top-1 margin, output length. Does NOT see internal hidden states, masks, permutations, KV cache contents, or any `per_layer_traces`.
- **Timing attacker** sees: simulated per-forward latency under a Stage 5.2c cost model. Does NOT see real TEE wall-time, real GPU utilisation, real CPU/cache counters, or hardware-level side channels.
- **Inter-block analyst** reads the Stage 5.5b artifact and verifies the structural plain-boundary finding for the Stage 6.4c model wrapper. Does NOT execute any new wrapper code.
- All attackers are PROXY attackers, not formal security proofs.

## Black-Box Query Attacker

**Prompt linkability** (same vs different prompt via signatures):

| metric | value |
|---|---|
| same_prompt_similarity | 1.0000 |
| different_prompt_similarity | 0.6512 |
| linkability_auc_proxy | 1.0000 |
| nearest_prompt_retrieval_top1 | 1.0000 |
| nearest_prompt_retrieval_random_chance | 0.0667 |

> Same-prompt similarity is trivially 1.0 because greedy generation is deterministic — that's a property of greedy decoding, NOT a finding against the obfuscation envelope.

**Prompt class inference** (4-way nearest-neighbour classifier):

| metric | value |
|---|---|
| class_accuracy | 1.0000 |
| random_chance_baseline | 0.2667 |
| num_classes | 4 |
| classes | ['general_knowledge', 'generation_template', 'machine_learning', 'security'] |

**Mitigation-mode distinguishability** (nonlinear_mode × bundle × use_pad):

| metric | value |
|---|---|
| mode_classification_accuracy | 0.0000 |
| random_chance_baseline | 0.2258 |
| modes_observed | ['compatible_islands|fresh_perm_only|use_pad=False', 'compatible_islands|fresh_perm_only|use_pad=True', 'compatible_islands|fresh_perm_plus_sandwich_plus_pad|use_pad=False', 'compatible_islands|fresh_perm_plus_sandwich_plus_pad|use_pad=True'] |

> If `mode_classification_accuracy` ≈ `random_chance_baseline`, an API attacker cannot tell from the output which mitigation configuration produced it. Stage 6.4c verified that obfuscated and plain greedy outputs are byte-identical, so this metric is bounded at random chance by construction.

## Timing / Boundary-Call Side-Channel Proxy

Simulated latency = Stage 5.2c per-forward op-counts plugged into the Stage 5.2c cost model (`gpu_flops_per_ms`, `tee_to_gpu_flops_ratio`, `tee_call_overhead_ms`, `tee_bytes_per_ms`) + Gaussian noise with std = 0.05 × |mean|.

**This is NOT a real TEE wall-time measurement**; the wider profile keeps `wall_time_source = projected_from_op_counts` and `implemented = False`.

| sub-attack | accuracy | random_chance | correlation | risk |
|---|---|---|---|---|
| Prompt length | 0.402 | 0.249 | 0.101 | medium |
| Decode step | 0.749 | 0.249 | 0.790 | high |
| Method | 0.602 | 0.332 | n/a | medium |
| Mitigation bundle | 0.508 | 0.499 | n/a | low |

**Boundary-call pattern (static structural leakage):**

| method | per_forward_boundary_calls | formula |
|---|---|---|
| ours_current | 9 | L=2 formula: 4L + 1 = 9 |
| ours_compatible_nonlinear_islands | 4 | L=2 formula: L + 2 = 4 |
| tslp_trusted_nonlinear_baseline | 8 | L=2 formula: 3L + 2 = 8 |

## Inter-Block Residual Masking Gap

| field | value |
|---|---|
| current_plain_boundary_detected | True |
| affected_tensors | boundary_input, final |
| accounting_risk_level | medium |
| single_transition_probe_status | single_transition_probe_passed |
| masked_boundary_experimental_status | implemented_in_stage_5_6_extension |
| masked_boundary_experimental_default | off |
| overall_inter_block_risk_level | medium |

## Single-Transition Masking Probe

Math-only verification that an orthogonal inter-block mask `N_inter` is absorbed by the next block's RMSNorm + folded Q/K/V projection — no plain inter-block transcript required.

| metric | value |
|---|---|
| status | ok |
| rmsnorm_invariant_max_abs_error | 0.000000 |
| rmsnorm_invariant_allclose | True |
| q_projection_path_max_abs_error | 0.000006 |
| q_projection_path_allclose | True |
| residual_recovery_max_abs_error | 0.000000 |
| residual_recovery_allclose | True |

_Note_: Single-transition probe: orthogonal N_inter applied to one block boundary; rmsnorm_core is invariant under N_inter and the folded Q-projection (w_q_tilde = N_inter^T @ (γ ⊙ w_q)) reproduces the plain Q exactly. The attacker view of the inter-block residual is x_tilde = x @ N_inter (orthogonal, information-theoretically equivalent under random-N_inter sampling but with the same caveats as Stage 6.4b).

## Inter-Block Masking Mode

| field | value |
|---|---|
| status | implemented |
| masked_boundary_experimental_status | implemented |

### Plain Boundary vs Masked Boundary Experimental

| tensor | mode | risk | inter_block_plain | linear_rel_l2 | linkability_cosine |
|---|---|---|---|---|---|
| boundary_input | plain_boundary | high | True | 0.0313 | 1.0000 |
| boundary_input | masked_boundary_experimental | low | False | 1.9040 | -0.0145 |
| final | plain_boundary | high | True | 0.0132 | 1.0000 |
| final | masked_boundary_experimental | low | False | 1.6114 | 0.0200 |

### Boundary Input / Final Risk Before and After

_Note_: Head-to-head: plain_boundary keeps boundary_input / final as inter_block_plain_recovered (structural high). Stage 5.6 extension's masked_boundary_experimental mode rotates the inter-block residual with a fresh orthogonal N_inter so those tensors join the masked tensor set and the attacker's linear / MLP / linkability proxies fail.

## Constant-Time Decode Proxy

| field | value |
|---|---|
| mode | proxy_equalized |
| decode_step_accuracy_before | 0.7487 |
| decode_step_accuracy_after | 0.2539 |
| correlation_latency_step_before | 0.7899 |
| correlation_latency_step_after | -0.0003 |
| risk_level_before | high |
| risk_level_after | low |
| overhead_ms_estimate | 7.3667 |

### Decode-Step Timing Leakage Before and After

_Limitation_: proxy only — no real sleep, no real wall-time. The equalisation upper bound is the maximum simulated latency across (prompt_length × decode_step) bins per method; a real deployment would calibrate this from a production timing budget.

### Overhead Proxy

Mean per-step latency padding ≈ 7.367 ms (simulated). PROXY only — does not change real wall-time.

## Comparison with Stage 5.4 / 5.5 / 5.5b

Stage 5.4 — synthetic adaptive proxy. Stage 5.5 — real-activation (random hidden input) adaptive proxy. Stage 5.5b — real-token-prompted real-activation adaptive proxy across prefill + decode_step. Stage 5.6 — black-box query + timing side-channel proxy + inter-block masking gap analysis. Stages 5.4 / 5.5 / 5.5b all reported `low` risk for masked tensors under the full mitigation bundle; Stage 5.6 extends to attack surfaces that DON'T require paired plain/visible supervision.

| stage | artifact |
|---|---|
| stage_5_4_artifact | `outputs/adaptive_island_attacks.json` |
| stage_5_5_artifact | `outputs/real_activation_attacks.json` |
| stage_5_5b_artifact | `outputs/real_token_activation_attacks.json` |
| stage_5_6_artifact | `outputs/stronger_attackers.json` |

## Overall Risk Summary

| dimension | level |
|---|---|
| envelope_integrity_risk_level | low |
| envelope_blackbox_risk_level | low |
| envelope_timing_risk_level | low |
| structural_leakage_risk_level | high |
| structural_timing_risk_level | high |
| structural_inter_block_risk_level | medium |
| overall_risk_level | high |

> **Envelope-integrity risk** is what the mitigation envelope is responsible for: can mode / bundle / use_pad be distinguished from outputs or timing? If `low`, the envelope holds under black-box + timing proxy attacks.

> **Structural-leakage risk** captures known model-wrapper / transformer properties (latency scales with prompt length and decode step; the Stage 6.4c model wrapper recovers between blocks). These are acknowledged limitations, NOT failures of the mitigation envelope; closing them requires constant-time computation and inter-block masked residual (Stage 5.6 extension / Stage 7.0).

## Recommendation

- `security_profile_detail_with_stronger_attackers = "adaptive-blackbox-and-timing-proxy-evaluated, not formal"`
- `security_profile_detail_with_extended_proxy = "inter-block-and-constant-time-proxy-evaluated, not formal"`
- `extended_proxy_eligibility = "yes"`
- `overall_recommendation = "acceptable_with_mitigation_under_extended_proxy"`
- `promotion_eligibility_note = "yes — envelope-integrity risk is `low` (modes / bundles are statistically indistinguishable from API output AND from timing). Eligible to label `adaptive-blackbox-and-timing-proxy-evaluated, not formal`. Structural leakage (decode step, prompt length, inter-block plain boundary) is reported separately and is acknowledged as a known limitation of the current model wrapper, not a failure of the mitigation envelope."`
- `inter_block_residual_masking_recommendation = "Stage 5.6 extension wires masked_boundary_experimental into the model-wrapper prefill / decode_step / greedy generation path. Default remains plain_boundary; the experimental mode is opt-in via inter_block_mask_mode='masked_boundary_experimental'."`
- `constant_time_decode_recommendation = "constant_time_decode_mode='proxy_equalized' equalises per-step simulated latency to a per-method upper bound; PROXY ONLY (no sleep, no real wall-time change). Decode-step timing leakage is reduced under this proxy."`
- `inter_block_mask_mode_used = "masked_boundary_experimental"`
- `constant_time_decode_mode_used = "proxy_equalized"`
- `default_mode_unchanged = "plain_boundary"`
- `default_mitigation_bundle_unchanged = "fresh_perm_only"`
- `default_nonlinear_mode_unchanged = "trusted"`
- `default_constant_time_decode_mode_unchanged = "off"`

**Promotion eligibility:** yes — envelope-integrity risk is `low` (modes / bundles are statistically indistinguishable from API output AND from timing). Eligible to label `adaptive-blackbox-and-timing-proxy-evaluated, not formal`. Structural leakage (decode step, prompt length, inter-block plain boundary) is reported separately and is acknowledged as a known limitation of the current model wrapper, not a failure of the mitigation envelope.

## Limitations

- These are stronger proxy attacks, not formal security proofs.
- Black-box attacks only use generated outputs and logits summaries.
- Timing results are model-based proxies, not real TEE timing measurements.
- No hardware side-channel attack is implemented.
- No real TEE isolation is evaluated.
- Synthetic fallback results are not real Qwen/TinyLlama results.
- Inter-block masking is experimental unless explicitly marked implemented.
- Dense sandwiching and fresh permutation reduce tested recovery but do not imply semantic security.
- _Note_: stronger proxy attacks, not formal security proofs; timing results are model-based proxies, not real TEE timing measurements; inter-block residual masking gap remains open until Stage 5.6 extension / Stage 7.0; not a real TEE measurement; not formal security.

## Next Stage Plan

- Stage 5.6 extension (optional) — implement the full `masked_boundary_experimental` mode in ObfuscatedModernDecoderModelWrapper so the inter-block residual stays masked across layers under an orthogonal N_inter.
- Stage 7.0 (deferred) — LoRA private-training path under the same obfuscation envelope. Requires Stage 5.6 to first establish the inference-side security baseline.
- Constant-time mitigations to suppress decode-step latency leakage (timing) are deliberately deferred — they cost throughput and require explicit deployment-side opt-in.

