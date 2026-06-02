# Privacy LLM Obfuscation — Calibrated Workload Profile (Stage 5.0.1)

Cost model splits every method into four explicit slices: **preprocessing trusted cost** (amortised), **online boundary crossings** (true trusted↔untrusted round trips), **online trusted compute** (LayerNorm / GELU / sampling / recovery FLOPs in the TEE), and **online GPU obfuscated compute** (linear matmuls, attention, LM head). Internal Python bookkeeping such as mask-state creation is **not** counted as a boundary call.

`model_id=sshleifer/tiny-gpt2`, `batch_size=2`, `prompt_len=8`, `max_new_tokens=4`, `device=cpu`, `dtype=float32`, `use_pad=True`, `warmup=2`, `repeat=5`.

GPU-FLOPs/ms calibration constant: `1.821e+06` (derived from measured `plain_hf_gpu` wall time).

> **Warning:** simulated cost model, not real SGX.

## Method comparison
| method | impl? | wall_time_ms (measured/proj.) | boundary calls | boundary formula | trusted compute (ops) | trusted transfer (bytes) | gpu (ops) |
|---|---|---|---|---|---|---|---|
| plain_hf_gpu | true | 2.435 | 0 | 0 (no boundary) | 0 | 0 | 4434424 |
| tslp_trusted_nonlinear_baseline | false | 37.194 (proj.) | 32 | 3L + 2 = 8 per forward (LN_1 + LN_2 + GELU per layer + ln_f + LM head) | 1110230 | 4427192 | 4429848 |
| ours_current | true | 6.813 | 36 | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | 1116310 | 4428424 | 4429848 |
| ours_ideal_gpu_nonlinear | false | 36.927 (proj.) | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4422792 | 4434424 |
| ours_compatible_nonlinear_islands | false | 36.992 (proj.) | 16 | L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model) | 1105830 | 4423496 | 4434424 |
| amulet_style_reference | false | 36.927 (proj.) | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4422792 | 4434424 |

## Preprocessing (amortised; excluded from online latency)
| method | preprocessing_trusted_ops | preprocessing_transfer_bytes |
|---|---|---|
| plain_hf_gpu | 0 | 0 |
| tslp_trusted_nonlinear_baseline | 0 | 0 |
| ours_current | 403784 | 402440 |
| ours_ideal_gpu_nonlinear | 403784 | 402440 |
| ours_compatible_nonlinear_islands | 403992 | 402440 |
| amulet_style_reference | 403784 | 402440 |

## Interaction breakdown (online slice by interaction type)
### boundary calls per interaction
| interaction | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| input_masking | 0 | 0 | 0 | 4 | 4 | 4 |
| trusted_layernorm | 0 | 20 | 0 | 0 | 0 | 0 |
| trusted_gelu | 0 | 8 | 0 | 0 | 0 | 0 |
| lm_head_recovery | 0 | 4 | 4 | 4 | 4 | 4 |
| sampling | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_weight_obfuscation | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_affine_folding | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_permutation_absorption | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_norm_core_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_activation_island_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_sandwich_transition | 0 | 0 | 0 | 0 | 8 | 0 |
| security_proxy_requirements | 0 | 0 | 0 | 0 | 0 | 0 |

### transfer bytes per interaction
| interaction | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| input_masking | 0 | 0 | 0 | 176 | 176 | 176 |
| trusted_layernorm | 0 | 1760 | 0 | 0 | 0 | 0 |
| trusted_gelu | 0 | 2816 | 0 | 0 | 0 | 0 |
| lm_head_recovery | 0 | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 |
| sampling | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_weight_obfuscation | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_affine_folding | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_permutation_absorption | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_norm_core_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_activation_island_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_sandwich_transition | 0 | 0 | 0 | 0 | 704 | 0 |
| security_proxy_requirements | 0 | 0 | 0 | 0 | 0 | 0 |

### trusted compute (ops) per interaction
| interaction | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| input_masking | 0 | 0 | 0 | 0 | 0 | 0 |
| trusted_layernorm | 0 | 1760 | 1760 | 0 | 0 | 0 |
| trusted_gelu | 0 | 2816 | 2816 | 0 | 0 | 0 |
| lm_head_recovery | 0 | 1105654 | 1105654 | 1105654 | 1105654 | 1105654 |
| sampling | 0 | 1105654 | 1105654 | 1105654 | 1105654 | 1105654 |
| preprocessing_weight_obfuscation | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_affine_folding | 0 | 0 | 0 | 0 | 0 | 0 |
| preprocessing_permutation_absorption | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_norm_core_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| compatible_activation_island_gpu | 0 | 0 | 0 | 0 | 0 | 0 |
| dense_sandwich_transition | 0 | 0 | 0 | 0 | 176 | 0 |
| security_proxy_requirements | 0 | 0 | 0 | 0 | 0 | 0 |

## Module breakdown (online slice by module category)

### boundary calls per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |
| layernorm | 0 | 20 | 0 | 0 | 0 | 0 |
| attention_qkv | 0 | 0 | 8 | 0 | 8 | 0 |
| attention_score | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_output | 0 | 0 | 8 | 0 | 8 | 0 |
| mlp_fc | 0 | 0 | 8 | 0 | 8 | 0 |
| activation | 0 | 8 | 0 | 0 | 0 | 0 |
| mlp_proj | 0 | 0 | 8 | 0 | 8 | 0 |
| lm_head | 0 | 4 | 4 | 4 | 4 | 4 |
| kv_cache_update | 0 | 0 | 0 | 0 | 0 | 0 |

### trusted compute (ops) per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |
| layernorm | 0 | 1760 | 1760 | 0 | 0 | 0 |
| attention_qkv | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_score | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_output | 0 | 0 | 0 | 0 | 0 | 0 |
| mlp_fc | 0 | 0 | 0 | 0 | 0 | 0 |
| activation | 0 | 2816 | 2816 | 0 | 0 | 0 |
| mlp_proj | 0 | 0 | 0 | 0 | 0 | 0 |
| lm_head | 0 | 1105654 | 1105654 | 1105654 | 1105654 | 1105654 |
| kv_cache_update | 0 | 0 | 0 | 0 | 0 | 0 |

### trusted transfer (bytes) per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 0 | 0 | 0 | 0 | 0 | 0 |
| layernorm | 0 | 1760 | 0 | 0 | 0 | 0 |
| attention_qkv | 0 | 0 | 1408 | 0 | 1408 | 0 |
| attention_score | 0 | 0 | 0 | 0 | 0 | 0 |
| attention_output | 0 | 0 | 704 | 0 | 704 | 0 |
| mlp_fc | 0 | 0 | 1760 | 0 | 1760 | 0 |
| activation | 0 | 2816 | 0 | 0 | 0 | 0 |
| mlp_proj | 0 | 0 | 1760 | 0 | 1760 | 0 |
| lm_head | 0 | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 |
| kv_cache_update | 0 | 0 | 0 | 0 | 0 | 0 |

### gpu ops per module
| module | plain_hf_gpu | tslp_trusted_nonlinear_baseline | ours_current | ours_ideal_gpu_nonlinear | ours_compatible_nonlinear_islands | amulet_style_reference |
|---|---|---|---|---|---|---|
| embedding | 44 | 44 | 44 | 44 | 44 | 44 |
| layernorm | 1760 | 0 | 0 | 1760 | 1760 | 1760 |
| attention_qkv | 1056 | 1056 | 1056 | 1056 | 1056 | 1056 |
| attention_score | 3008 | 3008 | 3008 | 3008 | 3008 | 3008 |
| attention_output | 352 | 352 | 352 | 352 | 352 | 352 |
| mlp_fc | 1408 | 1408 | 1408 | 1408 | 1408 | 1408 |
| activation | 2816 | 0 | 0 | 2816 | 2816 | 2816 |
| mlp_proj | 1408 | 1408 | 1408 | 1408 | 1408 | 1408 |
| lm_head | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 | 4422616 |
| kv_cache_update | 176 | 176 | 176 | 176 | 176 | 176 |

## Compatible Nonlinear Islands Method

ours_compatible_nonlinear_islands is a projected method based on Stage 5.2a correctness probes and Stage 5.2b security proxies. It is not yet integrated into GPT-2 / BERT / T5 wrappers — Stage 5.3 is the integration step.

### Boundary Call Formulas

- `ours_current`: 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head)
- `ours_compatible_nonlinear_islands`: L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model)
- `ours_ideal_gpu_nonlinear`: 1 per forward (single fused GPU pipeline round trip)

### Trusted Compute Reduction

- vs `ours_current`: 0.94%
- vs `tslp_trusted_nonlinear_baseline`: 0.40%
- vs `ours_current` boundary call count: 55.56%

### Preprocessing Cost Increase

- Preprocessing increase vs `ours_current`: 0.05% (affine folding + permutation absorption + compatible mask generation, all amortised over many sessions).

- Preprocessing breakdown (ops):
  - base weight obfuscation: 403784
  - affine folding: 40
  - permutation absorption: 128
  - compatible mask generation: 40

### Online Extra Matmul Count

- `online_extra_matmul_count = 0`. Stage 5.2a verified this across every MLP island cell — operator-compatible mask transitions are folded into adjacent Linear weights offline and add zero online matmuls.

### Security Proxy Caveats

- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.
- This is not a real TEE measurement.
- This is not a real TEE measurement.

### Stage 5.3a Wrapper Integration Status

- `partial_implementation = True` — the GPT-2 single-block wrapper now exposes a `nonlinear_mode="compatible_islands"` feature flag, but the GPT-2 model-level wrapper, BERT, and T5 paths are not yet wired up.
- `gpt2_single_block`: `implemented`
- `gpt2_model_level`: `implemented`
- `bert`: `implemented_probe_level`
- `t5`: `implemented_probe_level`
- Default mode remains `trusted`; compatible_islands must not be enabled by default.
- GPT-2 model-level integration is available.
- BERT/T5 are probe-level integrations, not full wrappers.
- Qwen / TinyLlama / modern decoder-only is a model-level wrapper (Stage 6.4c): multi-block stacking + embedding lookup + final RMSNorm + optionally-masked LM head + KV-cache-aware prefill / decode_step + hand-written greedy generation over RMSNorm + RoPE attention + GQA/MQA + SwiGLU MLP, both mitigation bundles supported. This is not full BERT/T5 wrapper integration.
- modern_decoder_probe: `implemented` (Stage 6.4 tensor-level probes).
- modern_decoder_block_wrapper: `implemented` (Stage 6.4b block-level wrapper).
- modern_decoder_model_wrapper: `implemented` (Stage 6.4c model-level wrapper; synthetic fallback for pytest, real HF load opt-in).

### Stage 5.5 Real-Activation Adaptive Attacker

- `real_activation_attacker_status = "implemented"`.
- `real_activation_attacker_scope = "modern_decoder_block_level"` (Stage 6.4b block-level activations).
- `real_activation_attacker_artifact = "outputs/real_activation_attacks.json"`.
- `security_profile_detail_with_real_activation = "real-activation-adaptive-proxy-evaluated, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- This is NOT a real TEE measurement, NOT formal security, and NOT a black-box query attack. `implemented` / `full_runtime_integrated` / `wall_time_source` are unchanged.

### Stage 5.6 Extension — Inter-Block Masked Boundary + Constant-Time Decode Proxy

- `inter_block_mask_mode_supported = True`, `masked_boundary_experimental_status = "implemented"`, `constant_time_decode_proxy_status = "implemented"`.
- `extended_proxy_status = "implemented"`, `extended_proxy_artifact = "outputs/stronger_attackers.json"`.
- `security_profile_detail_with_extended_proxy = "inter-block-and-constant-time-proxy-evaluated, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- Defaults unchanged: `inter_block_mask_mode ="plain_boundary"`, `constant_time_decode_mode ="off"`. Both experimental modes are opt-in (`--inter-block-mask-mode masked_boundary_experimental --constant-time-decode-mode proxy_equalized`). Constant-time decode is a PROXY (no sleep, no real wall-time change); inter-block masked boundary keeps the residual flow in an orthogonal n_inter mask space across all layers until the LM head absorbs n_inter.

### Stage 5.6 Stronger Attackers

- `stronger_attackers_status = "implemented"`.
- `stronger_attackers_artifact = "outputs/stronger_attackers.json"`.
- `blackbox_proxy_status = "implemented"`, `timing_sidechannel_proxy_status = "implemented"`, `inter_block_masking_gap_status = "identified"`, `inter_block_masking_experimental_status = "implemented_in_stage_5_6_extension"`.
- `security_profile_detail_with_stronger_attackers = "adaptive-blackbox-and-timing-proxy-evaluated, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- Black-box attacker sees only generated tokens + logits summaries; timing proxy is a model-based latency simulator, NOT a real TEE measurement; inter-block residual masking is `identified` but the model-wrapper `masked_boundary_experimental` mode is `not_implemented_in_stage_5_6` (deferred to Stage 5.6 extension or Stage 7.0). `implemented` / `full_runtime_integrated` / `wall_time_source` are unchanged.

### Stage 7.0 — LoRA Private Training Prototype

- `lora_private_training_status = "prototype"`.
- `lora_forward_masking_status = "implemented"`, `lora_training_step_status = "trusted_backward_prototype"`, `lora_security_proxy_status = "implemented"`.
- `lora_training_artifact = "outputs/lora_training_experiments.json"`, `lora_security_artifact = "outputs/lora_security_proxy.json"`.
- `lora_merge_adapter_into_w = False` (constraint 7 — adapter is NEVER merged into the public base weight).
- `security_profile_detail_with_lora = "private-adapter-trusted-backward, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- This is a single-linear, tiny-dimension prototype. It is NOT full Qwen / TinyLlama LoRA fine-tuning, NOT PEFT integration, NOT distributed training, and NOT real TEE training. Backward / optimizer remain trusted in Stage 7.0; a masked-gradient GPU path is deferred to Stage 7.1.

### Stage 7.1 — LoRA Masked Backward / Gradient-Side Obfuscation

- `lora_backward_status = "masked_backward_prototype"`, `lora_loss_status = "trusted_loss"`, `lora_optimizer_status = "trusted_optimizer"`.
- `lora_gradient_security_proxy_status = "implemented"`.
- `lora_backward_artifact = "outputs/lora_backward_experiments.json"`, `lora_gradient_security_artifact = "outputs/lora_gradient_security_proxy.json"`.
- `security_profile_detail_with_lora_backward = "masked-gradient-proxy-evaluated, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- Loss computation and optimizer update remain trusted. GPU only sees masked transcript including `G_tilde / grad_A_tilde / grad_B_tilde`. Rank padding is NOT implemented in Stage 7.1; LoRA rank `r` is still visible from gradient shape (deferred to Stage 7.2).

### Stage 7.2 — LoRA Rank Padding / Hidden-Rank Prototype

- `lora_rank_padding_status = "implemented"`, `lora_hidden_rank_status = "padded-rank-prototype"`.
- `lora_true_rank_hidden_from_shape = True`, `lora_padded_rank_visible = True`.
- `lora_rank_padding_artifact = "outputs/lora_rank_padding_experiments.json"`, `lora_rank_security_artifact = "outputs/lora_rank_security_proxy.json"`.
- `security_profile_detail_with_lora_rank_padding = "rank-padding-proxy-evaluated, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- Stage 7.2 hides true_rank from the GPU-visible shape of `A_pad_tilde / B_pad_tilde / grad_A_pad_tilde / grad_B_pad_tilde`. **padded_rank itself remains visible** to the GPU. dummy_strategy ∈ {"zero_dummy", "paired_cancellation_dummy"}; `zero_dummy` keeps shape-level hiding but the spectral attacker reads `true_rank` back from `SVD(B_pad_tilde)` exactly — the proxy reports `risk_level = high` accordingly. `paired_cancellation_dummy` raises the SVD-cliff from `true_rank` to `true_rank + ⌊(r_pad - r) / 2⌋`, an upper bound only — reported as `needs_more_evaluation`, not `low`.

### Stage 7.5 — Paper Artifact Consolidation + Measured Runtime + Claims Audit

- `paper_artifact_consolidation_status = "implemented"`, `measured_runtime_evaluation_status = "implemented"`, `paper_claims_audit_status = "implemented"`.
- `paper_artifact_consolidation_artifact = "paper_results/summary.md"`, `measured_runtime_artifact = "paper_results/json/measured_runtime.json"`, `paper_claims_audit_artifact = "paper_results/markdown/paper_claims_audit.md"`.
- Stage 7.5 aggregates every existing `outputs/*.json` into paper-ready CSV / Markdown / LaTeX tables (`artifact_inventory`, `correctness_summary`, `security_proxy_summary`, `workload_summary`, `lora_training_summary`, `limitations_summary`), measures local wall-clock latency for plain / masked / rank-padded / multi-layer LoRA primitives (**local emulation, NOT real TEE wall-time; no real sleep**), and classifies every paper claim into `supported / proxy_supported / unsupported`. No new obfuscation primitives, no new attackers.

### Stage 7.4 — Stronger Dummy Distributions / Spectral-Rank Hardening

- `lora_stronger_dummy_status = "implemented"`, `lora_stronger_dummy_security_status = "implemented"`, `lora_spectral_rank_hardening_status = "proxy-evaluated"`.
- `lora_stronger_dummy_artifact = "outputs/lora_stronger_dummy_experiments.json"`, `lora_stronger_dummy_security_artifact = "outputs/lora_stronger_dummy_security_proxy.json"`.
- `security_profile_detail_with_lora_dummy_hardening = "spectral-rank-hardening-proxy-evaluated, not formal"` — additive label only; top-level `security_profile` remains `"proxy-evaluated, not formal"`.
- Stage 7.4 adds five stronger dummy strategies (`gaussian_matched_dummy / spectrum_matched_dummy / noise_injected_cancellation_dummy / orthogonalized_cancellation_dummy / mixed_dummy_ensemble`) on top of Stage 7.2's `zero_dummy / paired_cancellation_dummy`. Every stronger strategy preserves `A_pad B_pad = A_real B_real` either exactly (cancellation strategies) or with a tracked trusted-side correction (`noise_injected_cancellation_dummy`). Stage 7.0 / 7.1 / 7.2 / 7.3 primitives are NOT modified; Stage 7.4 wraps them.
- Security proxy reports ensemble spectral-cliff / 99%-energy / log-elbow detectors over `A_tilde / B_tilde / grad_A_tilde / grad_B_tilde`, plus a dummy-strategy classifier and the Stage 7.3 cross-layer linkage proxy parametrised by dummy strategy. Conservative verdicts per requirement 12 — every paired-cancellation-derived strategy is reported as `needs_more_evaluation` when accuracy is low; `zero_dummy` is `high`. The dummy-strategy classifier itself is reported honestly — Stage 7.4 does NOT claim cryptographic hiding.

### Stage 7.3 — Multi-Layer LoRA Training + Cross-Layer Proxy + Training Timing Proxy

- `lora_multilayer_training_status = "prototype"`, `lora_multilayer_security_proxy_status = "implemented"`, `lora_training_timing_proxy_status = "implemented"`.
- `lora_multilayer_training_artifact = "outputs/multilayer_lora_training_experiments.json"`, `lora_multilayer_security_artifact = "outputs/multilayer_lora_security_proxy.json"`, `lora_training_timing_artifact = "outputs/lora_training_timing_proxy.json"`.
- `security_profile_detail_with_lora_multilayer = "multi-layer-lora-proxy-evaluated, not formal"` — additive label only; top-level `security_profile` remains `"proxy-evaluated, not formal"`.
- Stage 7.3 stacks Stage 7.0 / 7.1 / 7.2 across multiple LoRA-augmented linears (q/k/v/o + SwiGLU MLP) in a tiny synthetic block stack and verifies per-module loss / forward / grad / update allclose plain↔masked. Loss + optimizer remain trusted; adapter is NEVER merged into the public base weight `W`.
- Cross-layer security proxy reports linkage AUC across `fixed_masks_shared_u / independent_u_per_layer / fresh_masks_independent_u / rank_padding_full_bundle`; heterogeneous true_rank with shared padded_rank hides shape-level rank across all modules. Training-timing proxy is a cost-model latency simulator (NOT a real TEE wall-time; NO real sleep). `constant_time_training_mode ∈ {"off", "proxy_equalized"}` with `proxy_equalized` padding every step to the upper-bucket latency.

### Stage 5.5b Real-Token-Prompted Real-Activation Attacker

- `real_token_activation_attacker_status = "implemented"`.
- `real_token_activation_attacker_scope = "modern_decoder_model_level_prefill_decode"` (Stage 6.4c model-level wrapper prefill + decode_step).
- `real_token_activation_attacker_artifact = "outputs/real_token_activation_attacks.json"`.
- `security_profile_detail_with_real_token_activation = "real-token-real-activation-adaptive-proxy-evaluated, not formal"` — additive label only; `security_profile` itself remains `"proxy-evaluated, not formal"`.
- Real tokenizer / real model loading is opt-in (`--attempt-tokenizer-load --attempt-real-model-load`); pytest defaults stay synthetic. This is NOT a real TEE measurement, NOT formal security, and NOT a black-box / side-channel attack. `implemented` / `full_runtime_integrated` / `wall_time_source` are unchanged.

### Stage 5.3e Dense-Sandwich Mitigation Integration

- `mitigation_bundle_selectable = True`.
- `default_mitigation_bundle = "fresh_perm_only"` (preserves backward compatibility for every Stage 5.3a / 5.3b / 5.3c / 6.4 caller that omits the bundle argument).
- `recommended_default_on_bundle = "fresh_perm_plus_sandwich_plus_pad"`.
- `recommended_default_on_status = "acceptable_with_mitigation_under_adaptive_proxy"` (per Stage 5.4 adaptive proxy attackers — NOT a formal security claim).
- `dense_sandwich_supported = True`, `boundary_pad_required = True`, `fresh_permutation_required = True`.
- `compatible_islands` remains feature-flagged behind `nonlinear_mode`; default mode is still `"trusted"`.
- security is `adaptive-proxy-mitigated, not formal` when the full bundle is enabled; this is not a real TEE measurement.
- `measured_integration_scope = "cross_architecture_plus_modern_decoder_model_level"`.
- `full_runtime_integrated = False`.
- `all_architecture_probe_level_implemented = True`.
- `security_profile` remains `proxy-evaluated, not formal`.

## Paper metrics

- `boundary_call_reduction_vs_tslp` = -12.50% (ours_current vs tslp)
- `trusted_transfer_reduction_vs_tslp` = -0.03%
- `online_trusted_compute_reduction_vs_tslp` = -0.55%
- `gpu_offload_ratio` (ours_current) = 79.87%
- `preprocessing_amortized` = `True`
- `boundary_calls_per_forward` =
  - `plain_hf_gpu`: 0
  - `tslp_trusted_nonlinear_baseline`: 8
  - `ours_current`: 9
  - `ours_ideal_gpu_nonlinear`: 1
  - `ours_compatible_nonlinear_islands`: 4
  - `amulet_style_reference`: 1

## Interpretation

- **Main online bottleneck (ours_current):** `lm_head`
- **Next primitive to obfuscate on GPU:** `GELU`

*Note on ours_current vs TSLP boundary calls:* ours_current crosses the boundary once per obfuscated linear (4 per layer) while TSLP crosses once per non-linear (3 per layer plus ln_f). This is an **architectural** difference, not a bookkeeping artefact. Each ours_current crossing moves a smaller activation than a TSLP LayerNorm crossing, and the measured wall time is consistent with that tradeoff.

## Method semantics & citation caveats
### `plain_hf_gpu` — Plain HuggingFace on GPU

Plaintext HF GPT-2 forward / greedy decode. No protection.

- Implemented: **True**
- Implementation note: Hand-written HF greedy loop over plain model().
- Caveat: No security; measured wall time is the GPU-only baseline.

### `tslp_trusted_nonlinear_baseline` — TSLP-style trusted non-linear baseline

Linear / attention on GPU, every LayerNorm and GELU activation makes a TEE round-trip. Modeled after the trusted non-linear split common in shielded-inference literature.

- Implemented: **False**
- Implementation note: No real implementation in this repo. Wall time is projected from op counts using a documented cost model — not a measurement of any specific published system.
- Caveat: TSLP-style is used here as a generic non-linear-in-TEE baseline. It is not a faithful re-implementation of any single published system. Adjust the cost model constants in WorkloadProfileConfig before drawing system-level conclusions.

### `ours_current` — This work — current Stage 4.9 implementation

Right-multiply mask + per-block Conv1D pad compensation, trusted LayerNorm / GELU shortcuts, diagonal vocab output mask on the LM head, internal ObfuscatedGPT2KVCache.

- Implemented: **True**
- Implementation note: Wall time is measured against the real ObfuscatedGPT2ModelWrapper generation path.
- Caveat: Trusted LayerNorm / GELU are engineering shortcuts; their TEE cost is included in proxies but their security model is unprotected non-linearity. See Stage 5.1 / 5.2 roadmap.

### `ours_ideal_gpu_nonlinear` — This work — ideal: LN / GELU on GPU in masked domain

Same wrapper but with LayerNorm and GELU executed inside the obfuscated GPU domain. The trusted side only crosses the boundary to prepare the masked input and to recover the LM head logits. Used as an upper bound, not a measured system.

- Implemented: **False**
- Implementation note: Hypothetical. Op counts come from the same model graph; LN / GELU FLOPs are reattributed from TEE to GPU. Wall time is projected, not measured.
- Caveat: Upper bound. Real obfuscated LN / GELU primitives are deferred to Stage 5.1 / 5.2 and may carry additional overhead this estimate does not capture.

### `ours_compatible_nonlinear_islands` — This work — projected: operator-compatible nonlinear islands

Modeled / projected method. RMSNorm core uses an orthogonal mask, LayerNorm core uses a mean-preserving orthogonal mask, GELU / ReLU / SiLU activations use permutation masks, and SwiGLU uses a paired permutation. Every mask transition is folded into adjacent Linear weights offline, so the masked forward executes with the same number of matmuls as the plaintext forward (Stage 5.2a verified ``online_extra_matmul_count = 0`` for every MLP island cell). Trusted shortcuts for LN and GELU are removed.

- Implemented: **False**
- Implementation note: Projected, not measured. Stage 5.2a verified the correctness probe (28 cells, all_allclose=True, max_online_extra_matmul=0). Stage 5.2b validated the security proxy (fresh permutation + dense sandwich + pad at Linear boundaries are required mitigations). Not yet integrated into the GPT-2 / BERT / T5 wrappers — Stage 5.3 is the integration step.
- Caveat: Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands. The Stage 5.2b security proxy quantified per-strategy linkability and permutation recovery, but the result is a naive-observer upper bound, NOT a formal security proof and NOT a real TEE measurement.

### `amulet_style_reference` — Amulet-style reference (cost model)

Input masking + GPU obfuscated forward + output unmasking, modeled after the high-level pattern in Amulet-style systems. Reference only, not a re-implementation.

- Implemented: **False**
- Implementation note: No implementation in this repo. Cost-model reference under the assumption that the entire obfuscated forward runs as a single GPU pipeline between trusted input masking and trusted output unmasking.
- Caveat: Amulet-style here means the abstract pattern of input mask + GPU forward + output recovery. Real Amulet systems may include primitives and overheads not captured by this proxy. Use with explicit attribution to assumptions.

## Limitations
- Simulated TEE cost model — not real SGX wall-clock.
- Wall time for tslp_baseline, ours_ideal, and amulet_style_reference is projected, not measured.
- tiny-gpt2 (n_layer=2, n_embd=2, n_head=2) is far smaller than production GPT-2.
- FLOP / byte proxies use coarse constants; absolute numbers are illustrative.
- Qwen / Llama not yet covered — see Stage 5.4 roadmap.
- amulet_style_reference is a reference cost model, not a re-implementation of any published system.

## Reproducibility

```bash
python scripts/run_workload_profile.py --batch-size 2 --prompt-len 8 --max-new-tokens 4 --warmup 2 --repeat 5 --use-pad True
```
