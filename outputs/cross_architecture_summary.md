# Privacy LLM Obfuscation — Cross-Architecture Summary (Stage 6.3)

## Experiment scope

Cross-architecture summary aggregates Stage 5.0 (decoder-only), Stage 6.1 (encoder-only) and Stage 6.2 (encoder-decoder cross-attention) probe outputs plus the Stage 5.0.1 / 5.2c workload profile. It does not re-execute any probe.

## Cross-architecture coverage table

| architecture | status | model_id | model_class | attention_kind | cache_type | cells | rows |
|---|---|---|---|---|---|---|---|
| decoder_only | aggregated | sshleifer/tiny-gpt2 | GPT2LMHeadModel | causal_self_attention | autoregressive_kv_cache | 36 | 36 |
| encoder_only | aggregated | hf-internal-testing/tiny-bert | BertForMaskedLM | bidirectional_self_attention | none | 12 | 24 |
| encoder_decoder | aggregated | hf-internal-testing/tiny-random-t5 | T5ForConditionalGeneration | cross_attention | encoder_memory_cache | 24 | 48 |

## Attention invariant summary

| architecture | all allclose | max output err | max score err | max prob err | max cache err |
|---|---|---|---|---|---|
| decoder_only | true | 9.219e-09 | 2.765e-10 | 0 | 4.098e-08 |
| encoder_only | true | 6.139e-06 | 3.338e-06 | 7.749e-07 | — |
| encoder_decoder | true | 8.643e-07 | 9.835e-07 | 2.086e-07 | 1.192e-06 |

## Cache support summary

| architecture | cache_type | max cache err |
|---|---|---|
| decoder_only | autoregressive_kv_cache | 4.098e-08 |
| encoder_only | none | — |
| encoder_decoder | encoder_memory_cache | 1.192e-06 |

## Pad support summary

| architecture | use_pad values seen | padding mask supported | bias (q/k/v/o) | relative position bias |
|---|---|---|---|---|
| decoder_only | False/True | false | True/True/True/True | false |
| encoder_only | False/True | true | — | false |
| encoder_decoder | False/True | true | False/False/False/False | false |

## Workload summary (from Stage 5.0.1 profiler)

| method | implemented | boundary calls | boundary calls formula | trusted compute ops | gpu ops | measured wall-time (ms) | source |
|---|---|---|---|---|---|---|---|
| plain_hf_gpu | true | 0 | 0 (no boundary) | 0 | 4434424 | 2.109e+00 | measured |
| tslp_trusted_nonlinear_baseline | false | 32 | 3L + 2 = 8 per forward (LN_1 + LN_2 + GELU per layer + ln_f + LM head) | 1110230 | 4429848 | — | projected_from_op_counts |
| ours_current | true | 36 | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | 1116310 | 4429848 | 7.428e+00 | measured |
| ours_ideal_gpu_nonlinear | false | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4434424 | — | projected_from_op_counts |
| ours_compatible_nonlinear_islands | false | 16 | L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model) | 1105830 | 4434424 | — | projected_from_op_counts |
| amulet_style_reference | false | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4434424 | — | projected_from_op_counts |

## Compatible Nonlinear Island Workload Projection

ours_compatible_nonlinear_islands is a projected method based on Stage 5.2a correctness probes (28 cells, all_allclose=True, `online_extra_matmul_count = 0`) and Stage 5.2b security proxies. It is not yet integrated into GPT-2 / BERT / T5 wrappers — Stage 5.3 is the integration step. Per-architecture status is `projected_from_probe`.

- Boundary formula: `L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model)`
- `online_extra_matmul_count` = 0
- `security_profile` = `proxy-evaluated, not formal`

| architecture | model_id | attention_kind | current method | current formula | compatible formula | boundary reduction | trusted compute reduction | online extra matmul | status | security_proxy_status |
|---|---|---|---|---|---|---|---|---|---|---|
| decoder_only | sshleifer/tiny-gpt2 | causal_self_attention | ours_current | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model) | 55.56% | 0.94% | 0 | projected_from_probe | proxy-evaluated, not formal |
| encoder_only | hf-internal-testing/tiny-bert | bidirectional_self_attention | stage6_probe_plus_trusted_shortcuts | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model) | 55.56% | 0.94% | 0 | projected_from_probe | proxy-evaluated, not formal |
| encoder_decoder | hf-internal-testing/tiny-random-t5 | cross_attention | stage6_probe_plus_trusted_shortcuts | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | L + 2 = 4 per forward (1 input mask + L per-layer dense-mask transition between islands + 1 LM head; projected, conservative model) | 55.56% | 0.94% | 0 | projected_from_probe | proxy-evaluated, not formal |

Security proxy caveats (from Stage 5.2b, applied to every architecture row above):
- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.
- Not yet integrated into the GPT-2 / BERT / T5 wrappers (`projected_from_probe`, not measured). No real TEE isolation.

## Compatible Island Integration Status

Stage 5.3c — per-architecture status of the operator-compatible nonlinear-island integration. Default mode remains `trusted` for every wrapper; `compatible_islands` is gated behind a `nonlinear_mode` feature flag.

| architecture_type | model_id | integration_level | nonlinear_mode_available | use_pad_supported | online_extra_matmul_count | security_proxy_status |
|---|---|---|---|---|---|---|
| decoder_only | sshleifer/tiny-gpt2 | model_level | trusted/compatible_islands | True | 0 | proxy-evaluated, not formal |
| encoder_only | hf-internal-testing/tiny-bert | probe_level | trusted/compatible_islands | True | 0 | proxy-evaluated, not formal |
| encoder_decoder | hf-internal-testing/tiny-random-t5 | probe_level | trusted/compatible_islands | True | 0 | proxy-evaluated, not formal |
| modern_decoder_only | qwen_like / llama_like / synthetic_modern_decoder | model_level | trusted/compatible_islands | True | 0 | adaptive-proxy-evaluated, not formal |

- `measured_integration_scope = "cross_architecture_plus_modern_decoder_model_level"`.
- `full_runtime_integrated = False`.
- `all_architecture_probe_level_implemented = True`.
- GPT-2 model-level integration is available.
- BERT/T5 are probe-level integrations, not full wrappers.
- default mode remains `trusted`.
- LayerNorm remains trusted unless explicitly stated otherwise.
- no generation changes for BERT/T5.
- security follows Stage 5.2b caveats (fresh permutation per session, dense sandwich at Linear boundaries, pad at Linear boundaries only).
- `security_profile` remains `proxy-evaluated, not formal`.
- not a real TEE measurement.
- not full BERT/T5 wrapper integration.

### Per-architecture limitations

- **decoder_only**:
  - GPT-2 model-level integration is measured smoke, not a real TEE measurement.
  - LayerNorm remains trusted.
  - Default mode remains trusted; compatible_islands is gated behind a feature flag.
- **encoder_only**:
  - BERT is probe-level integration, not a full BERT wrapper.
  - MLM head, pooler, and classifier are not modified.
  - LayerNorm remains trusted.
  - Default mode remains trusted; compatible_islands is gated behind a feature flag.
- **encoder_decoder**:
  - T5 / BART is probe-level integration, not a full wrapper.
  - LM head and encoder-decoder generation are not modified.
  - Cross-attention probe invariants (Stage 6.2) are not modified.
  - Gated-GELU is not yet supported (Stage 5.2a only covers SiLU gated MLP island).
  - Default mode remains trusted; compatible_islands is gated behind a feature flag.
- **modern_decoder_only**:
  - Model-level wrapper smoke is allclose vs plain reference; real TEE wall-time is not measured.
  - Greedy generation only; beam / top-k / top-p not implemented.
  - Real Qwen / TinyLlama loading is opt-in; pytest stays synthetic.
  - Inherits Stage 5.4 mitigation requirements.
  - Default mode remains trusted; compatible_islands is gated behind a feature flag.

## Mitigation Bundle Support

Stage 5.3e — per-architecture support for the two mitigation bundles. `default_mitigation_bundle = 'fresh_perm_only'` (preserves backward compatibility); `recommended_default_on_bundle = 'fresh_perm_plus_sandwich_plus_pad'` → `acceptable_with_mitigation_under_adaptive_proxy` per Stage 5.4 adaptive proxy attackers. `compatible_islands` remains feature-flagged behind `nonlinear_mode`; default mode stays `"trusted"`.

| architecture | integration_level | fresh_perm_only | fresh_perm_plus_sandwich_plus_pad | use_pad_supported | dense_sandwich_enabled | online_extra_matmul_count | default_on_candidate | security_profile |
|---|---|---|---|---|---|---|---|---|
| decoder_only | model_level | supported | supported | True | True | 0 | fresh_perm_plus_sandwich_plus_pad | proxy-evaluated, not formal |
| encoder_only | probe_level | supported | supported | True | True | 0 | fresh_perm_plus_sandwich_plus_pad | proxy-evaluated, not formal |
| encoder_decoder | probe_level | supported | supported | True | True | 0 | fresh_perm_plus_sandwich_plus_pad | proxy-evaluated, not formal |
| modern_decoder_only | model_level | supported | supported | True | True | 0 | fresh_perm_plus_sandwich_plus_pad | adaptive-proxy-evaluated, not formal |

- Bundle support is probe-level / wrapper-level: enabling the full bundle does NOT change the wrapper's default `nonlinear_mode` and does NOT promote `ours_compatible_nonlinear_islands.implemented` to `True`.
- security is `adaptive-proxy-mitigated, not formal` when the full bundle is enabled; this is not a real TEE measurement.

## Modern Decoder Model-Level Integration (Stage 6.4c)

Stage 6.4c stacks the Stage 6.4b block wrapper into a multi-layer model-level obfuscated decoder with embedding lookup, final RMSNorm, an optionally-masked LM head, KV-cache-aware prefill / decode_step, and a hand-written greedy generation loop. Real Qwen / TinyLlama loading is opt-in; pytest stays synthetic.

| field | value |
|---|---|
| integration_level | model_level |
| modern_decoder_block_wrapper_status | implemented |
| norm_type / activation_type / position_encoding | rmsnorm / swiglu / rotary |
| attention_variant | mha/gqa/mqa |
| online_extra_matmul_count | 0 |
| security_proxy_status | adaptive-proxy-evaluated, not formal |
| block_level_correctness_artifact | `outputs/modern_decoder_block_wrapper_smoke.json` |
| modern_decoder_model_wrapper_status | implemented |
| modern_decoder_generation_status | greedy_generation_implemented |
| modern_decoder_kv_cache_status | implemented |
| model_level_correctness_artifact | `outputs/modern_decoder_model_wrapper_smoke.json` |
| real_activation_attacker_status | implemented |
| real_activation_attacker_scope | modern_decoder_block_level |
| real_activation_attacker_artifact | `outputs/real_activation_attacks.json` |
| real_token_activation_attacker_status | implemented |
| real_token_activation_attacker_scope | modern_decoder_model_level_prefill_decode |
| real_token_activation_attacker_artifact | `outputs/real_token_activation_attacks.json` |
| security_profile_detail_with_real_token_activation | real-token-real-activation-adaptive-proxy-evaluated, not formal |
| extended_proxy_status | implemented |
| extended_proxy_artifact | `outputs/stronger_attackers.json` |
| inter_block_mask_mode_supported | True |
| masked_boundary_experimental_status | implemented |
| constant_time_decode_proxy_status | implemented |
| security_profile_detail_with_extended_proxy | inter-block-and-constant-time-proxy-evaluated, not formal |
| stronger_attackers_status | implemented |
| stronger_attackers_artifact | `outputs/stronger_attackers.json` |
| blackbox_proxy_status | implemented |
| timing_sidechannel_proxy_status | implemented |
| inter_block_masking_gap_status | identified |
| inter_block_masking_experimental_status | implemented_in_stage_5_6_extension |
| security_profile_detail_with_stronger_attackers | adaptive-blackbox-and-timing-proxy-evaluated, not formal |

### Stage 5.5b Real-Token-Prompted Real-Activation Attacker

Stage 5.5b drives the Stage 6.4c model-level wrapper (embedding + prefill + decode_step + greedy generation) with real (or deterministic synthetic) input_ids and replays the Stage 5.5 adaptive attacker family (linear / MLP / Sinkhorn permutation / linkability) against the resulting (plain, visible) trace pairs across PREFILL and DECODE_STEP. Real tokenizer / real model loading is opt-in; pytest stays synthetic. The masked-tensor risk classification stays `low`; the inter-block hidden states (`boundary_input` / `final`) are plain at the model-wrapper boundary by construction — this is a structural model-wrapper limitation, not a Stage 5.5b attacker finding. Not formal security; not a real TEE measurement.

### Stage 5.6 Extension — Inter-Block Masked Boundary + Constant-Time Decode Proxy

Stage 5.6 extension wires `masked_boundary_experimental` through ObfuscatedModernDecoderModelWrapper so the inter-block residual stays in a fresh orthogonal `n_inter` mask space across all layers; the LM head absorbs `n_inter` and `boundary_input` / `final` join the masked tensor set. A `constant_time_decode_mode = "proxy_equalized"` lever in the timing proxy equalises per-step simulated latency to a per-method upper bound (PROXY only — no sleep, no real wall-time change). Defaults stay `plain_boundary` and `off`. Both opt-in via CLI flags. Promotion eligibility: `security_profile_detail_with_extended_proxy = "inter-block-and-constant-time-proxy-evaluated, not formal"` when both modes are on and the envelope stays low-risk. Not formal security; not a real TEE measurement.

| lora_private_training_status | prototype |
| lora_forward_masking_status | implemented |
| lora_training_step_status | trusted_backward_prototype |
| lora_security_proxy_status | implemented |
| lora_training_artifact | `outputs/lora_training_experiments.json` |
| lora_security_artifact | `outputs/lora_security_proxy.json` |
| lora_merge_adapter_into_w | False |
| security_profile_detail_with_lora | private-adapter-trusted-backward, not formal |
| lora_backward_status | masked_backward_prototype |
| lora_loss_status | trusted_loss |
| lora_optimizer_status | trusted_optimizer |
| lora_gradient_security_proxy_status | implemented |
| lora_backward_artifact | `outputs/lora_backward_experiments.json` |
| lora_gradient_security_artifact | `outputs/lora_gradient_security_proxy.json` |
| security_profile_detail_with_lora_backward | masked-gradient-proxy-evaluated, not formal |
| lora_rank_padding_status | implemented |
| lora_hidden_rank_status | padded-rank-prototype |
| lora_true_rank_hidden_from_shape | True |
| lora_padded_rank_visible | True |
| lora_rank_padding_artifact | `outputs/lora_rank_padding_experiments.json` |
| lora_rank_security_artifact | `outputs/lora_rank_security_proxy.json` |
| security_profile_detail_with_lora_rank_padding | rank-padding-proxy-evaluated, not formal |
### Stage 5.6 Stronger Attackers (Black-box + Timing + Inter-block Gap)

Stage 5.6 ships three proxy attackers that do NOT require paired plaintext/visible internal supervision. (1) Black-box query attacker uses only generated tokens + per-step logits summaries; mode / bundle / use_pad distinguishability sits at random chance under Stage 6.4c's exact-token-match guarantee. (2) Timing side-channel proxy uses the Stage 5.2c op-count cost model + Gaussian noise; decode_step and prompt-length latency leakage is `high` (structural — any latency observer can count decode steps), mitigation-bundle distinguishability is `low`. (3) Inter-block residual masking gap analysis confirms the Stage 5.5b finding that `boundary_input` / `final` are plain at the model-wrapper boundary; a single-transition math probe verifies the orthogonal-mask fix is numerically correct, but the full `masked_boundary_experimental` mode is `not_implemented_in_stage_5_6` (deferred to Stage 5.6 extension / Stage 7.0). Envelope-integrity risk: `low`. Structural-leakage risk: `high`. Not formal security; not a real TEE measurement.

### Stage 7.2 — LoRA Rank Padding / Hidden-Rank Prototype

Stage 7.2 stacks rank padding on top of the Stage 7.0 forward + Stage 7.1 backward path. The trusted side constructs `A_pad ∈ R^{d_in × r_pad}`, `B_pad ∈ R^{r_pad × d_out}` with `A_pad B_pad = A_real B_real` exactly, so the function value (and the LoRA scaling `α / r`) are unchanged. The GPU only ever sees `A_pad_tilde / B_pad_tilde / grad_A_pad_tilde / grad_B_pad_tilde` whose rank dimension is `r_pad`; **true rank `r` is hidden from tensor shape**. After masked backward recovery, the trusted side slices `grad_A_pad[:, :true_rank]` / `grad_B_pad[:true_rank, :]` and feeds those into the SGD / AdamW step. The optimizer state is sized to `true_rank`, never `padded_rank`; the dummy slice is re-sampled fresh and never persists into the optimizer. Two dummy strategies are supported: `zero_dummy` (baseline; spectral attacker reads `true_rank` back from `SVD(B_pad_tilde)` exactly — proxy `risk_level = high`) and `paired_cancellation_dummy` (pair dummies as `[R, R], [S, -S]` so the spectral cliff sits at `true_rank + ⌊(r_pad - r) / 2⌋`, an upper bound only — proxy `risk_level = needs_more_evaluation`). `padded_rank` itself remains visible to the GPU (Stage 7.2 does not hide `r_pad`). `security_profile_detail_with_lora_rank_padding = "rank-padding-proxy-evaluated, not formal"`. `security_profile` itself stays `"proxy-evaluated, not formal"`. NOT full Qwen / TinyLlama LoRA fine-tuning, NOT PEFT integration, NOT distributed training, NOT real TEE training.

### Stage 7.1 — LoRA Masked Backward / Gradient-Side Obfuscation

Stage 7.1 extends Stage 7.0 by sending masked gradient tensors through the GPU boundary as well. The trusted side still computes G = dL/dY (loss stays trusted) and applies the optimizer update (SGD / AdamW state stays trusted); the GPU runs the gradient matmuls on masked tensors (G_tilde, X_tilde, A_tilde, B_tilde, grad_A_tilde, grad_B_tilde, optional grad_X_tilde) and returns masked gradients. Trusted side recovers via grad_A = N_in^{-T} grad_A_tilde U^T (+ pad compensation when use_pad=True) and grad_B = U^{-T} grad_B_tilde N_out^T (+ pad compensation). Per-step grad_A / grad_B match plain autograd to float64 precision across SGD / AdamW × use_pad ∈ {True, False} × fresh_u_per_step ∈ {True, False}. Gradient leakage proxy ranks five strategies; fresh masks drive gradient-side membership-style AUC from ≈ 1.0 (fixed) to ≈ 0.5 (random) — though LoRA rank `r` is still visible from the shape of grad_A_tilde / grad_B_tilde (rank padding is Stage 7.2). `security_profile_detail_with_lora_backward = "masked-gradient-proxy-evaluated, not formal"`. `security_profile` itself stays `"proxy-evaluated, not formal"`. This is NOT full Qwen / TinyLlama LoRA fine-tuning, NOT PEFT integration, NOT distributed training, NOT real TEE training.

### Stage 7.0 — LoRA Private Training Prototype

Stage 7.0 lands a LoRA primitive (`src/pllo/ops/lora.py`) + training-step correctness probe + leakage proxy: GPU sees only masked transcript `(X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde)`; backward / optimizer remain trusted; the adapter is NEVER merged into the public base weight. `lora_private_training_status = "prototype"`. Forward correctness allclose under both `use_pad=True` / `use_pad=False`; per-step loss / gradient / update-error match the plain reference to float64 precision. Leakage proxy ranks five strategies (unmasked / fixed / fresh-U / fresh-masks / fresh+pad) under three sub-attacks (adapter extraction, gradient visibility accounting, membership-style linkability). Rank `r` is visible from the shape of A_tilde / B_tilde — rank padding is NOT implemented in Stage 7.0. `security_profile_detail_with_lora = "private-adapter-trusted-backward, not formal"`. `security_profile` itself is unchanged (`"proxy-evaluated, not formal"`). This is NOT full Qwen / TinyLlama LoRA fine-tuning, NOT PEFT integration, NOT distributed training, NOT real TEE training.

- Default mode for the wider system remains `"trusted"`; default mitigation bundle remains `"fresh_perm_only"`.
- This is block-level integration, not a full model-level wrapper; `full_runtime_integrated` stays False.
- No generation / decode_step / KV cache runtime is implemented at the wrapper level.
- Not a real TEE measurement; not formal security.

## Trusted shortcuts still in place per architecture

- **decoder_only**:
  - `trusted_layernorm`
  - `trusted_gelu`
  - `lm_head_vocab_diag_mask_only`
- **encoder_only**:
  - `trusted_layernorm`
  - `trusted_gelu`
  - `no_mlm_head_obfuscation`
- **encoder_decoder**:
  - `trusted_layernorm`
  - `trusted_ffn_activation`
  - `no_decoder_self_attention_cache`
  - `no_relative_position_bias_obfuscation`

## Limitations

- **decoder_only**:
  - LayerNorm runs inside SimulatedTEE (trusted shortcut).
  - GELU runs inside SimulatedTEE (trusted shortcut).
  - LM head uses a diagonal vocab output mask only, no full pad.
  - Real TEE isolation is not implemented.
- **encoder_only**:
  - BERT obfuscated forward (LayerNorm / GELU / FFN / MLM head) is not implemented.
  - Only first-layer self-attention Q / K / V / O is validated.
  - Real TEE isolation is not implemented.
- **encoder_decoder**:
  - T5/BART obfuscated forward (LayerNorm / FFN / activation / LM head) is not implemented.
  - Decoder self-attention KV cache is not implemented.
  - Encoder-decoder generation is not implemented.
  - Relative position bias is not obfuscated.
  - Real TEE isolation is not implemented.
- This summary aggregates existing JSON; it does not re-run probes.
- It does not claim real TEE security; security claims are deferred to the security proxy report.

## Next stage plan

- **Stage 5.1** — GPU-side LayerNorm primitive (replaces the trusted LayerNorm shortcut shared by all three architectures).
- **Stage 5.2** — GELU / activation primitive feasibility (replaces the trusted activation shortcut).
- **Stage 6.4** — Qwen / ModelScope migration on top of Stage 6.0+'s architecture scaffold once a non-trusted nonlinear primitive is ready.
