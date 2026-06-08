# Paper-Ready Experiment Suite

_Stage 7.7: aggregate CPU local-emulation experiments into a paper-ready report. NO real GPU / TEE / formal cryptographic security._

## Environment

| Field | Value |
|---|---|
| device | cpu |
| dtype | float64 |
| real_gpu | False |
| real_tee | False |
| network_required | False |

## Executive Summary

We prepare a CPU local-emulation experiment suite that validates padded masked generation correctness, low-interaction operator-compatible execution, RoPE-safe pre-masking, norm-mask granularity, attention privacy modes, scalable LM-head alternatives, LoRA integration, paged KV abstraction, multi-session isolation, and probabilistic integrity spot-checking. These experiments establish algebraic correctness and leakage/cost accounting, but do not constitute real GPU/TEE deployment or formal cryptographic security.

## Experiment Matrix

| Stage | Experiment | Kind | Loaded |
|---|---|---|---|
| 7.6e | `modern_decoder_generation_correctness` | correctness | True |
| 7.6f | `modern_decoder_low_interaction_correctness` | correctness | True |
| 7.6g | `modern_decoder_rope_safe_low_interaction` | privacy_leakage | True |
| 7.6h | `norm_granularity_low_interaction` | privacy_leakage | True |
| 7.6i | `attention_privacy_modes` | privacy_leakage | True |
| 7.7a | `lm_head_scalability` | scalability | True |
| 7.7b | `lora_protocol_integration` | integration | True |
| 7.7c | `paged_kv_abstraction` | infrastructure | True |
| 7.7d | `multi_session_batching` | isolation | True |
| 7.7e | `integrity_spotcheck` | active_adversary_proxy | True |
| 7.7f | `paper_cost_model` | cost_model | True |
| 7.7g | `paper_claims_audit_v2` | audit | True |

## Stage Status

| Stage Key | Status | Stage Id | Paper-Safe Wording |
|---|---|---|---|
| `7.6e_modern_decoder_generation_correctness` | ok | n/a |  |
| `7.6f_modern_decoder_low_interaction_correctness` | ok | n/a |  |
| `7.6g_modern_decoder_rope_safe_low_interaction` | ok | n/a |  |
| `7.6h_norm_granularity_low_interaction` | ok | 7.6h | We add a granularity knob to the orthogonal RMSNorm-compatible mask. Token-wise and chunk-wise modes preserve per-row... |
| `7.6i_attention_privacy_modes` | ok | 7.6i | Exact low-interaction attention with ordinary accelerator-side softmax exposes the attention map because the QK invar... |
| `7.7a_lm_head_scalability` | ok | 7.7a | Dense orthogonal N_vocab is not scalable to real LLM vocab sizes. Permutation and block-diagonal masks scale but disc... |
| `7.7b_lora_integration` | ok | 7.7b | LoRA adapters integrate with the low-interaction main protocol via the same padded-boundary algebra used for the base... |
| `7.7c_paged_kv_abstraction` | ok | 7.7c | The masked KV invariant ``K_tilde = K @ N_K`` and ``V_tilde = V @ N_V`` is preserved under a CPU synthetic paged cach... |
| `7.7d_multi_session_batching` | ok | 7.7d | Per-session orthogonal masks are sampled independently and produce per-session boundary fingerprints; the same prompt... |
| `7.7e_integrity_spotcheck` | ok | 7.7e | We prototype a probabilistic spot-check defence against an active adversary that returns corrupted masked tensors. Th... |
| `7.7f_complexity_model` | ok | 7.7f | We provide symbolic and tiny / real-config FLOP and storage estimates for every protocol mode. These are complexity-m... |
| `7.7g_paper_claims_audit_v2` | ok | 7.7g | All paper claims are classified into supported, proxy_supported, cost_model_only, or unsupported, with the correspond... |

## Paper Claims Summary

| Status | Count |
|---|---|
| supported | 10 |
| proxy_supported | 1 |
| cost_model_only | 0 |
| unsupported | 4 |

## Supported Claims

- `padded_full_generation_correctness` — evidence: `outputs/modern_decoder_generation_correctness.json`
- `one_round_low_interaction_exact_mode` — evidence: `outputs/modern_decoder_low_interaction_correctness.json`
- `rope_transient_plain_qk_eliminated` — evidence: `outputs/modern_decoder_rope_safe_low_interaction.json`
- `norm_full_gram_reduced_by_token_chunk_masks` — evidence: `outputs/norm_granularity_low_interaction.json`
- `attention_maps_hidden_only_in_trusted_softmax_mode` — evidence: `outputs/attention_privacy_modes.json`
- `attention_maps_visible_in_exact_low_interaction_mode` — evidence: `outputs/attention_privacy_modes.json`
- `scalable_lm_head_dense_mask_not_feasible` — evidence: `outputs/lm_head_scalability.json`
- `lora_integration_supported_for_specified_sites` — evidence: `outputs/lora_protocol_integration.json`
- `paged_kv_invariant_supported_in_synthetic_abstraction` — evidence: `outputs/paged_kv_abstraction.json`
- `multi_session_mask_isolation_supported_in_cpu_simulation` — evidence: `outputs/multi_session_batching.json`

## Proxy-Supported Claims

- `integrity_only_probabilistic_spot_check` — evidence: `outputs/integrity_spotcheck.json`

## Unsupported Claims (NEVER write as supported)

- `no_real_gpu_or_tee_wall_clock` — blocker: Actual hardware access (H100 CC / SGX).
- `no_formal_cryptographic_security` — blocker: Cryptographic protocol design or formal proof out of scope for this project.
- `no_full_qwen_or_llama_deployment_unless_real_wrapper` — blocker: Real model loader + real GPU.
- `no_hardware_side_channel_evaluation` — blocker: Real hardware + side-channel platform.

## Mode Comparison (from 7.6i and 7.7f)

| Mode | exact | one_round_trip | attention_hidden | intermediate_tee_reentry |
|---|---|---|---|---|
| exact_visible_attention | True | True | False | False |
| trusted_softmax_attention | True | False | True | True |
| score_blinding_experimental | True | True | False | False |

## Cost / Complexity (from 7.7f)

See [outputs/paper_cost_model.md](paper_cost_model.md) for the symbolic formulas, tiny-config counts, and LLaMA-7B-ish real-config estimates per protocol mode.

## Scalability Warnings

- Dense V x V LM-head mask is NOT feasible for real LLM vocab; 
  use permutation or block-diagonal (see 7.7a).
- Token-wise norm mask is O(s d^3) per call; sequence mode is 
  cheaper but exposes full Gram (see 7.6h / 7.7f).
- Trusted-softmax mode breaks one-round-trip property; adds 
  L extra TEE round trips per decode step (see 7.6i / 7.7f).

## Remaining Blockers Before Real GPU / TEE

- Real H100 CC / SGX / Gramine / Occlum / TEE platform.
- Real CUDA / FlashAttention / vLLM backend with fused confidential kernels.
- Real GPU paged-attention kernel + serving scheduler.
- Cryptographic verifiable-computation primitive.
- Hardware side-channel evaluation platform.
- LoRA training (backward) integration.

## Limitations

- CPU local emulation only.
- No real TEE / GPU deployment.
- No hardware side-channel evaluation.
- No formal cryptographic / semantic / differential-privacy security.
- Validates algebraic correctness, leakage accounting, and cost-model evidence only.

## Recommended Paper Wording

> We prepare a CPU local-emulation experiment suite that validates padded masked generation correctness, low-interaction operator-compatible execution, RoPE-safe pre-masking, norm-mask granularity, attention privacy modes, scalable LM-head alternatives, LoRA integration, paged KV abstraction, multi-session isolation, and probabilistic integrity spot-checking. These experiments establish algebraic correctness and leakage/cost accounting, but do not constitute real GPU/TEE deployment or formal cryptographic security.

## Unsafe Wording to Avoid

- real TEE/GPU performance
- formal cryptographic security
- semantic security
- full Qwen/LLaMA deployment
- attention maps hidden in exact low-interaction mode
- dense vocab mask is scalable
- active malicious accelerator fully handled
- hardware side channels evaluated

