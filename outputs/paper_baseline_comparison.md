# Paper Baseline Comparison (CPU only)

_Risk levels are proxy-derived from the existing Stage 5-7 proxy summary, not formal security guarantees. Local CPU runtime is local-emulation only, NOT real TEE wall-time and NOT GPU throughput._

| variant | kind | correctness_error | token_match_rate | loss_diff | boundary_calls | online_extra_matmul_count | local_runtime_ms | proxy_risk_level | supported_claim_type | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| plain_cpu | inference | 0.0 | 1.0 | 0.0 | 0 | 0 | 0.008558599984098691 | high | baseline | Plain reference; GPU sees plaintext X, W; included only as the no-defense baseline. |
| trusted_nonlinear_partition | inference | 5.620504062164855e-16 | 1.0 | 6.938893903907228e-18 | 32 | 0 | 0.14833339996584982 | needs_more_evaluation | tee_partition_baseline | Linear masked; nonlinear computed trusted-side; coarse TEE-partition baseline. |
| fixed_permutation_only | inference | 2.942091015256665e-15 | 1.0 | 4.163336342344337e-17 | 4 | 0 | 0.16578320000917302 | high | risk_baseline | Activation island uses a fixed P across calls -- high linkability baseline. |
| fresh_perm_only | inference | 3.400058012914542e-15 | 1.0 | 1.3877787807814457e-17 | 4 | 0 | 0.16774159998931282 | medium | partial_mitigation | Fresh permutation per call; no dense sandwich, no boundary pad. |
| full_mitigation_bundle | inference | 3.5041414214731503e-15 | 1.0 | 0.0 | 16 | 0 | 0.15381700004581944 | needs_more_evaluation | proxy_supported_main | fresh_perm_plus_sandwich_plus_pad; matches Stage 7.5 'ours_compatible_nonlinear_islands' row. |
| full_bundle_masked_boundary | inference | 2.8449465006019636e-15 | 1.0 | 0.0 | 4 | 0 | 0.14143339999463933 | needs_more_evaluation | proxy_supported_ablation | Full bundle + inter_block_mask_mode=masked_boundary_experimental (opt-in ablation). |
| full_bundle_masked_boundary_constant_time_proxy | inference | 2.858824288409778e-15 | 1.0 | 6.938893903907228e-18 | 4 | 0 | 0.1406919999681122 | low | proxy_supported_timing | Full bundle + masked boundary + constant_time_decode_proxy=proxy_equalized; cost-model timing proxy. |
| lora_plain | lora | 0.0 | 1.0 | 0.0 | 0 | 0 | 0.008316399998875568 | high | lora_baseline | Plain LoRA training; GPU sees plaintext A, B and per-step gradients. |
| lora_masked_forward_backward | lora | 2.6367796834847468e-15 | 1.0 | 0.0 | 2 | 0 | 0.14179160002640856 | needs_more_evaluation | proxy_supported_lora | Stage 7.0 / 7.1 masked LoRA forward + backward; loss and optimizer remain trusted-side. |
| lora_rank_padded | lora | 1.6833756610878936e-14 | 1.0 | 4.85722573273506e-17 | 2 | 0 | 0.191816799906519 | needs_more_evaluation | proxy_supported_rank | Stage 7.2 rank padding with paired_cancellation_dummy; padded_rank still visible. |

## Limitations

- All risk levels are proxy-derived from the mitigation configuration and the existing Stage 5-7 proxy summaries; this module does NOT run new attackers.
- Boundary call counts and online extra matmul counts are derived structurally from the mitigation bundle; they are not measured kernel launches.
- Local CPU runtime only; not real TEE wall-time and not GPU throughput.
- No formal / cryptographic / semantic security is claimed.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).
- Reports publish summary metrics only; raw tensors / masks / adapters / gradients are never emitted.
