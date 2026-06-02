# Measured Runtime (Local Emulation, NOT real TEE)

_This is local runtime emulation, not real TEE wall-time. No real sleep, no real runtime gating._

| component | variant | num_warmup | num_repeats | mean_ms | median_ms | std_ms | min_ms | max_ms | device | dtype | wall_time_source | skipped_with_reason | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_synthetic_linear | X W | 2 | 5 | 0.0017586004105396569 | 0.0017089987522922456 | 0.00023257677432648142 | 0.0015420009731315076 | 0.0020420047803781927 | cpu | float64 | measured_local_emulation | None | Synthetic baseline; no obfuscation. |
| plain_lora_forward | plain_rank_r | 2 | 5 | 0.008008399163372815 | 0.008208997314795852 | 0.00115310244100327 | 0.00654099858365953 | 0.00945899955695495 | cpu | float64 | measured_local_emulation | None | Plain rank-r LoRA forward; no masking. |
| masked_lora_forward | fresh_masks_fresh_u_with_pad | 2 | 5 | 0.26265820051776245 | 0.2634999982547015 | 0.011309597121244989 | 0.247582996962592 | 0.27616599982138723 | cpu | float64 | measured_local_emulation | None | Stage 7.0 run_masked_lora_linear forward. |
| masked_lora_backward | fresh_masks_fresh_u_with_pad | 2 | 5 | 0.11773320002248511 | 0.11699999595293775 | 0.0025399268505254067 | 0.11537499813130125 | 0.12187500396976247 | cpu | float64 | measured_local_emulation | None | Stage 7.1 run_masked_lora_backward. |
| rank_padded_lora_forward | paired_cancellation_dummy | 2 | 5 | 0.26950839965138584 | 0.2671670008567162 | 0.00789179351680181 | 0.2630000017234124 | 0.2827499993145466 | cpu | float64 | measured_local_emulation | None | Stage 7.2 rank-padded masked forward. |
| multi_layer_lora_training_step | synthetic_tile | 2 | 5 | 4.684007998730522 | 4.685125000833068 | 0.08384449228170614 | 4.578499996569008 | 4.810165999515448 | cpu | float64 | measured_local_emulation | None | Stage 7.3 run_multilayer_lora_training (one training step). |
| modern_decoder_model_wrapper | opt_in_only | 2 | 5 | None | None | None | None | None | cpu | float64 | measured_local_emulation | modern_decoder_wrapper is opt-in (include_modern_decoder_wrapper=False) | Opt-in benchmark; pytest defaults stay synthetic. |

## Limitations

- This is local runtime emulation, not real TEE wall-time.
- No real sleep, no real runtime gating; ``time.perf_counter`` only.
- Workload tiles are small for pytest stability — absolute numbers are illustrative.
- Modern decoder model-wrapper benchmark is opt-in and recorded as skipped when unavailable.
- No formal / cryptographic / semantic security is claimed.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).
- Reports publish timing statistics only — raw tensors, raw adapters, raw gradients, and dense masks are never emitted.
