# Measured Runtime (Local Emulation, NOT real TEE)

_This is local runtime emulation, not real TEE wall-time. No real sleep, no real runtime gating._

| component | variant | num_warmup | num_repeats | mean_ms | median_ms | std_ms | min_ms | max_ms | device | dtype | wall_time_source | skipped_with_reason | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_synthetic_linear | X W | 2 | 5 | 0.0017916012438945472 | 0.0017080019460991025 | 0.00025515530856451467 | 0.0015420009731315076 | 0.002082997525576502 | cpu | float64 | measured_local_emulation | None | Synthetic baseline; no obfuscation. |
| plain_lora_forward | plain_rank_r | 2 | 5 | 0.007958199421409518 | 0.007791000825818628 | 0.0006766130077735035 | 0.007333001121878624 | 0.00904199987417087 | cpu | float64 | measured_local_emulation | None | Plain rank-r LoRA forward; no masking. |
| masked_lora_forward | fresh_masks_fresh_u_with_pad | 2 | 5 | 0.2471915984642692 | 0.24466699687764049 | 0.015932198492687927 | 0.22724999871570617 | 0.27033299556933343 | cpu | float64 | measured_local_emulation | None | Stage 7.0 run_masked_lora_linear forward. |
| masked_lora_backward | fresh_masks_fresh_u_with_pad | 2 | 5 | 0.11866640124935657 | 0.11804100358858705 | 0.0036911838356562545 | 0.11600000289035961 | 0.1249999986612238 | cpu | float64 | measured_local_emulation | None | Stage 7.1 run_masked_lora_backward. |
| rank_padded_lora_forward | paired_cancellation_dummy | 2 | 5 | 0.27414139913162217 | 0.2795830005197786 | 0.01261367039715004 | 0.2536250030971132 | 0.28458299493649974 | cpu | float64 | measured_local_emulation | None | Stage 7.2 rank-padded masked forward. |
| multi_layer_lora_training_step | synthetic_tile | 2 | 5 | 4.635358400992118 | 4.605207999702543 | 0.07070563046014282 | 4.554750004899688 | 4.732041998067871 | cpu | float64 | measured_local_emulation | None | Stage 7.3 run_multilayer_lora_training (one training step). |
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
