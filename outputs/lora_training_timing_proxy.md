# Stage 7.3 — LoRA Training Timing Side-Channel Proxy

## 1. Experiment Scope

- batch_sizes=(1, 2, 4, 8), seq_lens=(4, 8, 16), true_ranks=(2, 4, 8), padded_ranks=(8, 16).
- num_lora_modules=(2, 4, 7, 14), optimizers=('sgd', 'adamw').
- timing_noise_std=0.05, samples_per_config=8, base_hidden=64, base_intermediate=128.
- constant_time_training_mode=`proxy_equalized`.
- scope: FLOP / boundary cost-model proxy for LoRA training step latency; not real TEE wall-time

## 2. Training Timing Proxy Model

Total per-step latency proxy:  ```  latency = base_overhead_ms          + forward_ms          + backward_ms          + optimizer_ms          + mask_generation_ms          + boundary_ms          + rank_padding_dummy_ms          + Gaussian timing_noise(std=timing_noise_std) * total  ```
Cost-model constants:
- gpu_cost_per_flop_ms: 1e-09
- trusted_cost_per_flop_ms: 4e-09
- mask_gen_cost_per_flop_ms: 4e-09
- boundary_call_cost_ms: 0.02
- base_overhead_ms: 0.1
- rank_padding_dummy_cost_ms: 0.01
- timing_noise_std: 0.05

- num_samples_default: 4608
- num_samples_rank_padding_off: 4608
- num_samples_zero_dummy: 4608
- num_samples_paired_dummy: 4608
- note: Each sample = one training step latency under one (batch_size, seq_len, true_rank, padded_rank, num_modules, optimizer, dummy_strategy, rank_padding_on) configuration, with Gaussian timing noise applied.

## 3. Leakage Tasks

Constant-time mode: **off**

| task | accuracy | chance | bucket_separation | risk |
|------|----------|--------|--------------------|------|
| batch_size | 0.259 | 0.250 | 0.017 | low |
| seq_len | 0.340 | 0.333 | 0.013 | low |
| true_rank | 0.336 | 0.333 | 0.002 | low |
| padded_rank | 0.497 | 0.500 | 0.003 | low |
| num_modules | 1.000 | 0.250 | 1.539 | high |
| optimizer | 0.507 | 0.500 | 0.002 | low |
| rank_padding_on | 0.625 | 0.500 | 0.499 | low |
| dummy_strategy | 0.572 | 0.500 | 0.107 | low |

Constant-time mode: **proxy_equalized**

| task | accuracy | chance | bucket_separation | risk |
|------|----------|--------|--------------------|------|
| batch_size | 0.260 | 0.250 | 0.004 | low |
| seq_len | 0.345 | 0.333 | 0.005 | low |
| true_rank | 0.339 | 0.333 | 0.002 | low |
| padded_rank | 0.512 | 0.500 | 0.002 | low |
| num_modules | 0.256 | 0.250 | 0.002 | low |
| optimizer | 0.499 | 0.500 | 0.000 | low |
| rank_padding_on | 0.506 | 0.500 | 0.001 | low |
| dummy_strategy | 0.501 | 0.500 | 0.000 | low |

## 4. Constant-Time Training Proxy

- constant_time_training_mode: `proxy_equalized`
- did_actually_sleep: **False**
- upper_batch_size: 8
- upper_seq_len: 16
- upper_padded_rank: 16
- upper_num_modules: 14
- upper_optimizer: adamw
- upper_latency_ms: 1.3632453119999999
- note: proxy_equalized pads every step to the upper-bucket latency; we never invoke real sleep / runtime gating.

## 5. Overhead Estimate

- mean_native_latency_ms: 0.6789
- upper_latency_ms: 1.3632
- overhead_ratio: 1.0080
- overhead_pct: 100.80%
- note: Proxy estimate; no real wall-clock measurement. The equalization pads every step to the upper bucket latency across sensitive dimensions, so the overhead is the maximum of the original variance.

## 6. Limitations

- Timing results are proxy estimates from an FLOP-and-boundary cost model, not real TEE wall-time.
- Constant-time training mode is simulated (upper-bound latency padding); the proxy does NOT sleep, does NOT modify real runtime.
- Hardware side-channels (cache / power / EM) are NOT evaluated.
- Cost-model constants are coarse; absolute latencies are illustrative.
- Leakage classifier uses bucket-mean labels as the attacker; this is a generous attacker model.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- No real Qwen / TinyLlama / LLaMA fine-tuning; this is a multi-linear cost-model proxy.
- Adapter is NEVER merged into the public base weight W.
- No formal / cryptographic / semantic security is claimed.

## 7. Next Stage Plan

- Stage 7.4 — stronger dummy distributions / spectral-rank hardening.
- Stage 7.x — real TEE wall-time integration with actual constant-time gating.
- Stage 7.x — hardware side-channel (cache / power) proxies.
