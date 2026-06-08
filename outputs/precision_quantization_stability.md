# Precision / Quantization Stability

_Stage 7.8b: CPU-simulated precision modes (fp32, bf16, fp16, int8, int4) and condition-number sweep for the padded-boundary linear under various mask families. No real GPU / quantized kernel is measured._

## Configuration

| Field | Value |
|---|---|
| batch_size | 2 |
| seq_len | 4 |
| in_dim | 64 |
| out_dim | 64 |
| vocab_size | 97 |
| condition_numbers | [1.0, 2.0, 10.0, 100.0, 1000.0] |
| pad_scale | 0.5 |

## Why Float64 Correctness Does NOT Imply Low-Precision Correctness

The padded-boundary linear computes ``Y_rec = (X - T) M M^{-1} W + T W``. In exact arithmetic ``M M^{-1} = I`` cancels and ``Y_rec = X W``. At lower precision, ``M M^{-1}`` carries a residual ~ ``eps * cond(M)`` which is amplified by ``W`` and the activations. For well-conditioned ``M`` (orthogonal: cond = 1) the residual stays at machine epsilon; for ill-conditioned ``M`` the residual scales linearly with cond(M).

## Orthogonal Mask (cond = 1.0)

| precision | logits_max_abs_error | logits_relative_error | greedy_match | seq_exact | overflow | nan |
|---|---|---|---|---|---|---|
| `float64_reference` | 2.842e-14 | 9.371e-16 | 1.0 | True | False | False |
| `float32_simulated` | 1.475e-06 | 4.863e-08 | 1.0 | True | False | False |
| `bfloat16_simulated` | 0.0881211 | 0.00290532 | 1.0 | True | False | False |
| `float16_simulated` | 0.0105718 | 3.485e-04 | 1.0 | True | False | False |
| `int8_weight_only_simulated` | 0.313669 | 0.0103415 | 1.0 | True | False | False |
| `int4_weight_only_symbolic` | 5.35036 | 0.176399 | 0.75 | False | False | False |

## Permutation Mask (cond = 1.0)

| precision | logits_max_abs_error | logits_relative_error | greedy_match | seq_exact | overflow | nan |
|---|---|---|---|---|---|---|
| `float64_reference` | 2.132e-14 | 8.750e-16 | 1.0 | True | False | False |
| `float32_simulated` | 9.898e-07 | 4.063e-08 | 1.0 | True | False | False |
| `bfloat16_simulated` | 0.0858695 | 0.0035248 | 1.0 | True | False | False |
| `float16_simulated` | 0.00840632 | 3.451e-04 | 1.0 | True | False | False |
| `int8_weight_only_simulated` | 0.18223 | 0.00748021 | 1.0 | True | False | False |
| `int4_weight_only_symbolic` | 3.03522 | 0.12459 | 0.875 | False | False | False |

## Dense Condition-Number Sweep

| cond_target | actual_cond | precision | logits_max_abs_error | logits_relative_error | greedy_match |
|---|---|---|---|---|---|
| dense_condition_1 | 1 | `float64_reference` | 3.375e-14 | 1.192e-15 | 1.0 |
| dense_condition_1 | 1 | `float32_simulated` | 1.762e-06 | 6.219e-08 | 1.0 |
| dense_condition_1 | 1 | `bfloat16_simulated` | 0.0798349 | 0.00281843 | 1.0 |
| dense_condition_1 | 1 | `float16_simulated` | 0.0128396 | 4.533e-04 | 1.0 |
| dense_condition_1 | 1 | `int8_weight_only_simulated` | 0.273249 | 0.00964657 | 1.0 |
| dense_condition_1 | 1 | `int4_weight_only_symbolic` | 4.97213 | 0.175532 | 0.875 |
| dense_condition_2 | 2 | `float64_reference` | 3.286e-14 | 1.325e-15 | 1.0 |
| dense_condition_2 | 2 | `float32_simulated` | 1.713e-06 | 6.908e-08 | 1.0 |
| dense_condition_2 | 2 | `bfloat16_simulated` | 0.0933726 | 0.00376586 | 1.0 |
| dense_condition_2 | 2 | `float16_simulated` | 0.0136052 | 5.487e-04 | 1.0 |
| dense_condition_2 | 2 | `int8_weight_only_simulated` | 0.320037 | 0.0129075 | 1.0 |
| dense_condition_2 | 2 | `int4_weight_only_symbolic` | 6.18372 | 0.249399 | 0.875 |
| dense_condition_10 | 10 | `float64_reference` | 6.661e-14 | 2.505e-15 | 1.0 |
| dense_condition_10 | 10 | `float32_simulated` | 2.725e-06 | 1.025e-07 | 1.0 |
| dense_condition_10 | 10 | `bfloat16_simulated` | 0.195682 | 0.0073597 | 1.0 |
| dense_condition_10 | 10 | `float16_simulated` | 0.0263264 | 9.901e-04 | 1.0 |
| dense_condition_10 | 10 | `int8_weight_only_simulated` | 0.624933 | 0.023504 | 0.875 |
| dense_condition_10 | 10 | `int4_weight_only_symbolic` | 10.3965 | 0.391016 | 0.375 |
| dense_condition_100 | 100 | `float64_reference` | 3.335e-13 | 1.262e-14 | 1.0 |
| dense_condition_100 | 100 | `float32_simulated` | 1.214e-05 | 4.594e-07 | 1.0 |
| dense_condition_100 | 100 | `bfloat16_simulated` | 0.85609 | 0.0324004 | 1.0 |
| dense_condition_100 | 100 | `float16_simulated` | 0.117504 | 0.00444718 | 1.0 |
| dense_condition_100 | 100 | `int8_weight_only_simulated` | 2.90644 | 0.11 | 1.0 |
| dense_condition_100 | 100 | `int4_weight_only_symbolic` | 54.1611 | 2.04983 | 0.125 |
| dense_condition_1000 | 1000 | `float64_reference` | 2.378e-12 | 1.034e-13 | 1.0 |
| dense_condition_1000 | 1000 | `float32_simulated` | 7.981e-05 | 3.470e-06 | 1.0 |
| dense_condition_1000 | 1000 | `bfloat16_simulated` | 4.87113 | 0.211769 | 0.625 |
| dense_condition_1000 | 1000 | `float16_simulated` | 0.748346 | 0.0325338 | 1.0 |
| dense_condition_1000 | 1000 | `int8_weight_only_simulated` | 16.057 | 0.698065 | 0.25 |
| dense_condition_1000 | 1000 | `int4_weight_only_symbolic` | 374.316 | 16.2731 | 0.0 |

## Recommendations

Recommended mask families for low-precision deployment:

- orthogonal
- permutation
- RoPE-plane block rotation
- block-diagonal well-conditioned

NOT recommended for low precision:

- ill-conditioned dense masks (condition number >> 1)

## Limitations

- CPU local emulation only; no real GPU fp16 / bf16 / int8 / int4 kernels are measured.
- bf16 / fp16 / int8 are SIMULATED via round-trip casts on float64 storage. Real GPU tensor-core behaviour may differ in accumulator type and rounding.
- int4 is SYMBOLIC ONLY -- no real int4 path.
- float64 reference is for protocol correctness, NOT real inference precision; real LLM inference uses bf16 or fp16 plus mixed-precision accumulation.
- Condition-number sweep uses synthetic dense masks; real-world quantized weights have their own conditioning.
- No formal cryptographic / semantic / differential-privacy security.
- No full Qwen / LLaMA deployment.

## Paper-Safe Wording

> Mask transformations are stable for orthogonal / permutation / RoPE-plane block-diagonal masks under every simulated precision mode; ill-conditioned dense masks amplify error proportionally to the condition number. We recommend well-conditioned mask families (orthogonal, permutation, block) for low-precision deployment. We do NOT measure real GPU fp16 / bf16 / int8 / int4 wall-clock or hardware-specific rounding.

## Unsafe Wording to Avoid

- Real GPU fp16 / bf16 / int8 / int4 performance.
- Real quantized model deployment.
- GPU tensor-core matmul measured.
- This is formal cryptographic security.

