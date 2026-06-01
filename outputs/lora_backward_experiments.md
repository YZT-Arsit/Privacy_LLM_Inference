# Stage 7.1 — LoRA Masked Backward / Gradient-Side Obfuscation

## 1. Experiment Scope

- Single LoRA-augmented linear, d_in=32, d_out=16, rank=4, alpha=1.0, batch_size=4.
- optimizer=sgd, lr=0.01, num_steps=5, use_pad=True, fresh_u_per_step=True, recover_grad_x=True, dtype=float64.
- Synthetic private data; no network access; no PEFT integration.

## 2. Masked LoRA Backward Formula

```
Upstream gradient mask: G_tilde = G N_out^{-T}
chain rule invariance:  tr(G^T dY) = tr(G_tilde^T dY_tilde) for Y_tilde = Y N_out
grad_A_tilde            = s X_tilde^T (G_tilde B_tilde^T)
grad_B_tilde            = s (X_tilde A_tilde)^T G_tilde
grad_X_tilde (optional) = G_tilde W_tilde^T + s G_tilde B_tilde^T A_tilde^T
grad_A recovery         = grad_A = N_in^{-T} grad_A_tilde U^T (+ trusted pad compensation)
grad_B recovery         = grad_B = U^{-T}    grad_B_tilde N_out^T (+ trusted pad compensation)
grad_X recovery         = grad_X = grad_X_tilde N_in^T
```

## 3. Upstream Gradient Masking

- max upstream-gradient invariance error |tr(G^T Y) - tr(G_tilde^T Y_tilde)| = 1.421e-14
- step-0 autograd vs analytic plain reference: grad_A_err=4.337e-19, grad_B_err=0.000e+00, grad_X_err=0.000e+00

## 4. Grad-A / Grad-B Recovery

| step | loss_diff | forward_err | grad_A_err | grad_B_err | grad_X_err |
|------|-----------|-------------|-----------|-----------|-----------|
| 0 | 1.421e-14 | 1.621e-14 | 2.602e-18 | 7.772e-16 | 3.109e-15 |
| 1 | 1.421e-14 | 1.776e-14 | 2.949e-17 | 5.551e-16 | 2.554e-15 |
| 2 | 4.263e-14 | 2.842e-14 | 4.510e-17 | 9.714e-16 | 5.773e-15 |
| 3 | 3.553e-14 | 2.309e-14 | 8.327e-17 | 9.992e-16 | 6.661e-15 |
| 4 | 0.000e+00 | 2.132e-14 | 6.592e-17 | 3.955e-16 | 2.998e-15 |

- max grad_A err: 8.327e-17
- max grad_B err: 9.992e-16
- max grad_X err: 6.661e-15
- masked_backward_allclose: **True**

## 5. Training-Step Correctness

| step | adapter_A_update_err | adapter_B_update_err |
|------|---------------------|---------------------|
| 0 | 0.000e+00 | 7.806e-18 |
| 1 | 0.000e+00 | 1.214e-17 |
| 2 | 0.000e+00 | 1.388e-17 |
| 3 | 1.388e-17 | 2.082e-17 |
| 4 | 1.388e-17 | 2.429e-17 |

- final adapter_A update err: 1.388e-17
- final adapter_B update err: 2.429e-17
- final output err: 1.776e-14
- allclose: **True**

## 6. Optimizer Handling

| variable | visible_to_gpu |
|----------|----------------|
| x_tilde | True |
| w_tilde | True |
| a_tilde | True |
| b_tilde | True |
| grad_y_tilde | True |
| grad_a_tilde | True |
| grad_b_tilde | True |
| grad_x_tilde | True |
| raw_x | False |
| raw_a | False |
| raw_b | False |
| raw_grad_a | False |
| raw_grad_b | False |
| raw_upstream_gradient_g | False |
| optimizer_state | False |
| private_target_y | False |

- loss computation remains trusted (trusted_loss): Trusted side computes L = MSE(Y_recovered, Y_target) and G = dL/dY = (2/N)(Y - Y_target). Only G_tilde = G N_out^{-T} is sent to the GPU backward.
- backward arithmetic on GPU: masked_backward_prototype (masked tensors only).
- optimizer remains trusted (trusted_optimizer, optimizer=sgd, lr=0.01): Optimizer state (SGD momentum / AdamW m, v) lives entirely on the trusted side and is never exposed to the GPU or to JSON / CSV / Markdown reports.
- pad compensation (trusted-only): (alpha / r) T_in^T G B^T   (trusted side, plain space); (alpha / r) A^T T_in^T G   (trusted side, plain space)

## 7. Limitations

- Stage 7.1 implements masked-gradient prototype, not full private fine-tuning.
- Loss computation remains trusted (G = dL/dY is computed on the trusted side).
- Optimizer update remains trusted (SGD momentum / AdamW m, v never cross the boundary).
- PEFT / DeepSpeed / vLLM / FlashAttention are NOT integrated.
- This is not real TEE training; security_profile stays 'proxy-evaluated, not formal'.
- Rank padding is NOT implemented; LoRA rank r is still visible from A_tilde / B_tilde / grad_A_tilde / grad_B_tilde shapes (deferred to Stage 7.2).
- LoRA adapter is NEVER merged into the public base weight W.
- Distributed training is NOT implemented.
- Reports publish summary metrics + fingerprints only. Private data, raw adapter tensors, raw gradients, optimizer state, and masks are never emitted in outputs.
- No formal / cryptographic / semantic security is claimed.

## 8. Next Stage Plan

- Stage 7.2 — rank padding to hide r from A_tilde / B_tilde / grad_A_tilde / grad_B_tilde shapes.
- Stage 7.3 — multi-layer LoRA in a tiny transformer block end-to-end; calibrated LoRA workload + LoRA timing-side proxy.
