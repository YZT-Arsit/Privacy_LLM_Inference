# Stage 7.2 — LoRA Rank Padding / Hidden-Rank Prototype

## 1. Experiment Scope

- Single LoRA-augmented linear with d_in=32, d_out=16.
- true_rank=4, padded_rank=8, alpha=1.0, batch_size=4.
- optimizer=sgd, lr=0.01, num_steps=5, use_pad=True, fresh_u_per_step=True, dummy_strategy='paired_cancellation_dummy', dtype=float64.
- Synthetic private data; no network access; no PEFT integration.

## 2. Rank Padding Formula

```
A_pad ∈ R^{d_in × r_pad},     A_pad[:, :r] = A_real
B_pad ∈ R^{r_pad × d_out},    B_pad[:r, :] = B_real
Dummy slice:  A_pad[:, r:r_pad],  B_pad[r:r_pad, :]
Invariant:    A_pad B_pad = A_real B_real
LoRA scale:   alpha / true_rank   (NOT alpha / padded_rank)

Masked forward (rank dim = r_pad):
    Y_tilde = X_tilde W_tilde + (alpha / r) (X_tilde A_pad_tilde) B_pad_tilde
              + bias_tilde + pad_compensation

Masked backward (rank dim = r_pad):
    grad_A_pad_tilde = (alpha/r) X_tilde^T (G_tilde B_pad_tilde^T)
    grad_B_pad_tilde = (alpha/r) (X_tilde A_pad_tilde)^T G_tilde

Trusted side recovery + real-slice extraction:
    grad_A_pad = N_in^{-T} grad_A_pad_tilde U_pad^T (+ pad compensation)
    grad_B_pad = U_pad^{-T} grad_B_pad_tilde N_out^T (+ pad compensation)
    grad_A_real = grad_A_pad[:, :true_rank]
    grad_B_real = grad_B_pad[:true_rank, :]
```

## 3. Dummy Rank Strategy

- requested: `paired_cancellation_dummy`
- effective: `paired_cancellation_dummy`
- dummy_size: 4
- dummy_scale: 1.0
- fresh_dummy_per_step: True
- max dummy contribution norm across steps: 0.000e+00

## 4. Forward Correctness

| step | loss_plain | loss_padded | loss_diff | forward_err | dummy_contribution_norm |
|------|-----------|-------------|-----------|-------------|-------------------------|
| 0 | 4.932336e+01 | 4.932336e+01 | 4.263e-14 | 2.132e-14 | 0.000e+00 |
| 1 | 4.927997e+01 | 4.927997e+01 | 7.105e-15 | 2.176e-14 | 0.000e+00 |
| 2 | 4.923658e+01 | 4.923658e+01 | 3.553e-14 | 1.776e-14 | 0.000e+00 |
| 3 | 4.919305e+01 | 4.919305e+01 | 2.132e-14 | 1.954e-14 | 0.000e+00 |
| 4 | 4.914922e+01 | 4.914922e+01 | 1.421e-14 | 1.954e-14 | 0.000e+00 |

- max loss diff: 4.263e-14
- max dummy contribution norm: 0.000e+00

## 5. Backward Correctness

| step | grad_A_real_err | grad_B_real_err | adapter_A_update_err | adapter_B_update_err |
|------|-----------------|-----------------|----------------------|----------------------|
| 0 | 1.173e-15 | 1.887e-15 | 5.551e-17 | 1.908e-17 |
| 1 | 7.147e-16 | 7.633e-16 | 5.551e-17 | 1.388e-17 |
| 2 | 5.846e-16 | 1.221e-15 | 5.551e-17 | 1.041e-17 |
| 3 | 9.229e-16 | 6.939e-16 | 5.551e-17 | 1.388e-17 |
| 4 | 1.306e-15 | 1.221e-15 | 1.110e-16 | 2.082e-17 |

- max grad_A real err: 1.306e-15
- max grad_B real err: 1.887e-15
- final adapter_A update err: 1.110e-16
- final adapter_B update err: 2.082e-17
- allclose: **True**

## 6. Optimizer Handling

- location: **trusted**
- optimizer: sgd
- trainable_adapter_shape_a: [32, 4]
- trainable_adapter_shape_b: [4, 16]
- optimizer_state_shape_a: None
- optimizer_state_shape_b: None
- optimizer_state_contains_dummy: **False**
- dummy_update_applied: **False**
- note: Optimizer state (and the trainable A_real / B_real tensors) is sized to true_rank, never padded_rank. The dummy slice is re-sampled from scratch each step and never enters the optimizer.

## 7. Shape-Level Rank Hiding

- visible A_tilde_pad shape: [32, 8]
- visible B_tilde_pad shape: [8, 16]
- visible_rank_from_a_shape: **8**
- visible_rank_from_b_shape: **8**
- true_rank_hidden_from_shape: **True**
- padded_rank_visible: **True**
- note: Stage 7.2 hides true_rank from the dimensions of A_pad_tilde / B_pad_tilde / grad_A_pad_tilde / grad_B_pad_tilde. The padded_rank itself remains visible — see the security proxy for residual spectral / gradient-side inference risk.

## 8. Limitations

- Stage 7.2 hides true rank from tensor shape by exposing padded rank.
- Padded rank r_pad remains visible to the GPU.
- Dummy rank indistinguishability is evaluated by proxy (Stage 7.2 security proxy), not formally proven.
- Optimizer state remains trusted-only and is sized to true_rank, never padded_rank.
- Backward / loss computation remains trusted (Stage 7.1 contract): only G_tilde, A_pad_tilde, B_pad_tilde, grad_A_pad_tilde, grad_B_pad_tilde cross the boundary.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- No real TEE training.
- No full Qwen / TinyLlama / LLaMA LoRA fine-tuning.
- Adapter is NEVER merged into the public base weight W.
- No formal / cryptographic / semantic security is claimed.
- Reports publish summary metrics + fingerprints. Private data, raw adapters (real or padded), raw gradients, optimizer state, and dense masks are never emitted in outputs.

## 9. Next Stage Plan

- Stage 7.3 — multi-layer LoRA end-to-end + LoRA training timing-side proxy.
- Stage 7.x — stronger spectral / gradient-side dummy strategies that resist rank inference proxies.
