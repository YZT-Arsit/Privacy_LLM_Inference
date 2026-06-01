# Stage 7.0 — LoRA Private Training Prototype

## 1. Experiment Scope

- Tiny LoRA-augmented linear with d_in=32, d_out=16, rank=4, alpha=1.0, batch_size=4.
- Optimizer = sgd, lr=0.01, num_steps=5.
- use_pad=True, fresh_u_per_step=True, fresh_masks_per_step=True, dtype=float64.
- Synthetic private data; no network access; no PEFT integration.

## 2. LoRA Linear Masking Formula

```
Plain:  Y = X W + (alpha / r) X A B + bias
Masked: X_tilde     = (X - T_in) N_in    (or X N_in when use_pad=False)
        W_tilde     = N_in^{-1} W N_out
        A_tilde     = N_in^{-1} A U
        B_tilde     = U^{-1}   B N_out
        bias_tilde  = bias N_out
        C_W         = T_in W N_out
        C_LoRA      = (alpha / r) T_in A B N_out
        Y_tilde     = X_tilde W_tilde
                    + (alpha / r) (X_tilde A_tilde) B_tilde
                    + bias_tilde + C_W + C_LoRA
        Y_recovered = Y_tilde N_out^{-1}
```

LoRA adapter is NEVER merged into the public base weight W.

## 3. Forward Correctness

| step | loss_plain | loss_masked | loss_diff | forward_err |
|------|-----------|-------------|-----------|-------------|
| 0 | 4.932336e+01 | 4.932336e+01 | 1.421e-14 | 1.621e-14 |
| 1 | 4.927997e+01 | 4.927997e+01 | 1.421e-14 | 1.776e-14 |
| 2 | 4.923658e+01 | 4.923658e+01 | 4.263e-14 | 2.842e-14 |
| 3 | 4.919305e+01 | 4.919305e+01 | 3.553e-14 | 2.309e-14 |
| 4 | 4.914922e+01 | 4.914922e+01 | 0.000e+00 | 2.132e-14 |

- max loss diff: 4.263e-14
- final output err: 1.776e-14
- allclose: **True**

## 4. Training-Step Correctness

| step | grad_A_err | grad_B_err | adapter_A_update_err | adapter_B_update_err |
|------|-----------|-----------|---------------------|---------------------|
| 0 | 2.168e-18 | 5.551e-16 | 0.000e+00 | 5.204e-18 |
| 1 | 4.770e-18 | 6.661e-16 | 0.000e+00 | 1.041e-17 |
| 2 | 1.388e-17 | 6.384e-16 | 0.000e+00 | 1.214e-17 |
| 3 | 4.163e-17 | 8.396e-16 | 0.000e+00 | 1.735e-17 |
| 4 | 2.776e-17 | 5.551e-16 | 4.337e-19 | 1.735e-17 |

- max grad_A err: 4.163e-17
- max grad_B err: 8.396e-16
- final adapter_A update err: 4.337e-19
- final adapter_B update err: 1.735e-17

## 5. Gradient / Optimizer Handling

| variable | visible_to_gpu |
|----------|----------------|
| x_tilde | True |
| w_tilde | True |
| a_tilde | True |
| b_tilde | True |
| bias_tilde | True |
| pad_compensation | True |
| raw_x | False |
| raw_a | False |
| raw_b | False |
| grad_a | False |
| grad_b | False |
| optimizer_state | False |
| private_target_y | False |

- backward_location: **trusted**
- optimizer_state_location: **trusted**
- adapter_location: **trusted**
- merge_adapter_into_w: **False**
- trusted_backward_status: training backward remains trusted in Stage 7.0 prototype
- masked_backward_status: not_implemented; deferred to Stage 7.1 (masked backward / gradient-side obfuscation)

## 6. Pad Compensation

- use_pad: **True**
- pad_scale: 1.0
- compensation_formula: `C = T_in W N_out + (alpha / r) T_in A B N_out`
- trusted_only: True
- forward_err_under_pad: 2.842e-14

## 7. Limitations

- Stage 7.0 is a prototype LoRA private training path, not full Qwen/TinyLlama LoRA fine-tuning.
- Backward / optimizer update remains trusted in Stage 7.0 — only the forward is masked and offloaded.
- Optimizer state (SGD momentum / AdamW moments) is trusted-only and never exported to JSON/CSV/Markdown.
- PEFT / DeepSpeed / vLLM / FlashAttention are NOT integrated.
- Real TEE isolation is NOT evaluated; security_profile stays 'proxy-evaluated, not formal'.
- LoRA adapter is NEVER merged into the public base weight W.
- Distributed training is NOT implemented.
- Reported metrics are summary statistics + fingerprints. Private data, raw adapter tensors, masks, and pads are never emitted in outputs.
- No formal / cryptographic / semantic security is claimed.

## 8. Next Stage Plan

- Stage 7.1 — masked backward path: send masked gradients to GPU (e.g. fold N into the gradient pipeline) so the trusted side only generates and applies the update.
- Stage 7.2 — multi-layer LoRA in a tiny transformer block end-to-end.
- Stage 7.3 — calibrated LoRA workload + LoRA timing-side proxy.
