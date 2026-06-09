# Stage 7.6 — LoRA Training-to-Inference Lifecycle Report

## 1. Scope

This report enumerates which objects are plaintext / trusted-only, GPU-visible masked, or public during each phase of the Stage 7.6 masked-gradient LoRA training-to-inference lifecycle. It is a paper-claims consistency artifact: it asserts the visibility classification implied by the construction but is not itself a formal information-flow proof.

## 2. Visibility classes

- `plaintext_trusted_only`: held only on the trusted (user) side.
- `gpu_visible_masked`: uploaded to the cloud accelerator in masked form.
- `public`: known to the cloud accelerator and any external observer.

## 3. Invariants

| invariant | value |
|---|---|
| `plaintext_A_or_B_visible_to_gpu` | False |
| `plaintext_grad_A_or_B_visible_to_gpu` | False |
| `masks_visible_to_gpu` | False |
| `plaintext_optimizer_state_visible_to_gpu` | False |
| `user_input_visible_to_gpu` | False |
| `base_model_W_public` | True |
| `base_model_W_transformed_to_preserve_hidden_state_masks` | True |
| `gpu_sees_masked_hidden_states` | True |
| `gpu_sees_masked_lora_adapters` | True |
| `gpu_sees_masked_lora_gradients` | True |
| `gpu_sees_masked_momentum_buffers` | True |
| `adamw_under_dense_masks_supported` | False |

## 4. Lifecycle table

| phase | object | visibility | exposed_form | note |
|---|---|---|---|---|
| `lora_initialization` | `A_real, B_real` | `plaintext_trusted_only` | never_exported | Plaintext LoRA factors live only on the trusted side. |
| `lora_initialization` | `N_x, N_y, M` | `plaintext_trusted_only` | never_exported | Orthogonal masks held only by trusted side. |
| `lora_initialization` | `A_pad, B_pad` | `plaintext_trusted_only` | never_exported | Cancellation-padded factors: A_pad=[A_real, R, -R], B_pad=vstack(B_real, S, S); dummy contribution is zero at init. |
| `lora_initialization` | `A_tilde, B_tilde` | `gpu_visible_masked` | N_x^T A_pad M, M^T B_pad N_y | Masked, rank-padded LoRA factors uploaded to the GPU. Visible padded rank, true rank hidden. |
| `lora_initialization` | `base_model_W` | `public` | transformed_to_preserve_hidden_state_masks (e.g., boundary linear absorbs N_x / N_y on trusted side) | Base weights are publicly known but their use on the GPU is composed with the trusted-side mask boundary so that hidden-state masks survive the linear path. |
| `masked_forward` | `user_input / token_ids` | `plaintext_trusted_only` | never_exported | Tokenisation, embedding lookup happen trusted-side. |
| `masked_forward` | `X (plaintext hidden state)` | `plaintext_trusted_only` | never_exported | Plaintext hidden state never crosses the boundary. |
| `masked_forward` | `X_tilde = X N_x` | `gpu_visible_masked` | X @ N_x (masked) | Only masked hidden state is uploaded. |
| `masked_forward` | `A_tilde, B_tilde` | `gpu_visible_masked` | N_x^T A M, M^T B N_y | GPU sees masked, rank-padded LoRA factors only. |
| `masked_forward` | `Y_tilde = X_tilde A_tilde B_tilde` | `gpu_visible_masked` | masked LoRA output | Algebraically equals X A B N_y so the trusted side recovers Y by Y_tilde N_y^T. |
| `masked_forward` | `Y (plaintext output)` | `plaintext_trusted_only` | never_exported | Recovered only on the trusted side. |
| `masked_backward` | `target_tilde = target @ N_y` | `gpu_visible_masked` | target @ N_y (masked) | MSE loss is computed against masked target. Orthogonal N_y preserves the L2 loss exactly. Plain target never leaves the trusted side. |
| `masked_backward` | `grad_Y (plaintext)` | `plaintext_trusted_only` | never_exported | Plaintext output gradient never crosses the boundary. |
| `masked_backward` | `grad_Y_tilde = 2 (Y_tilde - target_tilde) / n` | `gpu_visible_masked` | masked output gradient | Equals 2 (Y - target) N_y / n; orthogonal masks commute through MSE. |
| `masked_backward` | `grad_A_tilde` | `gpu_visible_masked` | N_x^T grad_A M (masked) | Algebraic equivalence verified per step at float64 machine precision. |
| `masked_backward` | `grad_B_tilde` | `gpu_visible_masked` | M^T grad_B N_y (masked) | Algebraic equivalence verified per step at float64 machine precision. |
| `masked_backward` | `grad_A, grad_B (plaintext)` | `plaintext_trusted_only` | never_exported | Plaintext gradients never leave the trusted side. |
| `masked_sgd` | `A_tilde_next, B_tilde_next` | `gpu_visible_masked` | A_tilde - lr * grad_A_tilde, B_tilde - lr * grad_B_tilde | Linear update; right-multiplication by orthogonal masks distributes, so masked SGD is algebraically equivalent to plaintext SGD after trusted-side recovery. |
| `masked_sgd` | `plaintext optimizer state` | `plaintext_trusted_only` | never_exported | SGD has no persistent state besides the parameters. |
| `masked_momentum_sgd` | `V_A_tilde, V_B_tilde (masked momentum buffers)` | `gpu_visible_masked` | V_tilde <- mu V_tilde + grad_tilde; param_tilde <- param_tilde - lr V_tilde | Heavy-ball update is linear in the gradients; orthogonal masks commute, so masked momentum SGD is algebraically equivalent to plaintext momentum SGD after recovery. |
| `masked_momentum_sgd` | `plaintext momentum buffers V_A, V_B` | `plaintext_trusted_only` | never_exported | Plaintext momentum buffers never live on the GPU; they can be recovered from the masked buffers on the trusted side only when needed. |
| `masked_adamw_unsupported` | `AdamW second moments` | `plaintext_trusted_only` | masked_adamw_step_unsupported raises DenseMaskedAdamWUnsupported | Coordinate-wise second moments are not invariant under dense orthogonal mixing; AdamW under dense masks is unsupported. |
| `final_adapter_recovery_or_audit` | `A_pad_recovered, B_pad_recovered` | `plaintext_trusted_only` | A_pad = N_x A_tilde M^T, B_pad = M B_tilde N_y^T (trusted side) | Recovery uses the orthogonal inverses; never executed on the GPU. |
| `final_adapter_recovery_or_audit` | `A_real, B_real (trained)` | `plaintext_trusted_only` | extracted from A_pad / B_pad (trusted side) | True rank slices used for downstream inference; never exposed to the GPU in plaintext. |
| `final_adapter_recovery_or_audit` | `published fingerprints` | `public` | shapes and 16-char SHA-256 prefixes only | Outputs publish summary scalars and short fingerprints to enable third-party audit without exposing raw tensors. |
| `trained_lora_inference` | `user_input / token_ids` | `plaintext_trusted_only` | never_exported | As in masked forward, tokens are trusted-side only. |
| `trained_lora_inference` | `X_infer (plaintext hidden state)` | `plaintext_trusted_only` | never_exported | Plaintext hidden state not uploaded. |
| `trained_lora_inference` | `X_infer_tilde = X_infer N_x` | `gpu_visible_masked` | masked hidden state | GPU only sees the masked hidden state. |
| `trained_lora_inference` | `A_tilde, B_tilde (trained)` | `gpu_visible_masked` | N_x^T A_real_pad M, M^T B_real_pad N_y | Trained masked LoRA factors uploaded for inference; true rank still hidden behind cancellation padding. |
| `trained_lora_inference` | `base_model_W` | `public` | transformed_to_preserve_hidden_state_masks at the boundary | Base model is public but its composition with the trusted-side mask boundary preserves hidden-state masks at inference time. |
| `trained_lora_inference` | `Y_infer (plaintext output)` | `plaintext_trusted_only` | never_exported | Final output recovered on the trusted side by Y_infer = Y_infer_tilde @ N_y^T. |

## 5. Per-phase summary

### lora_initialization

- `A_real, B_real` -- **plaintext_trusted_only** -- Plaintext LoRA factors live only on the trusted side.
- `N_x, N_y, M` -- **plaintext_trusted_only** -- Orthogonal masks held only by trusted side.
- `A_pad, B_pad` -- **plaintext_trusted_only** -- Cancellation-padded factors: A_pad=[A_real, R, -R], B_pad=vstack(B_real, S, S); dummy contribution is zero at init.
- `A_tilde, B_tilde` -- **gpu_visible_masked** -- Masked, rank-padded LoRA factors uploaded to the GPU. Visible padded rank, true rank hidden.
- `base_model_W` -- **public** -- Base weights are publicly known but their use on the GPU is composed with the trusted-side mask boundary so that hidden-state masks survive the linear path.

### masked_forward

- `user_input / token_ids` -- **plaintext_trusted_only** -- Tokenisation, embedding lookup happen trusted-side.
- `X (plaintext hidden state)` -- **plaintext_trusted_only** -- Plaintext hidden state never crosses the boundary.
- `X_tilde = X N_x` -- **gpu_visible_masked** -- Only masked hidden state is uploaded.
- `A_tilde, B_tilde` -- **gpu_visible_masked** -- GPU sees masked, rank-padded LoRA factors only.
- `Y_tilde = X_tilde A_tilde B_tilde` -- **gpu_visible_masked** -- Algebraically equals X A B N_y so the trusted side recovers Y by Y_tilde N_y^T.
- `Y (plaintext output)` -- **plaintext_trusted_only** -- Recovered only on the trusted side.

### masked_backward

- `target_tilde = target @ N_y` -- **gpu_visible_masked** -- MSE loss is computed against masked target. Orthogonal N_y preserves the L2 loss exactly. Plain target never leaves the trusted side.
- `grad_Y (plaintext)` -- **plaintext_trusted_only** -- Plaintext output gradient never crosses the boundary.
- `grad_Y_tilde = 2 (Y_tilde - target_tilde) / n` -- **gpu_visible_masked** -- Equals 2 (Y - target) N_y / n; orthogonal masks commute through MSE.
- `grad_A_tilde` -- **gpu_visible_masked** -- Algebraic equivalence verified per step at float64 machine precision.
- `grad_B_tilde` -- **gpu_visible_masked** -- Algebraic equivalence verified per step at float64 machine precision.
- `grad_A, grad_B (plaintext)` -- **plaintext_trusted_only** -- Plaintext gradients never leave the trusted side.

### masked_sgd

- `A_tilde_next, B_tilde_next` -- **gpu_visible_masked** -- Linear update; right-multiplication by orthogonal masks distributes, so masked SGD is algebraically equivalent to plaintext SGD after trusted-side recovery.
- `plaintext optimizer state` -- **plaintext_trusted_only** -- SGD has no persistent state besides the parameters.

### masked_momentum_sgd

- `V_A_tilde, V_B_tilde (masked momentum buffers)` -- **gpu_visible_masked** -- Heavy-ball update is linear in the gradients; orthogonal masks commute, so masked momentum SGD is algebraically equivalent to plaintext momentum SGD after recovery.
- `plaintext momentum buffers V_A, V_B` -- **plaintext_trusted_only** -- Plaintext momentum buffers never live on the GPU; they can be recovered from the masked buffers on the trusted side only when needed.

### final_adapter_recovery_or_audit

- `A_pad_recovered, B_pad_recovered` -- **plaintext_trusted_only** -- Recovery uses the orthogonal inverses; never executed on the GPU.
- `A_real, B_real (trained)` -- **plaintext_trusted_only** -- True rank slices used for downstream inference; never exposed to the GPU in plaintext.
- `published fingerprints` -- **public** -- Outputs publish summary scalars and short fingerprints to enable third-party audit without exposing raw tensors.

### trained_lora_inference

- `user_input / token_ids` -- **plaintext_trusted_only** -- As in masked forward, tokens are trusted-side only.
- `X_infer (plaintext hidden state)` -- **plaintext_trusted_only** -- Plaintext hidden state not uploaded.
- `X_infer_tilde = X_infer N_x` -- **gpu_visible_masked** -- GPU only sees the masked hidden state.
- `A_tilde, B_tilde (trained)` -- **gpu_visible_masked** -- Trained masked LoRA factors uploaded for inference; true rank still hidden behind cancellation padding.
- `base_model_W` -- **public** -- Base model is public but its composition with the trusted-side mask boundary preserves hidden-state masks at inference time.
- `Y_infer (plaintext output)` -- **plaintext_trusted_only** -- Final output recovered on the trusted side by Y_infer = Y_infer_tilde @ N_y^T.

### masked_adamw_unsupported

- `AdamW second moments` -- **plaintext_trusted_only** -- Coordinate-wise second moments are not invariant under dense orthogonal mixing; AdamW under dense masks is unsupported.

## 6. Honesty phrases (verbatim)

- Base model W is public but transformed to preserve hidden-state masks; the trusted side never exports plaintext A or B.
- User input and token ids are trusted-side only and never leave the user device in plaintext.
- GPU sees masked hidden states, masked LoRA adapters, masked LoRA gradients, and masked momentum buffers.
- GPU does not see plaintext A/B, plaintext gradients, masks, or plaintext optimizer state.
- AdamW under dense masks is unsupported; the module raises DenseMaskedAdamWUnsupported rather than approximating.
- Masked-gradient LoRA provides algebraic equivalence for SGD/Momentum under orthogonal masks and proxy-evaluated leakage mitigation; it does not provide formal, cryptographic, or semantic security.

## 7. Limitations

- Lifecycle classification is descriptive; it is not a formal information-flow proof.
- Base model W is public; we rely on a trusted-side boundary transformation to preserve hidden-state masks. A real deployment would have to verify the boundary transformation matches the trusted side's mask choice.
- AdamW under dense masks is unsupported and is gated by DenseMaskedAdamWUnsupported in the ops module.
- Raw tensors, masks, adapters, gradients, and optimiser states are NEVER exported.

## 8. Paper-safe wording

> masked-gradient LoRA provides algebraic equivalence for SGD/Momentum under orthogonal masks and proxy-evaluated leakage mitigation; it does not provide formal, cryptographic, or semantic security.

`formal_security_claim`: `False`

