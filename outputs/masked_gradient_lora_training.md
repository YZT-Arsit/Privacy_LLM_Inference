# Stage 7.6 — Masked-Gradient LoRA Training with Rank-Space Mixing

## 1. Experiment Scope

We train a small synthetic LoRA regression task in lockstep between a plaintext reference and a masked-gradient cloud path. The cloud accelerator computes masked forward, masked backward, and masked optimiser updates. The trusted side owns the orthogonal masks ``N_x``, ``N_y``, ``M`` and never exports them. We validate per-step that the masked update recovers exactly to the plaintext update at float64 machine precision.

## 2. Threat Model

Honest-but-curious cloud accelerator. The GPU never receives plaintext LoRA adapters or plaintext LoRA gradients in this experiment. The user side does not require a GPU; the simulated cloud accelerator performs masked forward, backward, and optimizer updates. No formal, cryptographic, or semantic security is claimed.

## 3. Masked LoRA Forward Construction

```
  A_tilde = N_x^T A M
  B_tilde = M^T B N_y
  X_tilde = X N_x
  Y_tilde = X_tilde A_tilde B_tilde
          = X N_x N_x^T A M M^T B N_y
          = X A B N_y
```
Forward recovery (per-step max abs err): `<= 1.03e-14`.

## 4. Masked Gradient Derivation

With L = MSE(X A B, target) we have
```
  grad_A = X^T (dL/dY) B^T
  grad_B = (X A)^T (dL/dY)
```
Under the masked forward, ``grad_Y_tilde = 2 (Y_tilde - target_tilde) / n``. The chain rule gives
```
  grad_A_tilde = X_tilde^T grad_Y_tilde B_tilde^T = N_x^T grad_A M
  grad_B_tilde = (X_tilde A_tilde)^T grad_Y_tilde = M^T grad_B N_y
```
We verify both relations per step:
- `max(|grad_A_tilde - N_x^T grad_A M|) <= 4.41e-15`
- `max(|grad_B_tilde - M^T grad_B N_y|) <= 4.30e-15`

## 5. GPU-side Masked SGD

Update rule: ``A_tilde <- A_tilde - lr * grad_A_tilde``, ``B_tilde <- B_tilde - lr * grad_B_tilde``. Because right-multiplication by an orthogonal mask distributes over the linear combination, this is algebraically equivalent to plaintext SGD: Masked SGD is algebraically equivalent under orthogonal masks. Per-step recovery error against plaintext SGD:
- `max(|recovered A_tilde_next - A_plain_next|) <= 2.22e-15`
- `max(|recovered B_tilde_next - B_plain_next|) <= 1.55e-15`

## 6. Momentum SGD

We track masked momentum buffers ``V_A_tilde``, ``V_B_tilde`` with the heavy-ball update ``V <- mu V + grad``, ``param <- param - lr V``. Right-multiplication by orthogonal masks distributes over both updates, so the masked momentum-SGD step recovers exactly to plaintext momentum SGD:
- `max(|recovered A_tilde_mom - A_plain_mom|) <= 2.22e-15`
- `max(|recovered B_tilde_mom - B_plain_mom|) <= 1.55e-15`

## 7. Adam/AdamW Limitation

Dense masked AdamW is not claimed because coordinate-wise second moments are not invariant under dense orthogonal mixing. For a dense orthogonal ``Q``, ``(g Q)_{i, j}^2 != g_{i, j}^2 Q`` in general, so the running second moment ``v <- beta_2 v + (1 - beta_2) g^2`` does not commute with the mask. Stage 7.6's module raises ``DenseMaskedAdamWUnsupported`` when AdamW is requested rather than silently approximating. A future stage could add (i) a trusted-assisted update (recover, AdamW on plain, re-mask), (ii) signed-permutation masks (the only orthogonal class that commutes with coordinate-wise squaring), or (iii) a specialised masked optimiser.
AdamW gate status in this run: `explicitly_raised_as_designed` (DenseMaskedAdamWUnsupported).

## 8. Rank Padding and Rank-Space Mixing

Strategy: `paired_cancellation`; true_rank = `2`; padded_rank = `6`; dummy columns added to A = `4`; dummy rows added to B = `4`. The cancellation block is ``A_pad = [A_real, R, -R]``, ``B_pad = vstack(B_real, S, S)`` so that ``A_pad B_pad = A_real B_real`` (initial dummy contribution norm = `6.26e-16`). The rank-space orthogonal mixer ``M`` is then applied over the padded rank so the accelerator-visible inner dimension is `padded_rank`, while the true rank `true_rank` is hidden from any shape inspection. Per step the dummy contribution norm is verified to remain at machine zero.

## 9. Correctness Results

| step | fwd_err | loss_err | grad_A rel | grad_B rel | sgd_A rec | sgd_B rec | mom_A rec | mom_B rec | dummy norm |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.44e-15 | 5.55e-16 | 3.66e-15 | 3.52e-15 | 2.00e-15 | 1.44e-15 | 2.00e-15 | 1.44e-15 | 1.04e-01 |
| 1 | 1.03e-14 | 4.44e-16 | 4.41e-15 | 4.30e-15 | 2.00e-15 | 1.33e-15 | 2.00e-15 | 1.33e-15 | 1.94e-01 |
| 2 | 7.90e-15 | 0.00e+00 | 3.90e-15 | 3.47e-15 | 2.00e-15 | 1.33e-15 | 2.00e-15 | 1.33e-15 | 2.73e-01 |
| 3 | 4.50e-15 | 1.11e-16 | 1.55e-15 | 1.42e-15 | 2.00e-15 | 1.55e-15 | 2.00e-15 | 1.55e-15 | 3.44e-01 |
| 4 | 7.52e-15 | 3.33e-16 | 2.86e-15 | 2.53e-15 | 2.22e-15 | 1.55e-15 | 2.22e-15 | 1.55e-15 | 4.08e-01 |
| 5 | 4.66e-15 | 1.11e-16 | 1.89e-15 | 2.52e-15 | 1.78e-15 | 1.33e-15 | 1.78e-15 | 1.33e-15 | 4.67e-01 |

## 10. Gradient Leakage Proxy

GPU-visible per-call gradient fingerprints are published as short SHA-256 prefixes so cross-step linkability can be audited externally without exposing raw gradients. The companion module `masked_gradient_lora_security_proxy.py` runs a more structured proxy: true-rank inference from ``A_tilde / B_tilde`` spectra, real-vs-dummy subspace separation, and cross-step linkability under fixed vs fresh masks. Raw tensors, masks, and adapters are NEVER exported.

### GPU visibility table

| variable | visible_to_gpu | exposed_form |
|---|---|---|
| `plaintext_A` | False | never_exported |
| `plaintext_B` | False | never_exported |
| `plaintext_grad_A` | False | never_exported |
| `plaintext_grad_B` | False | never_exported |
| `plaintext_optimizer_state` | False | never_exported |
| `N_x / N_y / M` | False | trusted_only |
| `X_tilde` | True | X @ N_x (masked) |
| `A_tilde` | True | N_x^T A_pad M (masked, rank-padded) |
| `B_tilde` | True | M^T B_pad N_y (masked, rank-padded) |
| `grad_A_tilde` | True | N_x^T grad_A M (masked) |
| `grad_B_tilde` | True | M^T grad_B N_y (masked) |

## 11. Limitations

- Dense masked AdamW is not claimed because coordinate-wise second moments are not invariant under dense orthogonal mixing.
- AdamW under dense masks would require trusted-assisted update, signed-permutation masks, or a specialised masked optimiser; none are implemented in Stage 7.6.
- CPU local emulation only; no real TEE / GPU runtime is measured.
- This is an algebraic correctness + proxy leakage stage; no formal cryptographic / semantic / differential-privacy security is claimed.
- Synthetic regression task only; this is not a real Qwen / LLaMA LoRA fine-tuning workload.
- Loss boundary uses MSE on Y_tilde vs target_tilde; orthogonal N_y preserves the L2 loss exactly, but a softmax / cross-entropy loss would require a trusted loss boundary, which is out of scope for Stage 7.6.
- Raw tensors, adapters, gradients, and masks are NEVER exported; outputs contain only summary scalars, shapes, and short fingerprints.

## 12. Next Stage Plan

Future work: (i) integrate signed-permutation masks for AdamW exactness; (ii) explore a trusted-assisted AdamW boundary where the cloud accelerator returns the masked gradient and the trusted side runs the per-coordinate second-moment update on a small slice; (iii) extend the construction to softmax / cross-entropy losses via a trusted loss boundary; (iv) integrate with the Stage 7.5c deployable runtime API so a real serving runtime can route masked gradients without seeing the trusted-side recovery.

`formal_security_claim`: `False`

## Honesty phrases (verbatim)

- The GPU never receives plaintext LoRA adapters or plaintext LoRA gradients in this experiment.
- Masked SGD is algebraically equivalent under orthogonal masks.
- Dense masked AdamW is not claimed because coordinate-wise second moments are not invariant under dense orthogonal mixing.
- This is a CPU-only algebraic and proxy-leakage experiment, not a real TEE/GPU training benchmark.
- The user side does not require a GPU; the simulated cloud accelerator performs masked forward, backward, and optimizer updates.
- No formal, cryptographic, or semantic security is claimed.

