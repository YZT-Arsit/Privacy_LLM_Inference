# Privacy LLM Obfuscation — Nonlinear Islands (Stage 5.2)

## Experiment scope

Stage 5.2 implements and validates operator-compatible nonlinear islands. Each island matches a nonlinear operator with the mask family that commutes with it, folds mask transitions into adjacent Linear weights offline, and verifies the masked forward equals the plaintext forward times the residual output mask. The goal is to keep the nonlinear core in the GPU-visible masked domain without adding extra online matmuls.

## Operator-Compatible Mask Families

| operator | mask family | preserved invariants |
|---|---|---|
| Linear / Attention / KV cache | dense_invertible | — |
| RMSNorm core | orthogonal | row L2 norm (rms preserved) |
| LayerNorm core | mean_preserving_orthogonal | row mean + centered L2 norm |
| GELU / ReLU / SiLU activation | permutation | coordinate-value multiset |
| SwiGLU activation | paired_permutation | paired (up,gate) multiset |

## Norm-Compatible Island Results

| island | hidden | ortho err | mean preservation err | max abs err | allclose |
|---|---|---|---|---|---|
| rmsnorm_orthogonal_affine_fold | 64 | 4.172e-07 | — | 1.526e-05 | true |
| layernorm_mean_preserving_affine_fold | 64 | 2.980e-07 | 2.384e-07 | 1.717e-05 | true |
| rmsnorm_orthogonal_affine_fold | 128 | 5.364e-07 | — | 2.289e-05 | true |
| layernorm_mean_preserving_affine_fold | 128 | 4.172e-07 | 2.384e-07 | 2.480e-05 | true |

## Affine Folding Results

Both RMSNorm and LayerNorm affine parameters are folded into the adjacent Linear layer *offline* — the GPU never sees `gamma` or `beta` as separate tensors. Folding rules:

```text
LayerNorm:  W_folded = diag(gamma) @ W       b_folded = beta @ W + b
RMSNorm  :  W_folded = diag(gamma) @ W       b_folded = b
```

After folding the masked weight becomes `W_tilde = N_in^T W_folded N_out` (orthogonal / mean-preserving orthogonal ``N_in``).

## Activation Permutation Island Results

| activation | hidden | max abs err | allclose |
|---|---|---|---|
| gelu | 64 | 0 | true |
| relu | 64 | 0 | true |
| silu | 64 | 0 | true |
| gelu | 128 | 0 | true |
| relu | 128 | 0 | true |
| silu | 128 | 0 | true |

## SwiGLU Paired-Permutation Island Results

| activation | hidden | max abs err | allclose |
|---|---|---|---|
| swiglu | 64 | 0 | true |
| swiglu | 128 | 0 | true |

## Full MLP Island Results

| mlp_type | hidden | intermediate | use_pad | max abs err | online extra matmul | allclose |
|---|---|---|---|---|---|---|
| gelu_mlp | 64 | 256 | true | 8.345e-07 | 0 | true |
| relu_mlp | 64 | 256 | true | 7.749e-07 | 0 | true |
| silu_mlp | 64 | 256 | true | 7.153e-07 | 0 | true |
| swiglu_mlp | 64 | 256 | true | 4.936e-07 | 0 | true |
| gelu_mlp | 64 | 256 | false | 6.855e-07 | 0 | true |
| relu_mlp | 64 | 256 | false | 8.345e-07 | 0 | true |
| silu_mlp | 64 | 256 | false | 7.153e-07 | 0 | true |
| swiglu_mlp | 64 | 256 | false | 4.172e-07 | 0 | true |
| gelu_mlp | 128 | 512 | true | 1.132e-06 | 0 | true |
| relu_mlp | 128 | 512 | true | 1.162e-06 | 0 | true |
| silu_mlp | 128 | 512 | true | 1.311e-06 | 0 | true |
| swiglu_mlp | 128 | 512 | true | 7.451e-07 | 0 | true |
| gelu_mlp | 128 | 512 | false | 1.371e-06 | 0 | true |
| relu_mlp | 128 | 512 | false | 1.252e-06 | 0 | true |
| silu_mlp | 128 | 512 | false | 1.192e-06 | 0 | true |
| swiglu_mlp | 128 | 512 | false | 5.066e-07 | 0 | true |

## Pad Placement Rule

Pad is allowed at Linear boundaries only and compensated through the linear compensation term C = T W N_out. Pad is never pushed through an activation; the activation input is Z P (no pad).

- Pad enters and exits only through Linear boundaries: ``X_tilde = (X - T) N_in`` at island entry, ``C = T W_perm`` adds the standard linear compensation, and the activation operates on ``Z P`` (no pad).
- Pushing pad through an activation is invalid: ``f((Z - T) P) ≠ f(Z P) - f(T P)`` for any nonlinear ``f``, so simple additive compensation cannot cancel it. The island therefore strips pad at the Linear entry and downstream wrappers may re-introduce a fresh pad at the next Linear boundary.

## Online Cost Interpretation

All mask + permutation transitions are *preprocessing-only* and are folded into the masked weight tensors before the GPU forward starts. Concretely, the offline pipeline computes:

```text
# Norm island (offline):
W_folded = diag(gamma) @ W       (RMSNorm / LayerNorm fold)
b_folded = beta @ W + b          (LayerNorm only)
W_tilde  = N_in^T @ W_folded @ N_out
b_tilde  = b_folded @ N_out

# MLP island (offline):
W1_tilde   = N_in^{-1} @ W1[:, perm]
b1_tilde   = b1[perm]
W2_tilde   = W2[perm, :] @ N_out
b2_tilde   = b2 @ N_out
```

The online masked path executes exactly the same number of matmuls as the plaintext path. `online_extra_matmul_count = 0` across every cell — folded mask transitions add zero online cost.

## Limitations

- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Orthogonal masks preserve norms by design.
- Mean-preserving orthogonal masks preserve mean and centered norm by design.
- Security relies on freshness, dense-mask sandwiching, and pad at Linear boundaries.
- This stage does not prove semantic security.
- This stage does not implement adaptive permutation-recovery attacks beyond proxy experiments.
- This stage does not implement real TEE.

## Next Stage Plan

- **Stage 5.3** — Stronger leakage experiments (adaptive attackers, learned inversion) targeting the compatible mask family boundaries.
- **Stage 6.4** — Qwen / TinyLlama migration. The RMSNorm orthogonal island + SwiGLU paired-permutation island land exactly the two operators Qwen / LLaMA need, on top of the Stage 5.1 RMSNorm primitive.
