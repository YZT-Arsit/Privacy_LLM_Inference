# Notation

This appendix fixes the symbols used throughout the paper draft. We use bold-free upright math notation in the Markdown source so that the conversion to LaTeX is mechanical.

## Tensors and dimensions

- `X` — token-by-feature activation matrix entering a Linear layer; shape `[T, d_in]` (or `[B, T, d_in]` when batched).
- `W` — the proprietary base-model weight of a Linear layer; shape `[d_in, d_out]`. Private: only the masked form `W_tilde` is exposed to the GPU.
- `b` — the public bias of a Linear layer; shape `[d_out]` or broadcasted.
- `Y` — the plain reference output of a Linear layer, `Y = X W + b`.
- `T` — a trusted-side translation tensor used to center the masked input.
- `Z` — generic intermediate activation inside a nonlinear island.
- `phi(.)` — a pointwise activation (GELU, ReLU, SiLU, etc.).
- `RMSCore(.)`, `LNCore(.)` — the orthogonally-invariant cores of RMSNorm and LayerNorm, separated from the trainable scale/shift parameters.
- `Q`, `K`, `V` — attention queries, keys, values per head.
- `A`, `B` — LoRA factors; `A` of shape `[d_in, r]`, `B` of shape `[r, d_out]`.
- `r` — the true LoRA rank.
- `r_pad` — the GPU-visible padded LoRA rank, with `r_pad >= r`.
- `alpha` — the LoRA scaling factor.

## Mask families

- `N_in`, `N_out`, `N_Q`, `N_K`, `N_V` — boundary mask matrices applied at Linear boundaries; in this paper draft they are right-multiplicative and (depending on the operator) either invertible, orthogonal, mean-preserving orthogonal, paired-permutation, or paired-cancellation in structure.
- `P` — a channel permutation matrix applied inside a nonlinear island.
- `U` — a trusted-side inner LoRA mask satisfying `U U^{-1} = I`.
- `M` — a generic GPU-visible mask family that the design ensures is operator-compatible with the surrounding nonlinear operator.

## Decorated symbols

- `X_tilde`, `Y_tilde`, `W_tilde`, `A_tilde`, `B_tilde`, `Q_tilde`, `K_tilde`, `V_tilde`, `Z_tilde` — the GPU-visible masked counterparts of the corresponding plain tensors.
- `A_pad`, `B_pad` — the rank-padded LoRA factors actually shipped to the GPU, satisfying `A_pad B_pad = A B` (exactly, or up to a tracked trusted-side correction for `noise_injected_cancellation_dummy`).
- `C_T`, `C1` — trusted-side compensation tensors absorbed before / after the GPU call.

## Equality conventions

- `=` denotes equality of plain tensors at the trusted side.
- `==_tested` denotes empirical `allclose` equality in our reported configurations; we never use it to mean formal proof of equality across arbitrary inputs.
- `=:=` is used in proof sketches when we want to flag an identity that follows algebraically from the construction (e.g., `Q_tilde K_tilde^T =:= Q K^T` when `N_Q N_K^T = I`).

## Parties

- **TEE-like controller (trusted side):** holds the user prompt, the user's LoRA adapter `(A, B)`, the optimizer state, the loss closure, the sampler, and the mask sampler. In this prototype, the controller is emulated as a local trusted runtime.
- **GPU (untrusted side):** performs masked matmuls, masked nonlinearities (when an operator-compatible mask family is used), masked attention, and masked LoRA forward/backward matmuls. It also observes shapes, sequence length, and the cost-model timing proxy.

## Things this draft never writes

- "provably", "guaranteed", "cryptographically secure", "semantically secure", "TEE-level secure", "prevents all leakage", "hides padded rank", "production wall-time on TEE", "full Qwen / TinyLlama / LLaMA fine-tuning".

If any sentence in the body would require one of these phrases to be true, that sentence is reframed as "in our tested configurations" or moved to Limitations.
