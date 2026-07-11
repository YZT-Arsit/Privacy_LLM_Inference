# O1 — preconditioner visibility

The O1-B update needs `P P^T` (left) and `Q^T Q` (right) — i.e. the mask **Gram**.
Question: does the untrusted GPU get to see that Gram?

| profile | preconditioner | who applies it | Gram visible to GPU? | +trusted crossings |
|---|---|---|---|---|
| **O1-A** | identity (orthogonal masks) | GPU | **No** (there is none) | 0 (2 inv/step total) |
| **O1-B** | `P P^T`, `Q^T Q` (mask Gram) | GPU | **Yes** — unless a custom backward hides it | 0 (2 inv/step total) |
| **O1-C** | applied inside the trusted optimizer | TDX | **No** | +1 (3 inv/step total) |

## Why O1-B's exposure matters
The whole premise of a non-orthogonal feature mask is to deny the attacker the
mask **Gram** (the self-Gram alignment attack recovers an *orthogonal* mixer from
the weight Gram). But the O1-B update multiplies the masked gradient by `P P^T` /
`Q^T Q` **on the GPU** — handing the attacker exactly that Gram, in closed form,
which is *strictly more* than what the self-Gram attack has to work to extract.

Where the exposure actually lands (given the RMSNorm constraint):
- **q/k/v/gate/up LoRA**: input mask = residual mask = **orthogonal** (forced by
  RMSNorm) ⇒ `N_in^T N_in = I` ⇒ left preconditioner is trivial, **no feature Gram
  exposed** from the left. Only `U^T U` (rank Gram) appears.
- **o_proj / down_proj LoRA**: input mask is a non-residual basis (`S` / island
  perm). If `S` is non-orthogonal, `S^T S` is exposed on the GPU.
- **`U^T U` (rank Gram)** appears for every target. `U` is a small `r×r` adapter-only
  mask; exposing its Gram is a lesser leak than the feature Gram but is still nonzero.

## Candidate mitigations (open, not yet evaluated)
1. **Custom backward** that folds `P P^T` / `Q^T Q` into the gradient computation
   without ever materialising the Gram as a GPU-resident tensor. Whether an attacker
   can still infer the Gram from `grad_t` before/after the step is unresolved —
   likely yes for a static-package attacker, so this is NOT assumed to work.
2. **Trusted-side application (O1-C)** — send packed masked grads to TDX, apply the
   preconditioner/optimizer there. Removes GPU exposure at the cost of +1 crossing.
3. Restrict non-orthogonality to sub-bases whose Gram is already implied by the
   exposed attention scores (attention `R`), so the optimizer exposes nothing new —
   only meaningful for the attention path, not the residual weight chain.
