# Scheme A — Nonlinear Backward Primitive Audit (SiLU / SwiGLU)

Experiment/audit module (`src/pllo/experiments/nonlinear_backward_primitive_audit.py`); no production path modified. Scoped to the minimal SiLU/SwiGLU island that Qwen/LLaMA use — **not** full Qwen training. Uses the real `pllo.ops.amulet_right_mask_islands` forward primitive; all math float64.

## Core finding

The Amulet/Kronecker right-mask island gives a genuine **GPU-side forward** primitive for SiLU and SwiGLU under a *dense* mask (`U n → phi(U) n` via a Kronecker lift), exact to ~1e-16. But the **A1 backward is blocked**: the nonlinear backward needs elementwise (Hadamard) products of tensors carried under *independent* masks (`GY·M_out ⊙ SiLU'(X)·N_in`), and the Hadamard product does not commute with dense masks. Even granting a perfect masked derivative (obtained via the same lift, err `1.92e-16`), the Hadamard step has rel error `1.56` — `blocked_by_hadamard`. The Amulet lift cannot rescue it because the lift requires both Hadamard operands to share the *same* mask and unit-selection (that is A0, which re-leaks the exact cross-Gram), not an independent A1 mask.

## Answers

1. **GPU-side SiLU forward implemented?** **Yes** — Amulet dense island, rel `5.21e-16`, runtime `0.08 ms`.
2. **GPU-side SiLU A1 backward implemented?** **No** — `blocked_by_hadamard` (Hadamard attempt rel `1.56`).
3. **GPU-side SwiGLU forward implemented?** **Yes** — Amulet dense island, rel `4.76e-16`.
4. **GPU-side SwiGLU A1 backward implemented?** **No** — `blocked_by_hadamard` (dU rel `1.49`, dG rel `0.85`).
5. **Does any variant avoid exact cross-Gram?** Trusted-shortcut A1 (TEE, not GPU) and permutation-A1 both break the exact cross-Gram (A0 dual = 1.000 → exact leak; A1-dense = -0.008; permutation-A1 = +0.120). **But** the only GPU-side backward (permutation) leaks the activation Gram (corr 1.000) — it does not deliver A1 dense-mask privacy. So: **no dense GPU-side variant both trains and avoids the cross-Gram**.
6. **Fast enough in small dims?** The forward island runs in a few ms at m=4,d=8, but the Kronecker lift blows the working tensors up by the factor `k` per side (see efficiency.csv), so it scales poorly to real dims.
7. **Tiny-transformer nonlinear training without trusted shortcut?** **No** — `blocked_by_hadamard` (masked SwiGLU forward works, A1 backward blocked). The permutation alternative trains but needs permutation masks throughout (incompatible with the dense masks linear/LoRA A1 needs, and leaks the Gram).
8. **Is full Scheme A still blocked?** **Yes.**

## SiLU forward/backward

| variant | fwd rel | bwd rel | gpu-side | TEE plaintext | fwd status | bwd status | act-Gram corr |
|---|---|---|---|---|---|---|---|
| trusted_shortcut | 3.01e-16 | 1.93e-16 | False | True | trusted_shortcut_only | trusted_shortcut_only | — |
| amulet_dense | 5.21e-16 | 1.56 | True | False | forward_only | **blocked_by_hadamard** | — |
| permutation | 0.00e+00 | 0.00e+00 | True | False | backward_passed | backward_passed | **1.000 (leaks)** |

## SwiGLU forward/backward

| variant | fwd rel | dG rel | dU rel | gpu-side | bwd status | act-Gram corr |
|---|---|---|---|---|---|---|
| trusted_shortcut | 4.54e-16 | 1.69e-16 | 2.60e-16 | False | trusted_shortcut_only | — |
| amulet_dense | 4.76e-16 | 0.85 | 1.49 | True | **blocked_by_hadamard** | — |
| permutation | 0.00e+00 | 0.00e+00 | 0.00e+00 | True | backward_passed | **1.000 (leaks)** |

## Cross-Gram after nonlinear (200 trials)

| variant | has bwd | gpu-side | cross-Gram corr | exact leak | act-Gram corr |
|---|---|---|---|---|---|
| trusted_shortcut A0 dual | True | False | +1.0000 | **True** | — |
| trusted_shortcut A1 dense | True | False | -0.0081 | False | — |
| permutation A1 | True | True | +0.1203 | False | **1.000** |
| amulet_dense | False | — | — | not_applicable | — |

## Allowed claims

- "The Amulet right-mask island is a genuine GPU-side **forward** primitive for SiLU and SwiGLU under dense masks (exact to fp)."
- "The A1 (independent-dense-mask) **backward** for SiLU/SwiGLU is blocked: it requires Hadamard products under independent dense masks, which do not commute (`blocked_by_hadamard`), and the Amulet lift cannot supply it without collapsing to shared-mask A0."
- "A working GPU-side nonlinear backward exists only under permutation masks, which leak the activation Gram — so it does not achieve A1 dense-mask privacy."
- "trusted_shortcut is exact but recovers plaintext inside the TEE; it is the correctness upper bound, not a GPU-side primitive."
- "Full Scheme A masked-domain training remains blocked by the nonlinear backward primitive."

## Disallowed claims

- ❌ "GPU-side SiLU/SwiGLU A1 backward is supported." (blocked_by_hadamard, measured)
- ❌ "The Amulet island enables masked-domain nonlinear training." (forward only)
- ❌ "Permutation masks solve Scheme A nonlinear privacy." (activation Gram leaks)
- ❌ "Standard autograd gives A1 nonlinear backward." (autograd = A0 dual, which leaks the exact cross-Gram; not used here)
- ❌ "Full Scheme A is unblocked / Qwen nonlinear training works." (not run; blocked)
