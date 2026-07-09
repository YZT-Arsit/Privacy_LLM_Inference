# Scheme A — Masked-Domain LoRA Backward Audit

Experiment/audit module (`src/pllo/experiments/masked_lora_backward_audit.py`); no production path modified. All core math float64. Compares **A0** (naive dual backward) vs **A1** (independent backward mask) on a synthetic linear LoRA layer + tiny transformer.

## Answers to the required questions

1. **Is A0 correct?** Yes. Forward `2.00e-15`; recovered grad_X `1.33e-15`, grad_A `2.22e-15`, grad_B `2.50e-15` — all machine precision.
2. **Does A0 leak exact cross-Gram?** Yes. `X_tilde G_tilde_X^T = X G_X^T` to `7.77e-15` (rel `5.84e-16`), correlation **1.0000** (deterministic exact leak, mean 1.0000 over 200 trials at every m).
3. **Does A1 keep gradients correct?** Yes. Recovered grad_X `1.33e-15`, grad_A `1.78e-15`, grad_B `2.50e-15`; GPU-visible input gradient equals `G_X M_in` to `1.55e-15`.
4. **Does A1 remove the exact cross-Gram equality?** Yes. Rel error jumps to `0.942` (not machine precision); correlation mean **-0.0075** with |corr| noise floor 0.240→0.095 as m 8→64. This breaks the *exact plaintext equality only*; it is **not** a proof of zero leakage (residual `X (N_in M_in^T) G_X^T` structure remains).
5. **Does A1 need custom backward?** Yes. Standard autograd reproduces **A0** exactly (grad match `1.78e-15`); A1 uses backward-only masks/weights and requires explicit/custom masked backward. GPU-visibility audit: no plaintext gradients/data and no raw mask / mask-switch matrix are published in either scheme.
6. **Is nonlinear backward implemented?** No. For GELU/SiLU/SwiGLU the masked-domain backward status is `not_implemented` (a pointwise `f` does not commute with a right mask: `f(XN)` recovers to `f(X)` at rel err ~0.7–0.95, so there is no masked forward to differentiate). Only a trusted-side (island / trusted_shortcut) forward path exists.
7. **Is AdamW only trusted-side?** Yes. A0/A1 recover plaintext grads; the optimizer runs trusted-side (`trusted_side`), AdamW final-adapter error `3.11e-15`. Dense masked-domain AdamW is refused: `raised_unsupported`.
8. **Model level reached?** `tiny_transformer_passed (linear-only); nonlinear blocked; no gpt2/qwen run`. Synthetic linear + tiny transformer (linear-only) pass; nonlinear path is blocked; no GPT-2/Qwen run.
9. **Forward/backward transformed-weight alignment risk?** The GPU sees both `W_tilde_fwd` and `W_tilde_bwd`. Singular spectra are shared/leaked (`spectrum_shared=True`) and the two are linkable/alignable (residual `1.30e-15`), but no trivial recovery of W was found (naive recovery rel err `1.394`). Verdict: **no trivial recovery found; singular spectrum is shared/leaked and fwd/bwd are linkable (alignable) but W itself is not recovered without a mask**.
10. **Allowed / disallowed claims:** see below.

## Cross-Gram leakage (multi-trial)

| m | A0 corr (mean) | A1 corr (mean) | A1 |corr| mean | A1 |corr| p95 |
|---|---|---|---|---|
| 8 | 1.0000 | +0.0095 | 0.2398 | 0.5339 |
| 32 | 1.0000 | -0.0104 | 0.1116 | 0.2870 |
| 64 | 1.0000 | -0.0075 | 0.0951 | 0.2195 |

## Optimizer equivalence (A1, trusted-side optimizer)

| optimizer | location | max param err | loss-curve dist | final adapter err |
|---|---|---|---|---|
| sgd | trusted_side | 6.94e-18 | 4.44e-16 | 6.94e-18 |
| momentum_sgd | trusted_side | 5.20e-18 | 2.94e-16 | 1.73e-18 |
| adamw | trusted_side | 3.11e-15 | 4.44e-16 | 3.11e-15 |

Dense masked-domain AdamW: **raised_unsupported**.

## Tiny transformer

- linear-only: **passed** — A0 cross-Gram corr 1.000, A1 0.329 (single small-m draw).
- nonlinear: **blocked_by_nonlinear_backward_primitive** (masked SiLU forward recovery rel err 0.870).
- attention leakage: scores plain-visible = **True** (`Q_tilde K_tilde^T = Q K^T` under a shared right mask). Correctness ≠ privacy.

## Allowed claims

- "Naive dual-mask backward (A0) is correct but leaks the exact activation–gradient cross-Gram (`X_tilde G_tilde_X^T = X G_X^T`, corr 1.0)."
- "Independent backward masks (A1) restore linear/LoRA backward correctness (grad recovery at machine precision) while removing the *exact* plaintext cross-Gram equality in synthetic tests (corr mean ≈ 0, |corr| decaying with m)."
- "A0 is reproducible by standard autograd; A1 requires explicit/custom masked backward."
- "AdamW is supported only trusted-side (on recovered plaintext gradients); dense masked-domain AdamW is refused."
- "Full Scheme A training remains conditional on nonlinear backward primitives and custom-backward integration."

## Disallowed claims

- ❌ "Scheme A fully solved." (nonlinear backward missing; only linear/LoRA proven)
- ❌ "Standard autograd is secure." (autograd = A0 = exact cross-Gram leak)
- ❌ "Nonlinear backward is supported." (`not_implemented`, measured)
- ❌ "Qwen/GPT-2 LoRA training passed." (never run)
- ❌ "No leakage remains." (A1 breaks *exact equality* only; residual structure remains; attention scores + shared weight spectrum still leak)
- ❌ "A1 hides the base weight." (fwd/bwd masked weights share the singular spectrum and are linkable)

_Model status_: **tiny_transformer_passed (linear-only); nonlinear blocked; no gpt2/qwen run**.
