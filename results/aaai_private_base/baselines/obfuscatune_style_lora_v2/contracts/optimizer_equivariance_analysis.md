# Optimizer equivariance analysis

Let a stored parameter be `theta* = T theta` with orthogonal linear map `T`, and
let `g* = T g`. Left and right matrix multiplication are orthogonal maps under
the Frobenius inner product.

| Optimizer | Orthogonal-coordinate equivalence | Reason |
|---|---|---|
| SGD | Exact in real arithmetic | `T(theta - eta g) = T theta - eta T g`. |
| SGD + momentum | Exact with transformed zero/equivalent initial momentum | The first-moment recurrence is linear and scalar hyperparameters commute with T. |
| Adam | Not rotation-equivariant | Elementwise `g^2`, square root and division do not commute with dense T. |
| AdamW | Adaptive part is not rotation-equivariant | Decoupled scalar decay commutes with T, but Adam's elementwise second moment does not. |

Coupled L2 decay inside SGD/momentum also commutes with an orthogonal T. For
Adam, adding `lambda theta` to the gradient does not repair the non-equivariant
second-moment operation. For AdamW, zero-gradient decay alone is exactly
equivariant: `T[(1-eta lambda)theta]=(1-eta lambda)T theta`.

## Concrete AdamW counterexample

Use a 45-degree rotation, `theta=[1,2]`, `g=[0.3,-0.7]`, LR .1,
betas (.9,.999), epsilon 1e-8 and weight decay .01, starting with zero moments.
Canonical AdamW gives `[0.8990000033, 2.0979999986]`. Updating in the rotated
coordinates and mapping back gives `[0.9989999985, 2.1394213527]`; maximum
absolute difference is `0.0999999952`. This is not a tolerance issue.

## Finite precision

Even SGD equivalence is algebraic rather than bitwise: forming `R^T A`, matrix
multiplication order, BF16 casts, and reductions introduce rounding differences.
FP32 tests use strict numerical tolerances; BF16 tests use declared tolerances
and must never be described as exact trajectories.

## Selected baseline statement

Candidate 2 uses a different but internally consistent AdamW optimizer over the
stored transformed coordinates. It does **not** reproduce plaintext AdamW
semantics. Candidate 3 would reproduce canonical AdamW by trusted updates, but
is not primary because the paper places LoRA outside and does not prescribe
trusted authoritative optimizer state.
