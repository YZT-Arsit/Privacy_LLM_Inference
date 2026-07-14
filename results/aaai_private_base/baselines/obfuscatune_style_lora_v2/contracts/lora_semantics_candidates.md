# Candidate LoRA training semantics

## Common convention

Use row-major mathematical weights. For batch/token count `N`, input dimension
`d_i`, output dimension `d_o`, and LoRA rank `r`:

- `X in R^(N x d_i)`
- `W in R^(d_i x d_o)`
- `A in R^(d_i x r)`, `B in R^(r x d_o)`
- scale `s = alpha / r`
- `M = W + s A B`, `Y = X M`
- orthogonal `R_i in R^(d_i x d_i)`, `R_o in R^(d_o x d_o)`.

An input-side projection uses `X* = X R_i`, `W* = R_i^T W` and returns
plaintext `Y`. An output-side projection uses `W* = W R_o` and returns
`Y* = Y R_o`, which the trusted runtime restores by `Y = Y* R_o^T`.

Let `G = dL/dY`. Frobenius gradients in canonical coordinates are
`dX = G M^T`, `dA = s X^T G B^T`, and `dB = s A^T X^T G`.

## Candidate 1 - plaintext-coordinate factors outside

The external factors are canonical `A,B`. For an input-side layer, the GPU has
`X*`, not `X`; evaluating `X* A B` is incorrect. Exact forward therefore
requires either sending plaintext `X` outside or performing `X A` in the trusted
runtime. If plaintext `X` is sent, the usual canonical gradients apply outside.
If `X A` is trusted, A must cross into or be replicated in the trusted runtime
and `dA` requires trusted computation.

For an output-side layer the GPU can form plaintext `X A B`, but must obtain
`(X A B) R_o` from the trusted runtime or receive `R_o`, which reveals the
transform. Backward has the analogous conversion problem.

- Boundary: plaintext X or LoRA intermediates/canonical factors cross.
- GPU view: plaintext private activation is exposed in the simplest variant.
- Optimizer/checkpoint: canonical state and plaintext A/B may remain outside.
- Inference: repeated trusted conversion or plaintext exposure.
- Paper fit: factors are outside, but the simplest executable form defeats the
  transformed-data objective. **Rejected as primary.**

## Candidate 2 - transformed-coordinate factors outside

### Input-side layer

Store `A* = R_i^T A`, `B* = B`. Then
`Y = X* (W* + s A* B*) = X (W + s A B)`.
With external upstream `G`:

- `dA* = s X*^T G B*^T = R_i^T dA`
- `dB* = s A*^T X*^T G = dB`
- `dX* = G (W* + s A*B*)^T = dX R_i`.

The trusted runtime recovers canonical `dX = dX* R_i^T` only when needed.

### Output-side layer

Store `A* = A`, `B* = B R_o`, and `M* = M R_o`. Then
`Y* = X (W* + s A*B*) = Y R_o`.
For canonical `G`, the trusted runtime sends `G* = G R_o` to the GPU:

- `dA* = s X^T G* B*^T = dA`
- `dB* = s A*^T X^T G* = dB R_o`
- `dX = G* M*^T = G M^T`.

- Boundary: transformed activations/gradients plus plaintext nonlinear inputs
  already exposed by the paper; no canonical factor or transform crosses.
- GPU view: A*/B*, gradients and optimizer state; true Q/K/V and MLP
  intermediates as in the paper.
- Optimizer: stored-coordinate SGD/momentum are orthogonally equivalent;
  Adam/AdamW define a different internally consistent trajectory.
- Checkpoint: transformed factors and moments, bound to exact transform and
  base/config. Inference directly reuses the same transformed domain.
- Paper fit: closest to “LoRA parameters outside TEE” with minimal new trusted
  functionality. **Selected primary adaptation.**

## Candidate 3 - external copies with trusted canonical update

Forward and external backward use Candidate 2. The GPU sends `dA*`,`dB*` to the
trusted runtime. It maps them to `dA,dB`, maintains canonical A/B and optimizer
moments, applies canonical AdamW, re-transforms updated factors, and returns the
new A*/B*.

- Input layer: `dA = R_i dA*`, `dB=dB*`; output layer: `dA=dA*`,
  `dB=dB* R_o^T`.
- Boundary: all factor gradients and updated transformed factors every step.
- GPU view: external copies and transformed gradients.
- Optimizer/checkpoint: authoritative canonical A/B/m/v in trusted runtime.
- Inference: transformed copy may be used, but export needs trusted conversion.
- Paper fit: reproduces canonical AdamW but adds authoritative trusted LoRA and
  optimizer state not specified by the paper and materially approaches G2's
  split trusted optimizer design. **Diagnostic variant only, not primary.**

## Compatibility decision

Candidate 2 is the only option that simultaneously preserves transformed input,
keeps LoRA factors outside as explicitly stated, minimizes trusted additions,
and remains distinguishable from G2. Its non-equivalent AdamW trajectory must
be reported rather than hidden.
