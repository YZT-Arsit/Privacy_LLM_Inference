# Variant F — Selective Token-Mixing Shield (STMS): summary

## Threat model (STRICTLY WEAKER THAN V4 — read first)
STMS shields the observable boundary artifact `U = A (H N)` against an observer
who does **not** know `A`/shield rows and does **not** see the post-recovery state
`H N`. It is a **boundary shield**, not full-transformer left mixing: `A` never
enters attention/RMSNorm/RoPE/KV, and `H N = A^{-1} U` is recovered before any
real compute. **If the adversary sees `H N`** (the V4 / compute-offload model),
**STMS adds nothing** — V4 and the O(d^3) weight-mask recovery apply to `H N`
unchanged. `protects_compute_visible_HN = False`.

## What observation is protected
Only the transmitted/logged snapshot `U`. Not the compute state, not decode
single-token states, not KV.

## Correctness (roundtrip + prefill integration)
- roundtrip max_abs over all cases: **1.14e-13** (fp64 exact).
- prefill: recovered state == `H N` to 8.53e-14;
  shield rows in recovered state: **0** (dropped before KV).

## Forward-matching oracle (candidate id, 16 same-length candidates)
| case | top1 | sensitive-token top1 |
|---|---|---|
| baseline (H N) | True | 1.00 |
| STMS orthogonal A | True | 1.00 |
| STMS non-orthogonal A | False | 0.00 |
| STMS block-local A | True | 1.00 |

- **orthogonal A does NOT stop the oracle**: `U U^T = A (H H^T) A^T` is an
  orthogonal *similarity*, so the singular-value spectrum of `H` is invariant and
  a spectrum-matching oracle still ranks candidates.
- non-orthogonal A distorts the spectrum (helps), but a free-`A` oracle overfits,
  so security reduces to BSS/ICA hardness (below).

## BSS / ICA recovery risk (median AND p95 |cosine|)
- orthogonal-A ICA: median 0.205,
  **p95 0.287** (residual risk if p95 high).
- shield / block / condition sweeps: see bss_ica.md.

## Efficiency
- token-mix adds an `[n,n] · [n,d]` matmul (`mix`) + an unmix (`A^T` for
  orthogonal, `solve` for non-orthogonal); both scale with sequence length `n`.
- see efficiency.md; non-orthogonal `solve` and shield rows add cost.

## Limitations / honesty
- **Not production-ready** (no full integration beyond bit-exact prefill roundtrip).
- **Does not protect decode single-token states** (a length-1 boundary has no token
  dimension to mix).
- **Does not cross attention/KV/RMSNorm** (by design) — so it cannot change what the
  compute engine sees.
- **No cryptographic privacy claim.**
- orthogonal A: forward oracle still succeeds (spectrum) → reported as failure of
  that configuration.
- shield rows cost throughput (extra rows through the boundary transform).
- if only p95 |cosine| stays high, that is residual BSS risk — reported, not hidden.
