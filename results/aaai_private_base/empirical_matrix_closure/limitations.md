# Limitations — empirical matrix closure

## The binding constraint: prototype transport wall-time
`transport_profile = mac_ferried_authenticated_prototype`. Measured on real hardware this session:
**~184 s / training step** (25.5 MB logits + 3.48 MB corrected grads each way over cross-cloud SSH,
Mac-relayed because H800↔TDX inbound ports are firewalled). In-enclave compute is ~0.05 s/step.
Consequences, stated up front so nothing is over-claimed:

| run | per-run wall | 3-seed cost |
|---|---|---|
| 1-step gate | ~7 min (incl. fresh attestation + verifier) | ~21 min |
| 10-step gate | ~35 min | ~105 min |
| **50-step trajectory** | **~2.5 h** | **~7.5 h per profile → ~23 h for L1/L2/L10** |

A full real 50-step × 3-seed × {SGD, momentum, AdamW} matrix plus converged 3-seed utility is
multi-day of ferry/GPU time — not completable in one session. `direct_h800_tdx_transport` (which
would remove the Mac relay) is registered but explicitly out of scope for closure.

## What is therefore real vs. missing (see `missing_runs_enumeration.json`)
- **Real this session:** PHASE-A provenance closure; **BF16 L10 gate 3-seed 1/10-step** on real
  H800+TDX; **L11 momentum 3-seed 1/10-step** on real H800+TDX; deployment provenance; timing/comm.
- **Missing (enumerated, not substituted):** SGD/momentum **50-step** 3-seed; **L12 AdamW on real
  hardware**; converged **GSM8K/SST-2 utility**; **real** rank-mask confirmation (Phase J).

## Specific caveats
- **L12 AdamW hardware** needs a *different* TDX protocol than SGD/momentum: the enclave must hold
  the AdamW moments (m, v) for the trusted factors (A of q/k/v/gate/up **and B of q/k** — the RoPE
  rotation is non-monomial) across steps and return *updated factors*, not a corrected gradient.
  That protocol is not implemented, so L12 is **fp64-validated only — NOT claimed hardware-validated.**
- **BF16 correction (corrected earlier error):** BF16 shares FP32's 8-bit exponent, so the tiny
  gate/up components (min_abs ~6e-17) do **not** underflow; the real gate shows
  `corrected_grad_zeroed_fraction ≈ 0`. Equivalence degrades to top1 ~0.976 / KL ~8e-3 vs the fp64
  plaintext reference — BF16 mantissa precision, not instability. Exactness stays an fp64 property.
- **Analytic factored gradients** (fp64 matrix) remain valid *algebraic* evidence of exactness but
  are **not** a substitute for the registered real 50-step hardware trajectories, which are listed
  as missing.
- **Utility:** the prior 10-step held-out CE (L0 45.26 vs L10 48.45 ppl) is a **deployment smoke,
  not** task improvement, and is not used as a utility result. Converged utility is missing.

## Stop conditions (this session): none triggered
BF16 L10 stable/finite across completed seeds; no gate/up overflow/underflow; no correction secret
or γ on H800 (`untrusted_* = 0`); no plaintext load on the worker path; package + worktree hashes
stable; no silent fallback. Momentum-buffer / AdamW-state hardware match is pending the respective
real runs (buffer match is fp64-proven; AdamW hardware is missing).
