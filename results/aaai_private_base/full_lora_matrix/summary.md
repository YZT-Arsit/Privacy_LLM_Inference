# Full LoRA validation matrix — summary

Profile decision (accepted, unchanged during the matrix): **MAIN = L10 O1-C hybrid** —
o_proj/down_proj updated on GPU (orthogonal transforms); q/k/v/gate/up corrected **inside real
Intel TDX**. Code baseline: HEAD `a95b59f`, clean worktree (`e3b0c442…`); nothing committed.

## 1. Code baseline (Section 0)
Three external commits (`36b2d6c`, `d4f44dd`, `a95b59f`) = this agent's own Gate-0/0.5 artifacts
committed under the user's identity, empty messages; agent issued **0** git commits. No experiment
semantics changed → no prereqs rerun. `code_baseline/` holds the audit + HEAD/worktree hashes bound
into every run manifest.

## 2. Registry (Section 1)
L0–L9 preserved verbatim; **L10/L11/L12 added** (O1-C SGD/momentum/trusted-AdamW). L2/L4 (O1-A
uniform GPU) demoted to efficiency-approximation with `gate05_role` annotations. Amendment reason:
"Gate 0.5 discovered gamma-fold-induced non-orthogonal LoRA input metrics." `registry_amendment.json`.

## 3. Exact equivalence (fp64) — Sections 6/7/8
`statistical_analysis.json` (3 optimizers × 3 seeds × 24 layers × steps{1,10,50}, fp64):

| | O1-C A-update (worst) | O1-A A-update |
|---|---|---|
| SGD | **4.98e-16 (exact)** | o/down exact; qkv drift 1.10; gate/up **diverges (NaN)** |
| Momentum | **6.09e-16 (exact)** incl. buffer | qkv drift 0.675; gate/up diverges |
| AdamW | **3.80e-13 (exact)** | qkv drift 0.920; gate/up diverges |

- **O1-C exact for all optimizers, all target groups, all seeds.** Momentum buffer and AdamW state
  matched (or exact in the TDX plaintext basis).
- **O1-A is NOT parameter-equivalent** (exact only on o/down). New finding: on gate/up it is
  **numerically unstable** — diverges to NaN in 66/144 fp64 cases by step 50 (ill-conditioned
  γ-fold, cond ~8e5), not merely "functionally aligned."
- **AdamW rigor (L12):** the 2nd moment is elementwise, so exactness needs a *monomial* transform
  (stronger than orthogonal). ⇒ q/k **B-factor** (2D RoPE rotation, non-monomial) ALSO runs in TDX;
  A∈{q,k,v,gate,up} + B∈{q,k} in TDX; o/down fully GPU-exact AdamW. Implemented + fp64-validated.

## 4. Real BF16+TDX L10 gate (Section 4) — PASS
Real H800 + real Intel TDX, fresh attestation bound to `optimizer_profile=o1c_hybrid` (**verified,
SUCCESS, debug=false**). Seed 1234, fp32.
- **1-step:** top1 **1.0**, KL 7.9e-9. **10-step:** CE **0.4542→0.2859 monotonic**, top1 1.0, KL 7.1e-9.
- Every step: **48** o/down A-updates on GPU, **120** q/k/v/gate/up A-grads corrected **in-enclave**,
  0 missing, HMAC ok, finite.
- Counters: `untrusted_gamma/correction_matrix/plaintext_gradient_materializations = 0`,
  `silent_fallbacks = 0`, 3 logical trusted invocations/step.
- In-enclave numerics: correction compute 48 ms; gate/up (mlp_in) corrected components reach
  min_abs 5.9e-17 (14 orders below max) → **would vanish in pure bf16** ⇒ fp32 master weights
  required (documented; full bf16-forward run preregistered).

## 5. Invocation / communication / performance (Section 5/13)
`communication/`, `performance/`: 3 logical invocations/step; 25.5 MB logits + 3.48 MB correction
each way (correction = **12%** of GPU→TDX bytes); in-enclave compute 0.05–0.1 s vs ~183 s step wall
(cross-cloud ferry). `transport_profile = mac_ferried_authenticated_prototype` — **not** a deployment
latency claim; `direct_h800_tdx_transport` registered.

## 6. Rank-mask ablation (Section 11)
`ablations/`: L7 (no mask), L10 (fixed U), refresh every-step/every-10/per-session **all exact**
(~1e-15) WITH optimizer-state transport. Negative control — refresh **without** state transport —
**breaks** (dW err 0.79). Linkability metric reported (secondary; simple correlation).

## 7. Deployment + utility (Sections 3B/9)
`deployment/L10_deployment_provenance.json`: **deployment_ok**, all Section-9 assertions true,
package-native, `plaintext_adapter/base_materializations = 0`. Held-out CE (4-prompt preregistered
subset): L0 45.26 ppl vs L10-10step 48.45 ppl — the 10-step single-batch run is a **smoke**
(slight overfit), NOT a utility claim. Utility of L10 **equals L1 by construction** (exactness);
converged exact-match/accuracy on official splits is **preregistered** (`utility/utility_summary.json`).

## Stop conditions (Section 14): none triggered
L10 parameter-equivalent within tolerance (fp64 exact); no overflow/instability in the fp32 gate;
no correction-secret/γ on H800; no plaintext checkpoint/adapter loaded on the worker path; no
dataset/order mismatch; package + worktree hashes stable; no silent fallback.
