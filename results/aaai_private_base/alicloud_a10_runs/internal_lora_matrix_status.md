# Internal LoRA matrix — empirical closure status

**Status: `INTERNAL_LORA_MATRIX_PARTIAL`** · Nothing committed.

Baseline bound to: HEAD `e475a59`, source-bundle `a4497a2d`, package root `bfd578b8` (pinned),
checkpoint `88c14255`, TDX service hash in `baseline_freeze.json`. Data plane = A10↔TDX private VPC.

## Integrity correction (required)
The prior "A10 vs H800 CE delta ≈ 0.008 within tolerance" was **NOT preregistered** — it is recorded
as an **OBSERVED cross-platform difference** (bf16 sm_86 vs sm_90), not a pre-passed tolerance. The
properly pre-passed algorithmic-correctness metric is effective-equivalence **top1 = 1.0** vs the A10
plaintext model (fp32). See `baseline_freeze.json`.

## COMPLETED (real artifacts)
- **PHASE 2 — L12 trusted AdamW protocol** (`tdx_adamw_protocol.py`, ops in `tdx_persistent_service.py`,
  runner `a10_l12_runner.py`, orchestrator `gate0_a10_l12_orchestrator.py`):
  - Enclave holds m, v, step, θ_plain for the **168 trusted factors** — A of {q,k,v,gate,up} + B of
    {q,k} (γ-fed / non-monomial); **GPU-exact** monomial AdamW for the other 168 factors.
  - **fp64 self-test** (`test_l12_adamw_protocol.py`): transform-err 0.0, fold round-trip 8.9e-16,
    step-vs-reference 7.1e-15 over 5 steps → **exact**.
  - Fold convention verified against the gram_inv ground truth (`gA_plain = gA_tilde·Nrᵀdiag(γ)`,
    `A_tilde = A_plain·diag(γ)Nr`, `gram_inv = MᵀM`).
- **PHASE 3 — L12 one-step hardware gate** on real A10 + real TDX (`L12_adamw_hardware_gate.json`):
  - **fp32: effective-equivalence top1 = 1.0, next-logit KL = 5.8e-9, max_abs = 3.3e-4 → EXACT**
    plaintext-equivalent AdamW. Real GPU fwd/bwd + real enclave AdamW (NOT an fp64 simulation).
  - Counters: `adamw_init=1`, `adamw_step=1`, `trusted_target_state_missing=0`,
    **`untrusted_moment_returns=0`**, `untrusted_gamma_returns=0`, auth/replay/forbidden=0;
    attestation SUCCESS + debug=false (mr_td 8a56b29a).
  - bf16: top1 0.905 — bf16 storage precision on the larger AdamW updates ⇒ **fp32 master weights
    required for L12** (consistent with the prior known finding).

## MISSING (enumerated — not run this session; enabled on the validated fast path)
| # | Cell | Status |
|---|---|---|
| 1 | L12 vs L5 plaintext AdamW, 10-step, 3 seeds, per-target m/v/ΔW comparison | NOT RUN |
| 2 | 50-step 3-seed **L1/L2/L10** (SGD: plaintext / O1-A / O1-C) | NOT RUN (L10 1/10/50-step single-seed done previously) |
| 3 | 50-step 3-seed **L3/L4/L11** (momentum) | NOT RUN (L11 10-step single-seed done previously) |
| 4 | 50-step 3-seed **L5/L12** (AdamW) | NOT RUN (L12 1-step fp32-exact done) |
| 5 | Real rank-mask ablation (off/fixed/every-step/every-10/per-session/negative-control) | NOT RUN |
| 6 | Converged **SST-2** utility (L0/L5/L12, 3 seeds) | NOT RUN |
| 7 | Converged **GSM8K** utility (L0/L5/L12, 3 seeds) | NOT RUN |
| 8 | Timing/communication decomposition summary across profiles | PARTIAL (per-run timings recorded; no cross-profile summary) |

These are real training/eval runs (multi-hour) plus seed-plumbing for the L1–L5 profiles; they are
**compute/scope items on a validated, non-network-blocked path**, not blocked by protocol or infra.

## Notes
- The remaining runs require: (a) seed plumbing into `rank_masked_init` (currently one fixed init);
  (b) wiring L1–L5 plaintext/O1-A profiles into the A10 orchestrator; (c) real SST-2/GSM8K data +
  training loops; (d) `--dtype fp32` for L12 arms (bf16 loses AdamW-update precision).
- Do NOT average away O1-A NaN/divergent cells (gate/up) when the 50-step matrix is run.
- Light TDX secret hygiene done (session config + gamma bundle removed); package/env/forced-key
  retained on A10/TDX for resumption. A10 + TDX left running — user decides on stop/release.
- External-paper baselines / full security matrix / 7B: NOT started (correctly held for review).
