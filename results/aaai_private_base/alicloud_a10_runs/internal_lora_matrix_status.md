# Internal LoRA matrix — empirical closure status (L12 mixed-precision stage)

**Status: `INTERNAL_LORA_MATRIX_PARTIAL`** · Nothing committed by me.

Bound to: code HEAD `f96bd26` (+ uncommitted PHASE 1/2 edits), package root `bfd578b8`,
checkpoint `88c14255`, TDX service hash recomputed in `l12_mixed/L12MIX_s*.tdx_session.json`.
Data plane = A10 (172.30.25.154) ↔ TDX (172.30.25.153) private VPC.

## COMPLETED THIS SESSION (real hardware artifacts, `results/.../alicloud_a10_runs/l12_mixed/`)

### PHASE 1 — L12 mixed-precision design freeze + claim/registry correction
- `l12_mixed_precision_design_freeze.json`: L12 = `trusted_adamw_fp32_master_bf16_compute`;
  L12-N = `trusted_adamw_bf16_parameter_storage_negative_ablation` (top1 0.905, unstable — negative control only).
- Claim language corrected: dropped "max_abs 3.3e-4 therefore exact" → **"algorithmically
  plaintext-equivalent (optimizer-semantic) and numerically aligned in FP32"**. bitwise equality NOT claimed;
  BF16-compute/FP32-master = to-be-validated (now validated, see below); pure-BF16 storage = unsupported.
- Fold correction recorded: trusted A fold `M = diag(gamma)@Nr`, `gram_inv = M^T M = Nr^T diag(gamma^2) Nr`.
- **Fold-fix audit**: the obsolete `diag(1/gamma)@Nr` (= `T_in^{-T}`) appears legitimately ONLY as the
  self-consistent offline-proof convention in `full_matrix_equivalence.py` / `gate05_o1_optimizer_audit.py`
  (valid as self-contained proofs; must NOT be cross-applied to the deployed package). Scanned all committed
  `*.equiv.json` for the mis-scale signature (ce>5): **none** → `INVALIDATED_BY_FOLD_FIX = []`. The one
  transient cross-convention diagnostic (ce 16.50) was fixed and never persisted.

### PHASE 2 — authoritative FP32 master-state flow + encrypted checkpoint/recovery
- `tdx_adamw_protocol.py`: FP32 state dtype, monotonic `version`, run/package/adapter binding, AEAD
  (SHA256-CTR + HMAC-SHA256, no external dep) `checkpoint()/restore()` with fail-closed on tamper / wrong
  key / binding mismatch / version rollback. Self-test `test_l12_adamw_protocol.py`: transforms 1e-16,
  5-step 3.5e-15, fp32 state 4.8e-7, **checkpoint restore exact (0.0), all 4 fail-closed cases hold**.
- Service `tdx_persistent_service.py`: ops `checkpoint_adamw`/`restore_adamw`; new invariant counters.

### PHASE 3 — one-step, 3 preregistered seeds (vs L5 plaintext-AdamW oracle)
| seed | worst master rel A | worst master rel B | ΔW cos (q/k) | PASS |
|---|---|---|---|---|
| 1234 | 4.41e-9 | 1.58e-8 | 0.99999999999985 | ✓ |
| 2025 | 2.17e-9 | 1.53e-8 | — | ✓ |
| 7 | 2.70e-9 | 1.58e-8 | — | ✓ |

120 A + 48 B trusted factors compared per seed; ~6 orders below the preregistered 5e-3 (fp32 fold round-off).

### PHASE 4 — ten-step, 3 seeds
worst master rel A = 6.63e-9 / 9.41e-9 / 8.67e-9; state version +1 per step (1..10), no rollback,
missing=0, rt_match=true and finite every step.

### PHASE 5 — fifty-step, 3 seeds + restart/rollback continuity
- 50-step: ce 0.4454→~5e-5; **effective-equiv top1 = 1.0** (KL 3.0e-6 / 7.3e-7 / 6.3e-6); final version 50.
- **Restart test (seed 1234)**: checkpoint at step 25 (enclave v26) → **fresh attestation** (verified,
  debug_false) → restore after AEAD+run_id+package+adapter+version validation → continue to v50.
  **worst adapter rel err restart-vs-uninterrupted = 0.0 (bit-identical)**; **live rollback rejected**.

### PHASE 9 — performance / communication (50-step medians, private VPC)
step 0.59s; A10↔TDX round-trips = ce_dlogits 0.24s + enclave_adamw 0.15s (~66% of step); 2 trusted logical
invocations/step; GPU proof = on-host 100Hz logger caught peak 2753 MiB VRAM + util bursts to 72%.

**Counters (all runs, incl. 50-step):** authoritative_state=TDX/FP32, runtime=BF16,
runtime_copy_matches_master_transform=true, gpu_trusted_factor_optimizer_step=forbidden,
untrusted_{m,v,fp32_master}_materializations=0, untrusted_moment/gamma_returns=0, silent_fallbacks=0,
trusted_target_state_missing=0, auth/replay/forbidden=0. Consolidated: `L12_mixed_precision_gate_results.json`.

## NOT RUN THIS SESSION (PARTIAL boundary — enumerated)
| # | Cell | Why not run |
|---|---|---|
| 6a | 50-step 3-seed **L1/L2/L10** (SGD) + **L3/L4/L11** (momentum) | SGD/momentum are linear-equivariant ⇒ masked==plaintext exactly (no enclave); L10/L11 exactness shown single-seed in prior stage. 3-seed 50-step + L1–L4 plaintext/O1-A baselines need dedicated SGD/momentum runners (not built this session). |
| 6b | Rank-mask 10-step ablation (off/fixed/every-step/every-10/per-session/neg-control) | Needs runner support to refresh the orthogonal rank mask U mid-run + re-seed enclave state; not built. |
| 7 | **SST-2** converged utility (L0/L5/L12, 3 seeds, official train/dev) | Real convergence on official data (multi-hour over cross-cloud CE path) + offline dataset provisioning on A10; not run. |
| 8 | **GSM8K** converged utility (L0/L5/L12) | As above + generation eval; not run. |

These are compute/data-provisioning/runner-build items on a validated, non-network-blocked path — not
blocked by the L12 protocol or infrastructure. The L12 mixed-precision protocol (the core novel claim)
is fully validated on hardware at 1/10/50 steps × 3 seeds + restart + rollback.

## Do NOT start (held for review): external-paper baselines, full security matrix, 7B.
