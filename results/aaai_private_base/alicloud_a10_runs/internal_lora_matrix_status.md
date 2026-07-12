# Internal LoRA matrix — closure status (rank-refresh + momentum-audit + freeze stage)

**Status: `INTERNAL_LORA_MATRIX_PARTIAL`** · Nothing committed by me.
Bound to code HEAD `f96bd26` (+ uncommitted edits), package `bfd578b8`, checkpoint `88c14255`.

## COMPLETED THIS STAGE (real A10 + real TDX, private VPC)

### PHASE 1 — momentum divergence audit (`l1_l11_matrix/PHASE1_momentum_divergence_audit.json`)
- Divergence is **seed-dependent and hits BOTH profiles** (L11 diverges s1234/s2025, stable s7; L4 diverges
  s7, stable s1234/s2025) → shared chaotic single-batch instability at effective LR ~1e-2, **not L11-only**.
- **Through-divergence oracle** (o1c momentum 50-step): worst master rel err = **0.0 every step including the
  divergence region** (ce→8.3), both seeds. ⇒ protected optimizer protocol remains equivalent; divergence is
  a shared plaintext-trajectory hyperparameter instability, NOT a protocol defect. Registered matrix NOT retuned.

### PHASE 2 — rank-refresh derivation + classification + implementation (`l1_l11_matrix/PHASE2_rank_refresh_classification.json`)
- `R = U_new@U_oldᵀ`; `A_new=R@A_old, B_new=B_old@Rᵀ` (effective ΔW invariant; commutes with γ fold).
- **Classification A–E**: SGD/momentum preserved by ANY invertible R (rebase θ, m); **AdamW elementwise v
  transports EXACTLY only for signed-permutation R** (Class C); dense R ⇒ trusted v-reconstruction (D);
  claiming dense exact = invalid (E).
- Local fp64 validation: signed_perm effective-ΔW invariance 6.7e-16, v-permutation exact 0.0; dense v→0.
- Implemented `TrustedAdamW.rebase(R,mode)` + service op `rebase_adamw` + runner `--refresh-*` flags.

### PHASE 3 — rank-mask refresh hardware ablation (`l12_mixed/PHASE3_rank_mask_ablation.json`, L12 AdamW, signed-perm)
- **R1 baseline / R2 every-step / R3 every-5 (correct)**: loss continuity preserved, converge to ≤ baseline
  loss (R2/R3 0.0009 vs R1 0.0037); R3 effective-adapter divergence 3% median (bf16 floor). Correct transport works.
- **R5 params_only (neg-control)**: measurably drifts — degraded convergence 0.0078 + 80% adapter drift.
- **R6 inconsistent A-not-B (neg-control)**: breaks — erratic non-monotone loss (jump to 0.71, final 0.16) +
  131% adapter divergence. All finite. Not claimed exact from algebra alone (real artifacts).

### PHASE 4 — utility budget FREEZE (`PHASE4_utility_budget_freeze.json`)
- Immutable preregistration for L0/L5/L12 SST-2 + GSM8K (seeds/splits/hashes/templates/verbalizers/lr/budget/
  margins/bootstrap); L5≡L12 settings, L12 not separately tuned. Recorded before any utility metric.

### PHASE 7 — final interpretation table (`PHASE7_final_interpretation.json`)
- Categories A (algebraic) / B (short-horizon numeric) / C (approximation failure) / D (utility) / E (systems)
  kept SEPARATE with explicit non-conflations (optimizer-equivalence ≠ utility; top1 ≠ bitwise; shared unstable
  trajectory ≠ protocol failure; effective-equiv-vs-HF checks fold-consistency not training-correctness).

### PHASE 8 — hashing + secure cleanup
- Reports hashed (`stage3_report_hashes.sha256`). TDX secrets (γ bundle, session config, checkpoint state)
  removed; A10 session state + tlogs + logits removed, GPU idle. Package/env/forced-key retained. **A10 + TDX
  left running.**

## NOT RUN THIS STAGE (PARTIAL boundary — enumerated)
| # | Cell | Why |
|---|---|---|
| 5 | **SST-2 converged utility** (L0/L5/L12, 3 seeds) | Data provisioning done (manifests/hashes/templates), but converged training needs a **batched training data plane** (per-batch input_ids on A10 + per-batch labels for the TDX ce_dlogits, which currently uses ONE fixed label set baked into the session); not built. |
| 6 | **GSM8K converged utility** (L0/L5/L12) | As above + a generation/decoding loop over the masked package; not built. |
| — | Utility equivalence analyses (bootstrap CI vs frozen margin) | Depend on 5/6. |
| — | Rank-refresh R0 (mask disabled), R4 (per-session), + SGD/momentum refresh variants | AdamW signed-perm (the HARD/motivating case) validated; these are Class A/B easy cases (any R) + variants, not built. |

## Do NOT start (held for review): external-paper baselines, full security matrix, 7B.
