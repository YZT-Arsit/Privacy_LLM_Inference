# Gate 1.5-D — Gradient / mask convention audit (frozen)

Repo-wide audit of `N_out^{-1}` / `N_out^{-T}` / `N_out^T` / `M_out^{-1}` / `M_out^{-T}`
usage in LoRA backward + recovery. **Finding: there are two *distinct, internally
consistent* conventions in the tree; the only defect was a transient mixing in the new
training service, fixed before any real run. No historical result is invalidated.**

## Frozen protocol definitions (the training service uses Convention 1)

Per LoRA layer, forward `Y = X(W + A B)`; forward output mask `N_out`; rank-space mask
`U` (a.k.a. `R`); input mask `N_in`.

| quantity | formula |
|---|---|
| forward output mask | `Y_t = Y N_out` |
| masked adapters | `A_t = N_in^{-1} A U`, `B_t = U^{-1} B N_out` |
| **upstream gradient convention** | `U_t = G_Y N_out^{-T}` (grad w.r.t. the masked output `Y_t`) |
| masked gradA (GPU, masked domain) | `gA_t = N_in^T gradA U^{-T}` |
| masked gradB (GPU, masked domain) | `gB_t = U^T gradB N_out^{-T}` |
| **gradA recovery (trusted)** | `gradA = N_in^{-T} gA_t U^T` |
| **gradB recovery (trusted)** | `gradB = U^{-T} gB_t N_out^T` |

Under **orthogonality** (masked-SGD mode) these collapse: `N_in^T = N_in^{-1}`,
`U^{-T} = U`, so `gA_t = N_in^{-1} gradA U` and `gB_t = U^{-1} gradB N_out` — the forms
used by `masked_sgd_audit`. Verified by `orthogonal_collapse_check` (agree <1e-10 for
orthogonal; differ >1e-3 for general GL).

## Per-item ledger

| item | file / function | formula | status |
|---|---|---|---|
| upstream inject (N_out conv.) | `src/pllo/ops/lora_backward.py::transform_upstream_gradient` | `G_tilde_Y = G_Y N_out^{-T}` (`grad_y @ inv(n_out).T`) | **correct**, Convention 1 |
| upstream un-inject | `ops/lora_backward.py::invert_upstream_gradient_mask` | `G = G_tilde N_out^T` | correct |
| gradB recovery (production) | `ops/lora_backward.py` docstring/impl | `grad_B = U^{-T} grad_B_tilde N_out^T` | correct, Convention 1 |
| gradA recovery (masked-cond audit) | `src/pllo/experiments/mask_conditioning_audit.py` (fixed earlier) | `G_Yt = GY N_out^{-T}`; `gradA_t = N_in^T gradA R^{-T}` | correct, Convention 1 |
| A0 dual inject | `experiments/masked_lora_backward_audit.py::a0_tee_grad_out` | `G_tilde_Y = G_Y N_out^{-T}` | correct (Convention 1) |
| A1 independent inject/recover | `masked_lora_backward_audit.py::a1_tee_grad_out / a1_recover` | `G_tilde_Y = G_Y M_out`; `grad_B = U^{-T} grad_B_tilde M_out^{-1}` | **correct under Convention 2** (independent backward mask `M_out`, NOT `N_out`) |
| training service (NEW) | `experiments/real_tdx_training_service.py::packed_update` | inject `U_t = U N_out^{-T}`; recover `gradB = U^{-T} gB_t N_out^T` | **FIXED** (was `N_out^{-1}` → now `N_out^T`); Convention 1 |

## The two conventions (both correct — do NOT mix)

1. **N_out (forward-mask-reused):** inject `G_Y N_out^{-T}`, recover `... N_out^T`.
   Used by production `ops/lora_backward.py`, `mask_conditioning_audit`, and the new
   training service.
2. **M_out (independent backward mask):** inject `G_Y M_out`, recover `... M_out^{-1}`.
   Used by `masked_lora_backward_audit` scheme A1 (the whole point of A1 is an
   *independent* backward mask that breaks the exact cross-Gram equality).

Both are self-consistent; the recovery is the exact inverse of the injection **within
each convention**. The bug was using Convention-1 injection with a Convention-2-style
`N_out^{-1}` recovery in the new service — a mismatch that produced wrong `gradB` unless
`N_out` was orthogonal (where `N_out^{-1} = N_out^T` accidentally coincide only if also
symmetric — it did not, hence the contract test caught it).

## Tests covering the convention
- `tests/test_real_qwen_protocol_contracts.py::test_packed_update_matches_plaintext_adamw_trajectory`
  (6-step, catches any gradB error: A/B/ΔW/Adam-m/v match plaintext <1e-9).
- `tests/test_masked_sgd_audit.py::test_orthogonal_collapse` (orthogonal vs GL forms).
- `tests/test_masked_sgd_audit.py` alignment cases (gradB recovery implicit in A/B alignment).
- `ops/lora_backward.py` is covered by the existing `tests/test_lora_backward_ops.py`.

## Re-run impact
**None.** The mixing existed only in the new `real_tdx_training_service.py` during this
session and was fixed before any real Qwen/TDX run. The production `ops/lora_backward.py`
and the A1 audit were already correct under their respective conventions. No prior
committed result depends on the buggy line.
