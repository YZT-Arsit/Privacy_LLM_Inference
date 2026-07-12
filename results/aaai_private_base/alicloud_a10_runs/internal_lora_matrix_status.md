# Internal LoRA matrix — empirical closure status (AEAD + SGD/momentum stage)

**Status: `INTERNAL_LORA_MATRIX_PARTIAL`** · Nothing committed by me.

Bound to: code HEAD `f96bd26` (+ uncommitted edits this stage), package root `bfd578b8`,
checkpoint `88c14255`. Data plane = A10 (172.30.25.154) ↔ TDX (172.30.25.153) private VPC.

## COMPLETED THIS STAGE (real hardware artifacts)

### PHASE 0 — standard AEAD optimizer-state sealing (`l12_mixed/PHASE0_standard_aead.json`)
- Replaced prototype SHA256-CTR+HMAC with **ChaCha20-Poly1305** (RFC 8439). Prototype preserved as
  `prototype_aead_seal/open` (magic `L12PROTOv0`) — history not erased.
- HKDF-SHA256 dedicated sealing key, **domain-separated** from transport HMAC/session key; random unique
  96-bit nonce per seal + in-session nonce-reuse ledger; AAD binds run_id/package_root/adapter_id/
  optimizer_profile/model_config_hash/service_hash/optimizer_state_version/checkpoint_sequence.
- Self-test: restore exact (0.0); **8/8 fail-closed** (tamper, wrong-key, rollback, bad-package,
  bad-profile, bad-adapter, bad-checkpoint-seq, nonce-reuse).
- **Hardware restart regression** (A10+TDX, seed 1234, ckpt@8, fresh attestation): worst adapter rel err
  restart-vs-uninterrupted = **0.0**, live rollback **rejected**, AEAD on wire = ChaCha20Poly1305.

### PHASE 1 — SGD matrix L1/L2/L10 (`l1_l11_matrix/PHASE1_sgd_matrix.json`), 3 seeds × 50 steps
- **L10 (O1-C) plaintext-EXACT**: effective top1 = 1.0 (all seeds); 10-step per-target oracle
  worst_master_A = **0.0**, worst_master_B 1.6e-8, dW cos 0.9999999999998, PASS.
- **L2 (O1-A) DIVERGES** (not hidden): median ~2–3% / max ~25% effective-ΔW relerr vs O1-C on γ-fed
  targets (propagating to all targets through the coupled trajectory); does not flip top1 at this
  single-batch overfit scale (ce 0.45→~0.16), but a real measurable divergence. All finite (no NaN).

### PHASE 2 — momentum matrix L3/L4/L11 (`l1_l11_matrix/PHASE2_momentum_matrix.json`), 3 seeds × 50 steps
- **L11 (O1-C) protocol-EXACT**: 10-step per-target oracle worst_master_A **0.0**, B 1.6e-8, dW cos
  1-2e-13, PASS. Momentum buffer verified DIRECTLY (oracle reconstructs plaintext-metric buffer from
  REAL logged grads; per-step buffer-derived master matches ~1e-8).
- **Honest hyperparameter finding**: at preregistered lr=1e-3 / mom=0.9 (effective LR ~1e-2), single-batch
  training DIVERGES by step ~50 (ce→3.8–5.5, still finite) for both O1-C and O1-A, so 50-step
  effective-equivalence top1 (0.81–0.98) is uninformative about protocol correctness — the 10-step
  pre-divergence oracle is the valid check and PASSES. Divergence retained, not smoothed.

### PHASE 8 — performance/communication aggregation (`l1_l11_matrix/PHASE8_perf_comms_aggregation.json`)
- Private-VPC deployment-like path ONLY (AutoDL throttled timings excluded). Per-step medians:
  L10 sgd 0.534s, L11 mom 0.536s, L12 adamw 0.587s; network ~67–70% (ce_dlogits ~0.24s + trusted step
  ~0.14s); 2 trusted logical invocations/step (O1-A = 1).

### PHASE 9 — hashing + secure cleanup (this doc + `l1_l11_artifact_hashes.sha256`)
- All new artifacts hashed. TDX: gamma bundle + session config + checkpoint state + attest temp removed
  (verified gone), no service/listener. A10: session config + gpu-state + gpulog + tlogs + logits removed,
  GPU idle. Package/env/forced-key retained. **A10 + TDX left running** (user decides stop/release).

## NOT RUN THIS STAGE (PARTIAL boundary — enumerated)
| # | Cell | Why |
|---|---|---|
| 3 | Real rank-mask ablation (off/fixed/every-step/every-10/per-session/neg-control) | Needs runner support to refresh the orthogonal rank mask U mid-run + re-basis the trusted optimizer state, plus a negative control that skips the re-basis; not built this stage. |
| 4 | Freeze utility budgets (templates/verbalizers/margins/…) | Not frozen (utility not run). |
| 5 | SST-2 converged utility (L0/L5/L12, 3 seeds, official train/dev) | Real convergence on official data (multi-hour over cross-cloud CE path) + offline SST-2 provisioning on A10; not run. |
| 6 | GSM8K converged utility (L0/L5/L12) | As above + generation eval; not run. |
| 7 | Utility/equivalence interpretation (equivalence margin, not CI overlap) | Depends on 5/6. |

## Do NOT start (held for review): external-paper baselines, full security matrix, 7B.
