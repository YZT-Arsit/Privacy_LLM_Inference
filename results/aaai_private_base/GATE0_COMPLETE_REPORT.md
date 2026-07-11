# GATE-0 REPORT — private-base + unified LoRA (real Qwen2.5-0.5B, real H800, real Intel TDX)

**Final status: `GATE0_COMPLETE`**

D2, D3, package, H800 dry run, and D4 (fresh-attested one-step + 10-step + deployment
smoke) all pass. Nothing committed. Proceeding to the full L0–L8 matrix is NOT done
(gated on review, per the plan).

---

## 1. H800 package transfer + hash verification
- Immutable package (root hash `bfd578b8…`, 242 transformed artifacts) byte-transferred
  to the H800 via a banner-safe `tar` over the authenticated SSH exec channel.
- **Destination root hash == `bfd578b8…` (byte-identical; 0 missing / 0 extra / 0
  differing files)** after stripping macOS AppleDouble `._*` sidecars.
- Fail-closed inventory: no plaintext checkpoint shard, no `*.safetensors`, no mask-secret
  file in the package. Provenance in `h800_unified_worker/` (`package_transfer_manifest`,
  `source_hashes.sha256`, `destination_hashes.sha256`, `destination_inventory`,
  `plaintext_checkpoint_inventory`).
- Recorded finding: an independent H800 **reproducible rebuild** validated functionally
  (fold err 5.7e-14) but produced a *different* root hash (`1076db08…`) due to
  cross-platform fp32 rounding — hence the literal byte transfer is required by the
  fail-closed root-hash rule.

## 2. Package-native loader evidence
- `h800_unified_worker.py::PackageNativeLoader` loads the 242 `*_tilde` artifacts and
  builds embedding / 24 decoder layers / folded RMSNorm (γ folded into projections) /
  Q,K,V,O / gate,up,down / final norm / tied-source LM-head view / masked LoRA.
- Per-component provenance (`loader_provenance.json`): every component
  `plaintext_source_loaded_by_worker=false`; `worker_loaded_plaintext_hf_model=false`;
  `worker_read_original_safetensors=false`.
- Tied-weight convention preserved: `tied_plaintext_source=true`,
  `transformed_views_count=2`, `transformed_storage_shared=false`,
  `cross_view_attack_surface=true`.

## 3. No plaintext HF model loaded on the worker path
The untrusted worker never calls `from_pretrained` on the protected path, never reads
`model.safetensors`, never reconstructs plaintext weights. (The HF plaintext model is
used ONLY by the trusted-eval verifier and the L1 reference — never the worker.)

## 4. All-24-layer execution provenance
- Dry run + D4 both execute all 24 layers; `per_layer_execution.csv` /
  `per_layer_gradients.csv` show **168/168** LoRA target instances (24 × 7) receiving
  gradients; base tensors `require_grad=false`.

## 5. H800 no-TDX dry run — `H800_UNIFIED_DRY_RUN PASS`
- Check A (base masked forward vs HF plaintext): **top1 1.0**, max|Δ| 3e-4, rel 2.3e-5.
- Check B: 168/168 LoRA connected, base no grad, one masked-SGD step finite, loss@init
  0.4554 vs HF 0.4539.
- Check C (operator LoRA equivalence, fp64, real folded base): max err **2.7e-15**.
- Counters all zero; artifacts in `h800_unified_worker/dry_run_metrics.json`.

## 6. Fresh TDX attestation evidence
- Real Alibaba TDX quote generated on the guest (`/dev/tdx_guest`, libtdx_attest),
  verified with `verifier -quote` (JWT appraisal; `relying_party` hangs on collateral).
- **`attestation_verified=true`**: appraisal **SUCCESS**, `reportdata` cryptographically
  **binds the full D4 manifest** (run_id + package root hash + worker/service/config/
  tokenizer hashes + all mask/optimizer profiles + nonce + ephemeral pub), **DEBUG=false**,
  mr_td extracted. Fresh nonce/ephemeral/run_id each run; a stale quote fails the
  reportdata binding. Evidence in `gate0_d4/attestation/attestation_evidence.json`.
- Attestation is called verified ONLY on appraisal+binding+debug-false — never on quote
  generation alone.

## 7. D4 one-step equivalence — `D4 PASS`
run_id `D4-1783795544-…`. Real H800 forward → real TDX loss boundary → real backward.

| metric | value |
|---|---|
| TDX CE vs independently-recomputed expected | 0.45415 vs 0.45414 (**ΔCE 1.4e-5**) |
| TDX dlogits vs expected re-masked dlogits | **max abs 1.4e-6** |
| Effective equivalence vs HF plaintext + un-folded LoRA | **top1 1.0**, KL 1.1e-9, CE diff 8.9e-8 |
| LoRA targets connected | **168/168**; base no grad; step finite |
| HMAC message auth (both directions) | verified |

## 8. 10-step trajectory — `10-STEP PASS`
run_id `D4-10STEP-…`. Protected CE **0.45415 → 0.28562** (monotonic; max consecutive
increase **−0.005** = never increased). Every step: TDX CE matches expected to ≤1.4e-5,
dlogits to ≤1.6e-6, 168/168 connected, HMAC ok, finite. Final effective equivalence
top1 **1.0**, KL 1.1e-9. Artifacts in `gate0_post_d4_10step/`.

## 9. Deployment smoke — `DEPLOYMENT SMOKE PASS`
Frozen masked adapter `99e40193…` (from the 10-step run) loaded into the SAME
package-native forward. Held-out GSM8K-test (4 prompts): mean CE 3.88 / ppl 48.5
(**smoke only — not a task-utility claim**). Assertions all hold: `same_transformed_package`,
`same_base_weight_view`, `same_qkv_forward`, `same_nonlinear_backend`,
`same_residual_domains`, `training_adds_backward_only`, `adapter_reencoding_required=false`,
`plaintext_adapter_materializations=0`, `plaintext_base_materializations=0`.

## 10. Trusted invocation + RPC accounting
- **Logical trusted invocations/step = 2**: (1) trusted input/session provisioning
  (attestation + private labels + session key), (2) TDX loss/dlogits boundary.
- Physical transport: ~6 SSH exec/ferry hops per step (Mac-orchestrated, not a direct
  socket). `packed_update_calls=0`, `trusted_optimizer_calls=0`. Logical ≠ physical is
  recorded honestly; the invocation convention was NOT altered to improve the count.

## 11. Communication + latency
Per step: **25,526,860 B** masked logits sent, **25,526,867 B** masked gradient returned;
TDX transform+CE **65 ms**; ferry ~47 s (slow cross-cloud tunnels). `communication_metrics.json`,
`performance.csv`.

## 12. Forbidden plaintext counters (untrusted worker) — all zero
`untrusted_worker_plaintext_{base_weight,embedding,hidden,lora,gradient}_materializations
= 0`, `nonlinear_trusted_calls=0`, `packed_update_calls=0`, `trusted_optimizer_calls=0`,
`silent_fallbacks=0`. A missing counter is not treated as zero — all are emitted.

## 13. Fail-closed tests
`gate0_d4/d4_fail_closed_tests.json` (+ dry-run `fail_closed_tests.json`): package
root-hash match, `paper_safe` profile, backend ≠ current/trusted_shortcut, Qwen2.5-0.5B
config, no plaintext shard, all tensors `*_tilde`, no mask secret. `failed_checks.json`
is empty. TDX loss service rejects bad HMAC / replay / malformed fail-closed.

## 14. Blockers
None outstanding for Gate-0. Honest caveats (not blockers):
- Transport is Mac-ferried (authenticated messages), not a direct H800↔TDX socket —
  documented; security properties (in-enclave CE/dlogits, HMAC, attestation) hold
  regardless of transport.
- Optimizer-domain note: the γ-fold makes the LoRA input basis non-orthogonal, so exact
  masked-SGD↔plaintext-SGD equivalence is the O1 question; D4 validates the **forward /
  loss / dlogits / effective-weight** equivalence exactly (top1 1.0) and reports the
  masked-SGD update as the paper_safe O1-A profile.
- Deployment smoke is not a utility claim (10 steps, one prompt).

## 15. Files changed (new; nothing committed)
`scripts/`: `gate0_build_private_package.py` (env-overridable ckpt), `h800_unified_worker.py`,
`h800_d4_worker.py`, `d4_trusted_verifier.py`, `tdx_trusted_loss_service.py`,
`gate0_d4_attestation.py`, `gate0_d4_orchestrator.py`, `h800_d4_deployment_smoke.py`.
`results/aaai_private_base/`: `h800_unified_worker/*`, `gate0_d4/*` (+`attestation/`),
`gate0_post_d4_10step/*`, `lora_deployment_smoke/*`, `GATE0_COMPLETE_REPORT.md`.

## 16. Tests
Package unit tests green (10/10). D3 (840 checks) + dry-run (Checks A/B/C) + D4 boundary
(CE 1.4e-5, dlogits 1.4e-6) + 10-step + smoke all pass on real hardware.

## 17. Failed runs (preserved)
- First orchestration hung ferrying a 2.86 GB effective-ΔW file to the Mac (bug: verifier
  reads it locally on H800) → fixed (compute effective ΔW from the 17 MB LoRA state).
- First attestation parse read `appraised_reports[0]` (QE/TCB report, no reportdata) →
  fixed to search for the TD report carrying `tdx_reportdata`.
- Effective-equivalence top1 0.93 → temporal misalignment (post-step ΔW vs pre-step
  logits) → fixed by snapshotting the pre-step LoRA state → top1 1.0.

## 18. Git status
Uncommitted (working tree modified/untracked); HEAD unchanged at `36b2d6c`.

## 19. Nothing committed.

**`GATE0_COMPLETE`** — awaiting review before the full L0–L8 validation matrix.
