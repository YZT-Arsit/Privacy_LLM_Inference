# Batched authenticated data plane — stage status

**Status: `INTERNAL_LORA_MATRIX_PARTIAL`** · Nothing committed.
Real A10 (172.30.25.154) + real Intel TDX (172.30.25.153 private VPC). Package root `bfd578b8`.

The "only blocking objective" from the prior stage — a real batched training/eval data plane to
replace the single-fixed-batch demo — is **BUILT and hardware-validated**. Converged 3-seed utility
remains compute-bound (enumerated below).

## BUILT (PHASES 0–6)

- **PHASE 0** — utility preregistration frozen + hashed before any implementation
  (`PHASE0_prereg_seal.json`, `PHASE0_seal.sha256`). Arm definitions + GSM8K training template
  recorded as pre-metric deviations (`datasets/tokenized/deviations.json`).
- **PHASE 5+1** — official SST-2 (67349/872) + GSM8K (7473/1319) tokenized with the pinned tokenizer
  (`c0382117…`), verified vs the frozen manifest row counts; split into an **A10 input artifact**
  (input_ids + supervised-position metadata, **no labels**) and a **TDX private label table**
  (cls target token+label / clm shifted target with ignore_index). Deterministic per-seed batch
  schedules hashed (`datasets/tokenized/tokenized_manifest.json`).
- **PHASES 2+3+4 (TDX)** — `ce_batch`/`eval_batch` ops with a strict state machine: batch descriptor
  bound into a per-batch HMAC (`batch_mac`); monotonic ledger rejecting stale / repeat / out-of-order /
  wrong-sample-id / wrong-split / wrong-shape / wrong-ignore-mask / wrong-accum-window; classification
  CE and causal-LM CE (ignore_index) over the un-permuted vocab; **row-chunked clm transport** (≤256
  supervised rows/frame, shared full-batch denominator ⇒ summed chunk dlogits == mean-reduction
  gradient; ledger advances once per optimizer batch) so the CPU-only TDX guest keeps up; AEAD-sealed
  batch-ledger checkpoint/restore.
- **PHASE 1+3+4 (A10)** — `a10_batch_runner.py`: microbatch forward loop + grad accumulation,
  dlogits scatter-back + autograd, GPU-exact monomial AdamW + enclave FP32 trusted AdamW split, eval.
  A10 never holds labels/targets/loss/dlogits.
- **Local tests (PHASE 3/4)** — `test_batch_dataplane.py`: **35/35 PASS** — cls+clm CE & dlogits exact
  vs autograd / ignore_index oracle (batch 1/2/8/partial, mixed seq/answer lengths), ledger rejections,
  snapshot round-trip, descriptor-HMAC binding.

## HARDWARE GATES (real A10 + real TDX)

| gate | result |
|---|---|
| SST-2 L12 **1-batch** (cls) | CE 2.22, dev acc 0.6875 (32); ledger+enclave, 0 rejections |
| SST-2 L12 **10-batch** (cls) | CE 2.22→~0.03, dev acc **0.9375** (32); ~1.7 s/batch |
| SST-2 **L0** floor (eval only) | dev acc 0.875 (128), NLL 1.024 |
| SST-2 **L5** fp32 10-batch | trains (fp32 ref); dev acc/NLL recorded — SMOKE gate, not converged |
| GSM8K **L12 1-batch** (clm) | CE 0.8201 over 1710 sup tokens; chunked `ce_batch_calls=7`; eval NLL 0.653 |
| GSM8K **L12 10-batch** (clm) | 10 steps, ledger gbi 0–9 monotonic, `ce_batch_calls=88` (chunked), test NLL 0.653→**0.424**, all finite, 0 rejections |
| **PHASE 6 restart** (SST-2 L12) | **PASS** — resume@K, no dup/skip, contiguous ledger, **resumed vs uninterrupted rel_err = 0.0** |

All gate runs: `fail_closed_all_pass`, `package_root_hash_matches`, TDX auth/replay/forbidden/malformed
rejections = 0, authoritative optimizer state FP32 in TDX.

**These gate numbers are SMOKE TESTS on small differing eval subsets — NOT the converged utility
comparison and NOT bitwise claims.** See PHASE9_10_notes.json for the strict category separation.

## NOT RUN (PARTIAL boundary — compute-bound, enumerated)

| cell | why |
|---|---|
| **PHASE 7 — SST-2 converged** (L0/L5/L12 × 3 seeds) | ~1.7 s/batch × 12630 batches × 3 seeds × {L5,L12} ≈ tens of GPU-hours; exceeds one session. Data plane ready. |
| **PHASE 8 — GSM8K converged** (train + generation) | L12 clm ~2–3 min/step (CPU-only TDX softmax on ~600MB/step) → ~100 h/seed; **generation infeasible** (greedy needs a per-token trusted vocab un-permute → ~1M round trips). |
| PHASE 7/8 equivalence analyses (bootstrap CI vs frozen margin) | depend on the converged runs above |
| PHASE 9 R0 (no rank mask) | optional; needs unmasked-init path (R4 covered a fortiori by prior R2/R3; R5/R6 neg-controls isolate the mask role) |

## Held for review (not started): external baselines, full security matrix, 7B.
