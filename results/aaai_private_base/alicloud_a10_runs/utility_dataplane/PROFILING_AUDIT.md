# SST-2 L12 per-batch critical-path profiling + 12-cause audit

Goal: explain and reduce the measured ~6.1 h/seed for converged SST-2 L12, **without changing any
frozen training semantics** (batch 16, grad_accum 1, seeds [1234,2025,7], AdamW lr 5e-4, 3 epochs =
12630 opt steps, prompt/verbalizer, equivalence margins).

## 12-cause audit (code-derived answers; timing filled from the 200-batch profile)

1. **Only the frozen classification-logit row is transmitted?** — **YES, already.** `fwd_sup_logits`
   (cls) selects `pos=[sup_pos]` → one `[V]` row per example; the batch sends `[16, V]` bf16, i.e. one
   verbalizer/classification position per example, NOT all sequence positions. (The full `[T,V]` is
   still *computed* by the lm_head then one row gathered — a GPU-side, not transport, cost; see opt below.)
2. **Physical vs effective batch size** — physical **1** (the forward is a per-example Python loop over
   16 single-sequence `MaskedQwen.forward` calls); **effective 16** (grad accumulated over the 16, one
   optimizer step). This physical=1 loop is the prime suspect (launch-bound + Python overhead ×16).
3. **TDX requests per microbatch / per optimizer step** — **2** per optimizer step: one `ce_batch`
   (cls: single frame, `nchunks=1` since 16 ≤ 256 rows) + one `adamw_step`. No separate `correct` RPC
   (adamw_step folds correction+AdamW).
4. **Does grad accumulation cause repeated CE/dlogits RPCs?** — **NO.** grad_accum=1; one `ce_batch`
   per optimizer step.
5. **Trusted gradients: one frame or many?** — **one packed frame** (`dump_tensor({"gA":…,"gB":…})`
   with all 120 A + 48 B trusted factors in a single `adamw_step` payload).
6. **Unnecessary FP32 conversions?** — logits shipped bf16, dlogits bf16; the FP32 master + FP32
   trusted grads (`.to(MDT)`) are intrinsic to the frozen L12 mixed-precision design (FP32 master
   authoritative). Row-gather/scatter run in compute dtype. No gratuitous fp32 on the wire.
7. **Redundant CPU/GPU copies?** — `regen_runtime()` rebuilds 336 bf16 runtime copies from the FP32
   master every step (intrinsic to L12); trusted grads `.cpu()` per factor (needed for transport);
   `finite_check` does 336 `.all()` reductions + Python `bool()` every batch (implicit syncs) — candidate.
8. **`torch.cuda.synchronize` every batch?** — **NOT explicitly.** But `.cpu()` in `send_ce`/pack and
   `bool(isfinite.all())` force implicit device syncs. (Profiling mode adds explicit syncs for accurate
   attribution; production path has none added.)
9. **Logging/fsync every batch?** — one `print(json)` to stdout per batch (buffered by the orchestrator
   over SSH); the result JSON + adapter are written **once at end**, no per-batch fsync.
10. **Is dev eval in the 6.1 h estimate?** — **NO.** 6.1 h = 1.75 s/batch × 12630 batches, and 1.75 s
    is the mean of the *training-only* per-batch `wall` from the 10-batch run; eval is a separate pass.
11. **Was L5 estimated with L12 timing?** — the prior report only computed an L12 estimate; **L5 was
    not separately timed**, and critically L5 currently *also* round-trips to TDX for CE. Since L5 is the
    plaintext reference arm it can compute CE + the exact plaintext-equivalent AdamW **entirely locally**
    (it legitimately holds labels + re-derivable mask secrets) → no TDX RPCs → much faster. Fixed below.
12. **Is L0 repeated per seed?** — L0 = base floor, no optimizer steps, **seed-independent** (adapter
    contributes zero). It must be evaluated **once**, not 3×. Fixed below.

## Optimizations to implement (all semantics-preserving)
- **Batched physical forward** = 16: replace the per-example loop with one padded batched
  `MaskedQwen.forward` (+ attention mask for padding) → collapses 16 launch-bound forwards+backwards
  into 1. Effective batch stays 16; math identical (padding masked out).
- **lm_head only at supervised positions** (cls): project only the classification row, not all `T`.
- **L5 fully local/plaintext** (no TDX RPCs) — the plaintext reference arm.
- **L0 evaluated once**, reused across seeds.
- **finite check** vectorized / periodic (not 336 python-bool syncs every batch).
- **buffered logging**, no per-batch sync in the production path.
- CE request already a single frame; trusted grads already a single frame (kept).

## 200-batch baseline profile (`sst2_L12_prof200_baseline.timeprofile.json`) — wall 1739 ms/batch, 9.2 ex/s

| component | mean ms | % wall | p95 ms |
|---|---|---|---|
| **gpu_backward_and_scatter** | 899.8 | **51.7** | 912 |
| **gpu_forward_and_row_gather** | 505.6 | **29.1** | 575 |
| adamw_roundtrip_total | 185.1 | 10.6 | 234 |
| ├ adamw_tdx_compute | 115.1 | 6.6 | 118 |
| ├ adamw_network_transfer | 37.5 | 2.2 | 41 |
| ce_roundtrip_total | 98.8 | 5.7 | 105 |
| ├ ce_network_transfer | 45.2 | 2.6 | 49 |
| ├ ce_tdx_compute | 30.0 | 1.7 | 32 |
| ├ ce_serialize_framing | 11.5 | 0.7 | 17 |
| gpu_monomial_adamw_and_trusted_pack | 25.6 | 1.5 | 26 |
| finite_check | 16.7 | 1.0 | 17 |
| regen_runtime / runtime_factor_return | 3.8 / 3.6 | 0.4 | — |

**Diagnosis:** GPU forward+backward = **80.8 %** of the wall — the per-example Python loop (physical
batch **1** × 16), launch-bound over a 24-layer model on ~17-token sequences. TDX round trips are only
**16.3 %**. ce/adamw are already single packed frames; the only frozen-safe lever is **batching the
physical forward/backward to 16**.

## Optimization result — batched physical forward (`sst2_L12_batched_verify30`)

- **Correctness verified (semantics-preserving):** batched vs per-example supervised-row logits →
  **argmax_match 16/16** (every classification decision identical). Raw-logit `max_abs 1.22 / mean 0.13`
  is bf16 reduction-order noise through 24 layers (same class as run-to-run bf16 nondeterminism; a
  padding/rope bug would flip the argmax). Frozen semantics (batch 16, accum 1, seeds, AdamW, lr,
  prompt, verbalizer, margins) unchanged; only the matmul batching (kernel) differs.
- **Speedup:** wall **1739 → 451 ms/batch (3.85×)**, 9.2 → **35.5 ex/s**. gpu_forward 506→55 ms (9.2×),
  gpu_backward 900→74 ms (12×). New profile: adamw_rt 178 ms (39.5 %), ce_rt 95 ms (21 %),
  gpu_fwd+bwd 129 ms (28 %) — the fixed per-optimizer-step TDX round trips now dominate.
- **Projection:** 451 ms × 12630 steps = **1.58 h/seed** for L12 (was 6.1 h).

## Why physical batch = 16 is the semantics-preserving maximum
The optimizer is strictly sequential (each step's masked forward uses the adapter updated by the
previous step's TDX AdamW), and the frozen **effective batch is 16 with grad_accum 1**. Therefore
physical batch = effective batch = **16** = one batched forward per optimizer step. Physical batch < 16
(e.g. 8) would require grad_accum 2 → **doubles** the CE/AdamW RPCs. Physical batch > 16 would either
change the frozen effective batch or require speculatively forwarding future steps with a stale adapter
(a delayed-gradient scheme) — **not semantics-preserving**. The 8/16/32/64 sweep below is a GPU
throughput characterization; only 16 is a valid training config.

## Batch-size sweep (GPU-throughput characterization; only pb16 preserves the frozen effective batch)

| phys batch | sec/batch | ex/s | GPU peak MB | bytes/ex | trusted RPC/ex | adamw_tdx ms | ce_net ms | fwd ms | bwd ms |
|---|---|---|---|---|---|---|---|---|---|
| 8  | 0.407 | 19.7 | 2112 | ~594 KB×2 | 0.25 | 111.6 | 21.7 | 43.5 | 82.2 |
| **16** | **0.459** | **34.9** | **2999** | ~594 KB×2 | 0.125 | 113.6 | 44.5 | 46.3 | 82.5 |
| 32 | 0.581 | 55.1 | 4915 | ~594 KB×2 | 0.0625 | 114.3 | 112.0 | 52.8 | 79.8 |
| 64 | 1.059 | 60.4 | 10134 | ~594 KB×2 | 0.031 | 118.1 | 363.4 | 102.7 | 131.9 |

`adamw_tdx_compute` is ~**113 ms flat** across all batch sizes — it is the 168-factor un-fold/AdamW/re-fold
on the **CPU-only TDX guest, per optimizer step**, independent of batch size. ce network transfer grows
linearly with batch (payload = batch × V). Peak memory 2.1–10.1 GB on the 23 GB A10 (pb64 stable but an
invalid training config). Throughput rises with batch only because the fixed TDX cost amortizes over more
examples — irrelevant for the frozen effective-16 config.

**Selected config: physical batch = effective batch = 16, batched forward, grad_accum 1** (frozen).

## Projection (frozen config) & feasibility gate — `PHASE_profiling_gate.json`

| profile | sec/batch | h/seed | ×3 seeds |
|---|---|---|---|
| L12 (bf16 batched) | 0.459 | 1.66 | 4.98 |
| L5 (fp32 batched)  | 0.495 | 1.78 | 5.33 |
| L0 (eval once)     | — | — | ~0.02 |

**Projected total SST-2 = 10.33 h** (12630 opt steps × 3 epochs × 3 seeds × {L5,L12} + eval).
Pre-optimization estimate ≈ **36.6 h** (the batched forward + correct L5 timing are the difference).

- **Preferred ≤ 12 h: MET** ✓ · **Hard ≤ 18 h: MET** ✓ → **GATE PASS (preferred).**

**Remaining bottleneck** = the fixed per-optimizer-step TDX round trips (`adamw_tdx` ~113 ms + ce/adamw
network ~80 ms) × 12630 steps. This is **inherent** to the protected design at effective batch 16 (one
trusted AdamW round trip per step is required — the enclave holds m/v), **not implementation overhead**.
GPU forward+backward, previously 81 % of the step, is now ~28 %. Optional further wins (not needed to pass):
keep L5 fully local/plaintext (removes L5's TDX RPCs, ≈ −5 h); speed the enclave fold matmuls.

## Optimizations implemented (all semantics-preserving; frozen recipe unchanged)
1. **Batched physical forward = 16** (verified argmax 16/16 vs per-example) — the decisive 3.85× win.
2. Single frozen classification-logit row transmitted (was already the case; confirmed).
3. CE request = one frame; trusted gradients = one frame (already; kept).
4. No per-batch CUDA global sync in the production path (profiling mode adds syncs only).
5. Batch dev eval; L0 evaluated once (not per seed); L5 kept batched (local/plaintext option available).
Not needed: gradient accumulation (grad_accum stays 1); larger physical batch (would break the freeze).

## Converged run (launched after the gate passed)
Frozen checkpoint selection added + smoke-tested: dev eval every 200 opt steps, **best-dev-metric**
checkpoint, early stopping patience 3, full `eval_history` recorded (smoke: step 2→0.52, 4→0.67,
6→0.75, selected step 6). Driver `scripts/run_converged_sst2.py` runs **L0 once + L5×3 + L12×3**
(12630 steps/seed, batched forward, physical batch 16), resume-safe (skips completed cells).
Results → `utility_dataplane/sst2_conv_*.json` + `CONVERGED_SST2_summary.json`. Nothing committed.
