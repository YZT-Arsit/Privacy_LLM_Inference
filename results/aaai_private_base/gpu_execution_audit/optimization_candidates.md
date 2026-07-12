# Optimization candidates (H800 protected worker)

**Framing first:** no correctness bug was found — the forward/backward run on CUDA (S1: 606/607
tensors on device, 0 disallowed). The step is **launch-overhead-bound** (S3: 104 ms CPU dispatch vs
13.4 ms GPU kernel time over 11,126 tiny kernels), and end-to-end the step is **transport-bound**
(S5: SSH + TDX ≫ compute). So GPU-compute optimizations have **low ROI** until transport is the
target. Ranked by (impact on real end-to-end step, risk):

| Candidate | Impact | Risk | Status |
|---|---|---|---|
| **A. Pack 120 A-grad stagings → 1 CUDA buffer + 1 DtoH** | Med (120→1 DtoH, ~3.4 MB single copy; fewer syncs) | Med (changes wire framing; TDX must unpack; must re-verify fail-closed forbidden-key check per sub-tensor) | **Proposed** — not applied (would perturb the just-hardened, validated protocol; do under a separate protocol rev + re-run fault tests) |
| **B. On-GPU norm accumulation (remove ~350 per-target `.cpu()`/step)** | Small (removes ~350 host syncs/step of self-inflicted telemetry) | Low (local to runner; identical values) | **APPLIED (S7)** — verified numerics unchanged |
| C. Pinned host memory + `non_blocking=True` for staging DtoH / dlogits HtoD | Small (overlap copy with compute) | Low | Proposed |
| D. `map_location=device` when deserializing dlogits/corrected wire buffers | Tiny (saves one CPU→GPU hop on ~MB) | Low | Proposed |
| E. CUDA graph capture of the forward (collapse 11k launches) | Med on pure-compute duty (would cut the 104 ms CPU dispatch) — **but invisible end-to-end** since transport dominates | High (eager graph with data-dependent shapes; must preserve masked semantics bit-exactly) | Proposed, low priority |
| F. Batch q/k/v (and gate/up) into fused projections | Small–Med launch reduction | Med (must keep per-projection folds/masks exact) | Proposed, low priority |
| G. Keep model+adapters+static masks resident on GPU; one persistent channel | Already done by the direct runner | — | **Done** |

## Where the real time goes (and the real optimization target)
Pure back-to-back compute is ~77 ms/step (S4) at ~23 % GPU-util. A **real** direct step is dominated
by the TDX round-trips (CE/dlogits + correction) and SSH framing (S5 `step_timeline.md`). The
highest-value work is therefore **transport**, not GPU: bf16 (not fp32) logits on the wire, avoiding
per-target RPC (candidate A), and overlapping SSH I/O with H800 compute. These do not change
numerical semantics or the security boundary.

## Explicitly NOT recommended
- Do **not** move any protected tensor off CUDA (already on CUDA).
- Do **not** fold the TDX CE/correction onto H800 to "raise GPU util" — that breaks the security
  boundary (labels/γ/gram_inv must stay in the enclave).
- Do **not** chase GPU-util as a headline: for a 0.5 B model at seq 42 a low duty cycle is expected;
  the meaningful metric is correctness + end-to-end latency, which is transport-limited.
