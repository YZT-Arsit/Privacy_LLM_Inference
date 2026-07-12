# GPU-execution audit — H800 protected worker (16-point report)

**Scope:** prove (not assume) whether the protected Qwen2.5-0.5B forward/backward run on CUDA, and
explain the "GPU idle / CPU-bound" observation. Real hardware: NVIDIA **H800 PCIe** (AutoDL) worker +
real Intel **TDX** guest (Alibaba, `39.96.4.252`). dtype bf16, seq_len 42, L10 = O1-C hybrid.
Nothing committed.

## Verdict up front
The protected **forward and backward run entirely on CUDA**. There is **no CPU-compute fallback and no
device-placement bug**. The process looks "idle/CPU-bound" for two real reasons, neither a compute bug:
1. **Dominant — throttled TDX→H800 network:** a real direct step spends **99.3 % of wall waiting on the
   network** to move the dlogits (and corrected grads) back to the GPU host (measured ~0.1 MB/s ingress).
2. A **service bug (found + fixed):** the enclave returned **fp32** dlogits (25.5 MB) even for bf16 runs,
   doubling the throttled download and tripping the 120 s watchdog. Now returns bf16 (12.76 MB),
   **provably semantics-preserving** (bit-identical CE/top1/KL vs the ferried reference).

Secondarily, pure compute is **launch-overhead-bound** (eager, tiny model): 104 ms CPU dispatch vs
13.4 ms GPU kernel time over 11,126 tiny kernels — expected for 0.5 B @ seq 42, not a bug.

---

### 1. Did the real forward run on CUDA?
**Yes.** S1 device inventory: **606/607 tensors on CUDA, 0 disallowed on CPU** (the 1 CPU tensor is the
q/k/v/gate/up A-grad transport-staging buffer). Embedding, all 168 base weights, 168+168 LoRA A/B,
RMSNorm, Q/K/V, attention scores/probs, attn-out, gate/up/down, residuals, final hidden, logits — all
`cuda:0`. Forward matmuls execute as cuBLAS Hopper `nvjet_tst_*` kernels (S3).

### 2. Did the real backward run on CUDA?
**Yes.** `torch.autograd.grad(logits, params, dlogits)` over on-device params; PyTorch even logs
"running cuBLAS … set the primary context" for the backward. `aten::mm`/`aten::bmm` show CUDA
self-time in the profile; S5 real backward = **0.509 s** of GPU work (launch-bound). Gradients recorded
on `cuda:0`.

### 3. Percentage of step time in CUDA kernels
- **Isolated pure compute:** GPU kernel time 13.4 ms / step-wall ~77 ms ≈ **17 %** (≈9 % of the
  profiler-instrumented 147 ms step). CUDA op self-time 27.9 ms.
- **Real end-to-end direct step (S5):** GPU compute 0.81 s / **155.85 s** = **0.52 %**; CUDA kernels
  ≈13.4 ms / 155.85 s ≈ **0.009 %**. The step is transport-bound.

### 4. GPU active-duty cycle
- **Pure back-to-back compute (S4, 300 steps @ 200 ms sampling):** **85.7 %** of samples had GPU-Util>0,
  **23.0 % average util**, max 34 %, power 86.6 W (vs 50 W idle), **2143 MB VRAM resident**. GPU is busy
  most of the time but underused because each kernel is microscopic.
- **Real direct step:** duty ≈ compute/wall ≈ **0.5 %** — the GPU finishes in ~0.8 s then waits ~155 s
  on the network.

### 5. CPU-fallback locations
**None.** S2 during one instrumented step: **0 `.item()`, 0 `.numpy()`, 0 host matmul, 0 NumPy/Scipy
linalg, 0 `torch.save`**. The only `.cpu()` calls (120) are the q/k/v/gate/up A-grad **stagings to TDX**
(5 targets × 24 layers) — legitimate trusted-boundary serialization, explicitly allowed. The plaintext
HF/`AutoModel` path exists only in the eval-only `TrustedVerifier` (`--mode` diagnostics), never in the
protected forward or the direct runner.

### 6. DtoH/HtoD transfer count and bytes
- **Real direct step (S5):** up logits **12.76 MB**, down dlogits **12.76 MB** (bf16 — was 25.5 MB fp32
  before the fix), corr up **3.48 MB**, corrected down **3.48 MB**. Total ~32 MB/step over the SSH channel.
- **Isolated step:** 120 DtoH stagings (3.44 MB total) + 1.73 MB HtoD. **Device-side** memcpy is
  negligible (profiler: HtoD 185 µs / DtoH 249 µs / DtoD 95 µs). **PCIe is not the bottleneck — the SSH
  network is.**

### 7. Synchronization count
**7** `torch.cuda.synchronize()` in the profiled step (all from the audit's own phase timing). The req-1
norm telemetry previously forced **~350 per-target `.cpu()` host syncs/step** — **fixed** to accumulate
on-GPU and sync once per aggregate (~6/step). No `CUDA_LAUNCH_BLOCKING`, no anomaly/deterministic debug
modes.

### 8. Largest CPU bottlenecks
1. **Throttled TDX→H800 network download (dominant):** net_ce 123.0 s + net_correct 31.9 s = **99.3 %**
   of the 155.85 s step — I/O wait, not compute.
2. **Kernel-launch/dispatch overhead:** 104 ms CPU self-time dispatching 11,126 tiny kernels (eager,
   `aten::mm` alone 19.5 ms CPU vs 5.9 ms CUDA).
3. The fp32-dlogits bug (now fixed) had doubled (1).

### 9. Largest GPU kernels
cuBLAS Hopper GEMMs `nvjet_tst_64x8_64x16_4x2_v_bz_NNT` (234×, 760 µs), `…1x4_h_bz_TNT` (144×, 586 µs),
`…64x48…TNT` (96×, 532 µs); elementwise `vectorized_elementwise_kernel<CUDAFunctor_add,bf16>` (978×,
1538 µs) and `…MulFunctor` (768×, 1098 µs). `aten::mm` aggregate CUDA self-time 5.9 ms; `aten::bmm`
(attention) 0.45 ms.

### 10. Is low GPU utilization expected or a bug?
**Expected, plus one now-fixed bug.** Low duty for a 0.5 B model at seq 42 is intrinsic (launch-bound).
The multi-minute idle per step is **network throttle** (~0.1 MB/s TDX→H800) amplified by the **fp32
dlogits bug** — both transport, not compute. **No device-placement or CPU-fallback bug exists.**

### 11. Proposed fixes
- **Applied:** bf16 dlogits from the enclave (halves the throttled download; semantics-preserving);
  on-GPU norm accumulation (removes ~350 host syncs/step); watchdog resized (600 s) for the throttled
  link; pre-existing `head.txt` path bug in the orchestrator.
- **Next (this is the real target):** raise **H800↔TDX transport throughput** so network wait stops
  dominating — dedicated transport benchmark + fixes (binary framing/no-PTY/large buffers/one-flush/
  prealloc/no-base64/compression). Lower priority: pack the 120 stagings into 1 buffer, pinned memory +
  `non_blocking`, CUDA graphs. GPU-compute optimization has ~0 end-to-end ROI until transport is fixed.

### 12. One-step post-fix correctness
S5 direct step with all fixes vs ferried bf16 reference — **VERDICT: CONSISTENT**:
CE **0.45357903838157654** (Δ=0), top1 **0.976190** (=ferried), next-logit KL **0.008147** (=ferried),
equiv max_abs 4.00750 (=ferried), counts 48 GPU-exact + 120 in-enclave (miss 0). Security counters all
zero (auth/replay/forbidden/untrusted-γ/missing). Attestation **verified** (SUCCESS, reportdata-bound,
debug=false). Session reuse: single SSH channel, 3 requests (= 1 handshake + 2/step), seqs monotone,
attestation once.

### 13. Files changed (uncommitted)
- **New:** `scripts/gpu_execution_audit.py`, `scripts/gpu_audit_sample.sh`,
  `scripts/build_step_timeline.py`, `scripts/test_direct_transport_faults.py`,
  `scripts/compare_direct_vs_ferried.py`.
- **Modified:** `scripts/tdx_persistent_service.py` (bf16 dlogits return; HMAC-key commitment check),
  `scripts/h800_direct_runner.py` (on-GPU norms; VRAM telemetry; timeout/bound/abort; READ_TIMEOUT 600),
  `scripts/gate0_direct_orchestrator.py` (`head.txt` path; HMAC-key commitment; equiv verifier call).
- **Artifacts:** `results/aaai_private_base/gpu_execution_audit/` (inventories, profiler trace, kernel/
  operator CSVs, utilization/timeline JSON+CSV+MD, hot-path + optimization MD, this report).

### 14. Tests
- Fault-injection (`test_direct_transport_faults.py`): **all pass** (bad-HMAC/replay/forbidden-key/
  wrong-runid/truncated/recover + counters).
- Device inventory: **PASS** (no main tensor on CPU).
- bf16-preservation compare: **CONSISTENT**.
- One-step post-fix direct: **PASS** (CE/top1/KL exact, counters 0, attestation verified).

### 15. git status
Working tree was clean at session start; now has the above uncommitted modifications + untracked audit
artifacts. Branch `fix/folded-remote-repeat-debug-and-guards`.

### 16. Commit status
**Nothing committed.**

---

## Data appendix (files in this directory)
`tensor_device_inventory.csv`, `module_device_inventory.csv`, `cpu_fallback_events.csv`,
`host_sync_events.csv`, `pcie_transfer_summary.json`, `torch_profiler_trace.json`,
`operator_time_breakdown.csv`, `cuda_kernel_summary.csv`, `gpu_timeseries.csv`, `cpu_timeseries.csv`,
`utilization_summary.json`, `step_timeline.csv`, `step_timeline.md`, `step_timeline_fractions.json`,
`bf16_preservation_compare.json`, `hot_path_audit.md`, `optimization_candidates.md`, `audit_summary.json`.
