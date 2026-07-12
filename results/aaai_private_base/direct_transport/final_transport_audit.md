# Final transport + GPU-execution audit

**Final status: `CURRENT_ENVIRONMENT_VALIDATED_BUT_UNSUITABLE_FOR_LONG_RUNS`**

The persistent direct H800↔TDX protocol is **correct and semantics-preserving**, but the current
AutoDL↔Alibaba network path throttles TDX→H800 to ~0.1 MB/s, so long runs on it would measure
infrastructure, not method. Nothing committed. Do not resume 10-/50-step, utility, 7B, or performance
runs on this environment.

## 1. GPU execution conclusion
Protected **forward runs on CUDA**; protected **backward runs on CUDA**. Device inventory:
**606/607 tensors on CUDA**, 0 disallowed on CPU; the single CPU tensor is the explicit q/k/v/gate/up
A-grad **TDX staging buffer**. No disallowed CPU matmul, **no hidden NumPy path** (0 `.item()`, 0
`.numpy()`). CUDA kernels confirmed (cuBLAS `nvjet_*` GEMMs, `bmm` attention, elementwise). GPU kernel
time ≈ **13.4 ms**; CPU dispatch ≈ **104 ms** over **11,126** small kernels → the short compute window
is **launch-overhead-bound** (expected for 0.5 B @ seq 42). **Distinguish:** GPU utilization *during
the compute window* ≈ 23 % avg / 85.7 % active-duty, VRAM 2143 MB resident, 86 W; GPU utilization *over
the full network-blocked step* ≈ **0.5 %** — the wall is network-dominated, not a compute or
device-placement bug.

## 2. Transport decomposition
| quantity | value |
|---|---|
| serialize_ms | ≈ 3.6 |
| deserialize_ms | ≈ 0.6 |
| hmac_ms | ≈ 4.0 |
| total_non_network_ms | ≈ 8.3 |
| dlogits_payload_mb | ≈ 12.76 (bf16, after fix; was 25.5 fp32) |
| network_wait_ms | ≈ 123,000 |
| measured TDX→H800 throughput | ≈ 0.10 MB/s |
| non_network_fraction | ≈ 0.007 % |

`bottleneck_class = infrastructure_ingress_throttle`; `code_path_bottleneck = false`,
`framing_bottleneck = false`, `serialization_bottleneck = false`, `hmac_bottleneck = false`,
`gpu_compute_bottleneck = false`, `tdx_compute_bottleneck = false`. Mechanism:
**consistent_with an AutoDL container ingress throttle** (not asserted as provider policy).

## 3. Measured directional throughput
- Download TDX→H800: **~0.104–0.109 MB/s** (two real S5 points, 12.76 MB and 3.48 MB).
- Upload H800→TDX: **~12.7 MB/s** (prior reachability audit). Strong **asymmetry** ⇒ ingress rate cap.

## 4. Is throughput flat by payload size?
**Yes.** 3.48 MB @ 0.109 MB/s and 12.76 MB @ 0.104 MB/s → ~constant ~0.10–0.11 MB/s independent of
size ⇒ a rate cap, not a window/MTU/per-frame effect. The synthetic 1/8/16/32 MB sweep was
**STOPPED_EARLY_SUFFICIENT_EVIDENCE** (moving 114 MB over a 0.1 MB/s link is unnecessary to show
flatness).

## 5. bf16 dlogits regression result
The enclave now returns dlogits in the run dtype (bf16) instead of fp32 — halving the throttled
download (25.5→12.76 MB). **Semantics-preserving** (the client already cast received dlogits to bf16
before backward). Verified bit-identical to the ferried reference: CE, top1, next-logit KL, equiv
max_abs all exactly equal.

## 6. One-step direct semantic comparison (`DIRECT_TRANSPORT_SEMANTIC_REGRESSION_ONE_STEP`)
Direct persistent-transport L10 BF16 one step vs Mac-ferried + plaintext reference — **CONSISTENT**:
- CE **0.45357903838157654** = ferried (Δ = 0); plaintext HF+effective-LoRA CE 0.36957.
- top1 **0.976190** = ferried; next-logit KL **0.0081466** = ferried; equiv max_abs **4.00750** = ferried.
- gradA 0.0589, gradB 15.08, corrected-grad 0.0463, ΔW-proxy 0.01508, momentum 0; finite = true.
- correction: 48 GPU-exact A + 120 in-enclave A, 0 missing.
- security counters all **0** (auth/replay/malformed/forbidden/untrusted-γ/missing); replay/HMAC
  fail-closed verified in `test_direct_transport_faults.py`.
- attestation **verified** (SUCCESS, reportdata-bound, debug=false); single channel, 3 requests, seqs
  monotone, attestation once.

## 7. Benchmark-key removal
Temporary scoped bench key `h800-bench-tdx` (fingerprint `SHA256:KNpHHHT/UvuKI+6hxKVGwOcDNDWhO8aOyaZa8vgyhNk`)
**removed** from TDX `authorized_keys` (before 1 → after 0); H800 key files deleted; the production
forced key is intact and still blocks arbitrary commands. No key material in logs or artifacts (bench
used an all-zero non-secret HMAC key; ephemeral per-run session keys redacted from scratch configs).
Only the non-secret fingerprint is recorded.

## 8. Deferred experiment list — `DEFERRED_NETWORK_ENVIRONMENT`
50-step multi-seed matrix; L12 real AdamW long run; converged GSM8K; converged SST-2; 7B long runs.
Reason: measured ~0.1 MB/s TDX→H800 makes these measure infrastructure throttling rather than method
behavior. **Not failed, not complete** — deferred to a migrated environment.

## 9. Migration package readiness
Portable stack + hashes + preflight documented in `migration_readiness.md`.
`scripts/migration_preflight.py` gates long runs on: real GPU, real TDX quote (DEBUG=false), package/
code hash match, persistent direct channel, no Mac ferry, and **sustained ≥ 20 MB/s both directions**.
Current AutoDL env **fails gate 6**; long runs refused until migration.

## 10. Files changed (uncommitted)
- **New:** `gpu_execution_audit.py`, `gpu_audit_sample.sh`, `build_step_timeline.py`,
  `test_direct_transport_faults.py`, `compare_direct_vs_ferried.py`, `tdx_bench_service.py`,
  `transport_bench.py`, `migration_preflight.py`.
- **Modified:** `tdx_persistent_service.py` (bf16 dlogits; HMAC-key commitment), `h800_direct_runner.py`
  (on-GPU norms; VRAM telemetry; timeout/bound/abort; READ_TIMEOUT 600), `gate0_direct_orchestrator.py`
  (`head.txt`; HMAC-key commitment; equiv verifier call).
- **Artifacts:** `results/aaai_private_base/gpu_execution_audit/*`, `results/aaai_private_base/direct_transport/*`.

## 11. Tests
Fault-injection **all pass**; device inventory **PASS**; bf16-preservation **CONSISTENT**; one-step
direct semantic regression **CONSISTENT**; overhead decomposition measured; benchmark stopped early
with sufficient evidence; bench-key removal verified.

## 12. git status
Branch `fix/folded-remote-repeat-debug-and-guards`; HEAD unchanged. Working tree has the above
uncommitted modified scripts + untracked audit artifacts. **No git commit was run this session.**

## 13. Commit status
**Nothing committed.**
