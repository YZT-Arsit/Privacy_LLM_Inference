# Hot-path anti-pattern audit (H800 protected worker)

Hot path = `h800_direct_runner.py` (persistent runner) + `MaskedQwen.forward` in
`h800_unified_worker.py` + the apply/correct blocks. The Mac-ferried `h800_d4_worker.py`
per-step CLI modes are the OLD path (abandoned for the resident direct runner) and are flagged
where relevant. Verdict per item:

| # | Anti-pattern | Present in hot path? | Evidence / notes |
|---|---|---|---|
| 1 | AutoModel / plaintext HF CPU path | **No** | `AutoModelForCausalLM` appears only in `h800_unified_worker.py` L336–362 = the `--mode init/forward` diagnostic **TrustedVerifier** (effective-equivalence eval), NOT the protected forward and NOT imported by `h800_direct_runner.py`. |
| 2 | `torch.load` without `map_location="cuda"` | **Minor** | Base weights load `map_location="cpu"` then `.to(cuda)` **once at startup** (not per step). `load_tensor` deserializes wire buffers (dlogits/corrected) to CPU then `.to(device)` — a tiny transport buffer; candidate to deserialize straight to device. |
| 3 | per-layer `.item()` in hot path | **No** | 0 `.item()` calls measured in the instrumented step (S2). |
| 4 | per-target `.cpu()` copies | **Yes (expected + 1 self-inflicted)** | (a) 120 `gA…cpu()` = the legitimate q/k/v/gate/up A-grad **staging to TDX** (5×24); one DtoH each → candidate to pack into 1 buffer. (b) my req-1 norm telemetry `_n2` did ~350 per-target `.cpu()`/step — **fixed** to accumulate on-GPU + one sync (S7). |
| 5 | Python loops over heads/targets/layers | **Inherent** | `MaskedQwen.forward` loops 24 layers × 7 projections in eager mode → the launch-overhead source (S3). No per-head Python loop (attention is batched `bmm`). |
| 6 | CPU mask construction every step / explicit inverse in hot path | **No** | `MaskedQwen.forward` contains **no** mask build, **no** `inverse`/`linalg.solve`, **no** `.cpu()`/numpy. Masks are folded into weights at package-build time; `orthogonal_signed_perm`/`permutation_matrix` run only in `TrustedVerifier.__init__` (one-time, eval-only). gram_inv is rebuilt **in TDX**, never on H800. |
| 7 | `CUDA_LAUNCH_BLOCKING` / deterministic / anomaly / retained graph across steps | **No** | None set. `torch.autograd.grad` retains only the current step's graph; S1 VRAM shows `graph_freed=True`, no cross-step growth. |
| 8 | hooks / provenance hashing in hot loop | **No** | No `register_hook`/`register_forward_hook`. `compute_root_hash(PKG)` runs **once at startup**. HMAC-SHA256 runs per message (2/step) over the payload — a necessary, cheap (~ms) security cost, not a GPU op. |

## Bottom line
The protected forward/backward are **on CUDA with no correctness anti-patterns**: no CPU fallback,
no per-step model reload (resident), no mask rebuild, no matrix inverse, no debug modes, no retained
graph, no per-step provenance hashing. The only real inefficiencies are (a) launch-overhead from
eager per-layer dispatch on a tiny model, (b) 120 per-target grad stagings to TDX, and (c) a
self-inflicted telemetry sync that is now fixed. **None of these is the dominant cost — network + TDX
transport is** (see S5 timeline). The Mac-ferried `h800_d4_worker.py` per-step `torch.load` reload
(model reloaded every CLI invocation) was a genuine anti-pattern of the OLD path; the resident direct
runner eliminates it.
