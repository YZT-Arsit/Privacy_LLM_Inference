# CPU-Only Protocol Audit — Summary

git `5208ae3b56` · python 3.13.12 · torch 2.10.0 · numpy 2.4.2 · dtype float64 · seed 0

**No CUDA · no real GPU · no real TEE.** `simulated_trusted_controller=True`, `uses_real_gpu=false`, `uses_real_tee=false`. fp64 exactness does NOT imply bf16/GPU exactness; op/byte counts are NOT measured latency; synthetic attacks are NOT real-model attacks.

**1. CPU fp64: linear/attention/KV/nonlinear/LoRA forward exact?**
YES — A1 masked linear (perm/orth/diag/dense_gl, square+rect, ±bias) max err ~1e-15; A2 attention score/prob/context invariant; A3 KV prefill+decode == full recompute; A4 stabilizer classifies GELU/SiLU(perm-only)/ReLU(+diag)/tanh(+signed); A5/A6 MLP+SwiGLU exact.

**2. LoRA independent-mask backward == plaintext autograd?**
YES — single-layer masked-vs-plaintext param error 1.3e-15 (A1); multi-layer allclose=True.

**3. Standard autograd enough for GELU/SiLU/SwiGLU backward?**
YES — A5 autograd vs manual-derivative both exact (YES); A6 SwiGLU backward gH exact under shared Pi (YES); no custom nonlinear primitive.

**4. Multi-step masked AdamW trajectory == plaintext?**
YES (trusted-side AdamW) — single-layer loss-curve distance 9.7e-16; multi-layer max_loss_diff 1.3e-12. Dense-domain masked AdamW is correctly REFUSED (status=raised_unsupported); optimizer runs trusted-side on recovered grads.

**5. Packed update changes only scheduling (not params/optimizer state)?**
YES — packed == per-layer: param_A err 0.0e+00, delta_W err 0.0e+00, Adam 2nd-moment err 0.0e+00, next-step logit err 0.0e+00.

**6. Trusted invocations per step?**
**3** (packed schedule): 1 input-mask + 1 loss/logits + 1 packed update. worker->trusted returns = 2/step after the initial masked input; nonlinear per-layer trusted calls = 0.

**7. Why 3, not 2?**
The packed gradient/update is a distinct trusted invocation from the loss/logits invocation: the worker must first return masked logits (invocation 2, trusted computes loss + injects the logit gradient), then return the packed masked adapter gradients (invocation 3, trusted unmasks + AdamW + remasks). Fusing them would require the worker to compute the loss, which is trusted-only.

**8. Is the invocation count independent of #LoRA layers?**
YES — packed stays 3/step at 1..32 LoRA layers (observed_match=True). The per-layer baseline grows: at 32 layers it is 34/step (2 + |S|).

**9. Packed-update communication bytes?**
packed grads 2,883,584 B + updated adapters 2,883,584 B (adapter-scale, O(sum_j r_j(d_in+d_out))); per-step total 24,248,584 B (189,442.06 B/token) at mid/float32.

**10. Logits boundary vs adapter update — which dominates communication?**
MEASURED: **logits** dominates at the mid config (logits O(mV)=16,384,000 B vs adapter 5,767,168 B). Reported, not presupposed — flips only if V is small relative to sum_j r_j(d_in+d_out).

**11. Trusted-side compute complexity per mask family?**
permutation/diagonal unmask = O(adapter_elems) (O(adapter_elems)); orthogonal/dense_gl unmask = O(d^2 r) (O(d^2 r)) — NOT O(adapter_elems). AdamW is O(adapter_elems) elementwise regardless.

**12. Dense-GL condition number vs numerical error (CPU fp64)?**
fp64 tolerant up to cond ~10000.0: dense_gl forward error 1.2e-09 at cond 1e4. CPU-float64 observation only — NOT a bf16/GPU promise.

**13. Which leakage is algebraically certain?**
cross-Gram in the nonlinear permutation region (corr 1.000, exact); the per-row value multiset (invariant, err 0e+00); masked-gradient rank; compensation Gram statistic. These are deterministic, not attacker-dependent.

**14. Which attack results are synthetic-only?**
ALL of Section F (F1-F6): cross-Gram, value-multiset linkability, subspace trajectory, compensation second-moment, one-vs-multi-view, packed-buffer metadata. Algebraic/synthetic red-team; not real-model attacks.

**15. Which conclusions must wait for real GPU?**
bf16/fp16 end-to-end correctness, real Qwen/Llama activations & LoRA quality, real token/adapter extraction, GPU throughput/VRAM/kernel overhead, real tokens/s and step latency. See claims.md (B).

**16. Which conclusions must wait for real TEE?**
enclave crossing latency, trusted AdamW wall-clock, enclave memory/paging, attestation overhead, host-TEE copy cost, end-to-end TEE slowdown, side-channels. See claims.md (C).

## Output files

`results.json`, `correctness.csv`, `multistep_training.csv`, `boundary_calls.csv`, `communication_bytes.csv`, `trusted_compute_estimates.csv`, `mask_conditioning.csv`, `synthetic_attacks.csv`, `claims.md`.
