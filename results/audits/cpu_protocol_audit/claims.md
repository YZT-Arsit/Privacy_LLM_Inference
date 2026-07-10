# CPU-Only Audit — Tiered Claims

Strict three-tier classification. Group A is what THIS CPU-only stage supports; B waits for real GPU; C waits for real TEE.

## A. Supported by CPU experiments (this stage)

- Algebraic implementation correctness in fp64 (and fp32 numerical-sensitivity contrast) for masked linear, attention score/KV invariants, GELU/SiLU/ReLU stabilizer classification, GELU/SiLU MLP and SwiGLU forward+backward, LayerNorm/RMSNorm permutation equivariance.
- LoRA independent-mask backward recovers plaintext gradients; multi-step masked training matches the plaintext trajectory when the optimizer runs trusted-side on recovered gradients.
- Exact packed-vs-per-layer scheduling equivalence (params, DeltaW, Adam moments, next-step logits identical).
- Protocol-level O(1) = 3 trusted invocations/step, independent of the number of LoRA layers (observed via a simulated schedule counter).
- Exact byte and operation ACCOUNTING (communication bytes per channel; trusted compute op-counts per mask family; peak-tensor byte estimate).
- Deterministic cross-Gram / value-multiset leakage boundaries (nonlinear region exact; general region not).
- Mask-conditioning error TRENDS on CPU float64.
- Synthetic attack signals (F1-F6) as algebraic red-team evidence.
- Dense-domain masked AdamW is unsupported (correctly refused) — the optimizer must stay trusted-side.

## B. Must wait for real GPU

- bf16/fp16 end-to-end correctness (fp64 exactness does NOT extrapolate).
- Real Qwen/Llama activations and real LoRA fine-tuning quality.
- Real token / adapter extraction attacks on real model activations.
- GPU throughput, VRAM, kernel overhead.
- Real generation tokens/s and real training step latency.

## C. Must wait for real TEE

- Enclave crossing latency and trusted AdamW wall-clock.
- Enclave memory / paging behavior.
- Attestation overhead and real host<->TEE copy cost.
- Real end-to-end TEE slowdown.
- Any side-channel (cache/power/EM/transient) claim.

## Forbidden extrapolations (explicitly NOT claimed)

- CPU trusted function is a real TEE.
- operation count is measured TEE latency.
- synthetic attack is a real LLM attack.
- fp64 exactness implies bf16 exactness.
- nonlinear leakage is harmless.
- non-orthogonal masks only leak rank (they remove EXACT spectral preservation but the observable remains a masked Gram statistic).
- one view is formally safer than multiple views (F5 is exploratory).
- training and serving must mathematically share one domain.
- formal privacy.