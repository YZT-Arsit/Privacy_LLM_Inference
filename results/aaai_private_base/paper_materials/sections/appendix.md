# Appendix (draft)

All settings below are extracted from frozen artifacts; nothing is re-run.

## A. Optimizer protocol (masked LoRA, trusted AdamW)
- **Arms.** L0 (no LoRA), L5 (plaintext-equivalent fp32), L12 (protected bf16, trusted AdamW).
  L5/L12 use the identical enclave/GPU optimizer split; only the execution boundary differs.
- **Trusted split.** The FP32 optimizer master + first/second moments (m, v) live inside the
  TDX enclave. Trusted factors (A of {q,k,v,gate,up} and B of {q,k}) are updated in-enclave; the
  GPU-exact monomial factors (A of {o,down} and B of {o,down,v,gate,up}) are updated on the GPU
  in the masked basis. Each optimizer step: the GPU sends packed masked gradients; the enclave
  un-folds, applies AdamW on the FP32 master, and re-folds — the GPU never sees plaintext m/v.
- **Exactness.** SGD, momentum, and AdamW reproduce the plaintext update at fp64 (top-1 = 1.0,
  all trust-domain counters 0; `full_lora_matrix/**`, `gate0_d4/**`). This is model-size
  independent (a per-factor algebraic identity).
- **Frozen recipe.** effective batch 16, grad-accum 1, AdamW, best-dev checkpoint, early stopping
  (patience 3), seeds {1234, 2025, 7}, 0.01 accuracy equivalence margin.

## B. TDX deployment
- **Hardware.** Real A10 GPU (untrusted) + real Intel TDX guest (trusted), private VPC. The GPU
  package contains only `*_tilde` artifacts (package root pinned; plaintext-absence scan clean).
- **Attestation.** TDX quote verified (mr_td / runtime-hash binding); the runtime hash binds the
  design so a code change forces re-attestation.
- **Data plane.** Per optimizer step = **2** trusted round trips (one `ce_batch` returning
  authenticated dlogits + one `adamw_step`). The A10 holds no plaintext labels/loss/dlogits;
  the enclave holds the private label table + a monotonic ledger. Each batch descriptor is
  HMAC-bound (run_id / dataset / split / step / sample_ids / shapes); stale/replayed/out-of-order
  batches are rejected (fail-closed).
- **Throughput.** Cross-machine data plane ≈170 MB/s on the private VPC.

## C. Attack settings (S1–S6) — `attack_budget_registry.json`
CPU-only, offline; attacker functions receive only transformed tensors + architecture.
- **S1.** A1 = closed-form ridge (1e-3) least squares; A2 = 1×512-GELU MLP (Adam 1e-2, 400 steps);
  A3 = 2×512-GELU MLP. Strict + known-pairs settings; synthetic n=4000/d=896 invertibility probe.
- **S2.** A1 = one truncated SVD (rank 16) of the masked product; A2 = Adam 5e-2 × 600 steps;
  A3 = functional-reproduction accounting. Adapter 896→896, rank 16.
- **S3.** one real forward; closed-form multiset / sorted-prob-KL / confidence-correlation; 64
  rows over the full 151936 vocab; monomial condition number ~7.
- **S4.** DLG/iDLG, LBFGS (lr 0.5, max_iter 20), 600 outer steps, 3 restarts, iDLG analytic
  label; masked model = exact Nr-conjugate; batch sweep {1,2,4,8}.
- **S5.** Shokri shadow-model MIA, 8 shadow models, member/non-member 120/120, LoRA-style linear
  probe over frozen features (150 epochs); channels B0/B1-perm/B1-monomial.
- **S6.** S1 linear + 1×512-MLP decoders on layer-0 KV (K_rope·Bk, V·Sv), closed-set token
  classification; plaintext / identity / masked-transfer / masked-adapted.
- **Controls / baselines.** every attack succeeds on its positive control first; random baseline
  reported (token chance = 1/V; AUC 0.5).

## D. Ablation settings (A0–A5) — `ablation_results.json`
CPU harness, real Qwen2.5-0.5B; security columns reuse S1–S6. Correctness = max fold err /
logits-KL; the A2 operator test compares SiLU through a sign-free permutation (err 0) vs a
signed/general mask (err 4.16); the A5 KPA test measures least-squares mask recovery under a
static mask (rel-err 1e-7) vs a refreshed mask (0.82). Seeds {0, 5, 11}.

## E. Efficiency accounting — `PHASE_profiling_gate.json`, `obfuscatune_latency_h800.json`
- **Inference (algorithmic).** GPU-TEE prefill slowdown ≤1.07× at **2** TEE boundary crossings
  (single trusted session, zero nonlinear crossings) vs 224 for ObfuscaTune; on a CPU-only TEE
  the crossing count dominates (ours 2.4–3.0× vs 82–114×).
- **Training (real A10 + TDX, batch 16).** 0.459 s/batch; per-optimizer-step decomposition —
  TDX AdamW 113.6 ms (fixed, CPU-only enclave), CE network 44.5 ms, GPU forward/backward 46.3/82.5
  ms; 2 trusted calls/step; 1.14 MB/example each way.
- **Transport vs algorithmic.** The two-machine prototype's per-token decode overhead (~9×,
  ~71 ms tunnel RTT + fp32-logit payload) is a network-path artifact, removed by a co-located
  GPU-TEE (the ≤1.07× row); we report the algorithmic overhead as the method's cost.

## F. Downstream-generation settings (GSM8K) — `gsm8k_lora_eval.json`
Qwen2.5-0.5B, MPS, minimal 7-target LoRA (rank 16), plaintext SFT (bs 2, lr 8e-5, grad-clip 1.0,
250 steps), greedy decoding (max 200 new tokens), n=40 official test problems. Answer extraction:
number after `####` else last integer; EM vs official gold. M2 reported by equivalence (training
equivalence + fp32 folded-generation parity), not re-run.
