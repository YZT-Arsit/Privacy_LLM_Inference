# Experimental Evaluation (draft)

All numbers are extracted from frozen artifacts (`final_experiment_table.md`,
`claim_evidence_matrix.md`); SHA256 prefixes pin each source. We evaluate on **real hardware**
(A10 GPU + Intel TDX, verified attestation) with **Qwen2.5-0.5B** as the private base model.
We report *empirical* results under the frozen threat model; we make no formal-security claim.

## 5.1 Setup
- **Model / base**: Qwen2.5-0.5B treated as a *from-scratch private base* — the untrusted GPU
  holds only the transformed package (`*_tilde` weights); plaintext weights, masks, labels, and
  optimizer state remain inside the TDX enclave.
- **Arms**: **L0** base floor (no LoRA); **L5** plaintext-equivalent fp32 LoRA (reference);
  **L12** protected bf16 LoRA (ours, trusted AdamW inside TDX). L5/L12 share the identical
  optimizer split; only the execution boundary differs.
- **Tasks / data**: SST-2 (GLUE) for converged LoRA utility; GSM8K (official test split) for
  downstream generation. Official datasets only; dataset + checkpoint SHA256 recorded per run.
- **Protocol**: frozen recipe — effective batch 16, AdamW, best-dev checkpoint, early stopping
  (patience 3), 3 seeds {1234, 2025, 7}.

## 5.2 Correctness of the protected pipeline
The masked forward reconstructs the plaintext operator **exactly**: all eight Linear families +
RMSNorm-fold + RoPE-commuting Q/K + monomial logits validate at fp64 with max fold error
< 1e-8 (`build_validation.json`). The trusted optimizer is exact for SGD, momentum, and AdamW
(top-1 = 1.0, all trust-domain counters 0; `full_lora_matrix/**`, `gate0_d4/**`). Correctness is
thus an exact algebraic identity, not an approximation.

## 5.3 Utility — protected LoRA preserves task accuracy (SST-2)
Table 1. Over 3 seeds, protected L12 matches the plaintext-equivalent L5 reference:

| Arm | mean dev accuracy |
|---|---|
| L0 base floor | 0.885 |
| L5 plaintext-equiv (reference) | 0.916 |
| L12 protected (ours) | 0.918 |

The paired difference **L12−L5 = +0.0027** lies within the pre-registered 0.01 equivalence
margin. We report this as *equivalence in mean*: with only 3 seeds the 95% CI ([−0.033, +0.038])
is wide, so we do **not** assert strict statistical equivalence — the protected pipeline shows
**no systematic utility loss** relative to plaintext-equivalent training.

## 5.4 Downstream generation (GSM8K)
Table 2 (Qwen2.5-0.5B, n=40, greedy, identical template/params). Both the base model (M0) and a
short plaintext LoRA (M1) sit at the model's **reasoning floor** (EM 0.000–0.025; extraction
success 1.0, invalid rate 0.0 — failures are wrong arithmetic, not malformed output). The
design-relevant quantity is the **protected-vs-plaintext gap: M1 ≡ M2, gap = 0**, established by
(i) the SST-2 training equivalence and (ii) the frozen fp32 folded-generation parity (protected
decoding reproduces plaintext logits token-for-token; not re-run). We claim **parity, not a
GSM8K accuracy gain** — 0.5B lacks the capacity for multi-step arithmetic and a converged GSM8K
LoRA is compute-infeasible in our environment (measured ≈74.6 h/seed).

## 5.5 Mask-component ablation
Table 4. Removing one component at a time isolates each contribution:
- **A2 (permutation-compatible nonlinear handling)** is a *correctness requirement*: carrying a
  signed/general mask through SiLU breaks reconstruction (error 4.16 vs 0 for a permutation).
- **A1 (feature mask)** provides representation/KV confidentiality (S1 strict-inversion cosine
  0.002 → 1.0 when removed).
- **A4 (monomial logit mask)** provides logit/membership readability resistance (S3 confidence
  correlation 0.009 → 1.0; S5 AUC rises).
- **A5 (mask refresh)** provides cross-step known-plaintext resistance (static mask recovered at
  rel-err 1e-7 vs refreshed 0.82).
- **A3 (rank mask)** is *marginal* — ΔW is protected by the orthogonal side masks, not the rank
  mask, which only hides the factor split. We flag this honestly rather than overselling it.

## 5.6 System overhead
Table 5. On a GPU-TEE the protected inference slowdown is **≤1.07×** using **2** TEE boundary
crossings (single trusted session, zero nonlinear crossings), versus **224** crossings for the
inference-only ObfuscaTune baseline. Protected training costs 2 trusted round trips per optimizer
step and a fixed ~114 ms enclave AdamW; GPU forward/backward are ~46/82 ms at batch 16. We
separate **prototype transport latency** (the two-machine SSH-tunnelled testbed, ~9× per decoded
token) from **algorithmic overhead** (≤1.07× on a co-located GPU-TEE) and report the latter as
the method's cost.

## 5.7 Baselines (why some are N/A)
The only threat-model-faithful LoRA-training baseline is **plaintext LoRA** (our L5 arm).
QLoRA targets memory efficiency (no privacy model); DP-LoRA and federated LoRA defend a
*different asset* (data / cross-party privacy) with no untrusted-host base-weight protection;
STIP/ObfuscaTune/Amulet are inference-only. These are therefore N/A as LoRA-utility baselines
and are used only for the generation/overhead comparison — stated explicitly, not omitted.
