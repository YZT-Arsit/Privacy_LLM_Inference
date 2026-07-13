# AAAI final experiment gap report (private-base LoRA)

Scope of this round: close reviewer-visible experimental gaps only — no method redesign, no
threat-model change, no new claims. Security wording stays scoped (empirical resistance under
the evaluated threat model; never zero-leakage / impossible / information-theoretic).

## 1. Completed experiments (this round + reused frozen)
| Phase | Experiment | Status | Artifact |
|---|---|---|---|
| P1 | GSM8K downstream generation after LoRA (M0 base, M1 plaintext LoRA; M2 by equivalence) | ✅ (0.5B) | `lora_generation_gsm8k/gsm8k_lora_eval.json` |
| P2 | Qwen2.5-7B LoRA scale | ⏸ deferred (env-limited) — scoped doc + frozen-7B reuse | `qwen7b_lora_scale/scale_validation.md` |
| P3 | LoRA baseline review + decision | ✅ | `lora_baselines/baseline_review.md`, `baseline_decision.md` |
| P4 | Mask-component ablation A0–A5 | ✅ | `ablation/ablation_results.json`, `ablation.md` |
| P5 | Efficiency / system overhead comparison | ✅ | `system_comparison/system_comparison.md` |
| — | SST-2 LoRA utility (L0/L5/L12, 3 seeds) | ✅ frozen | `utility_dataplane/PHASE7_SST2_UTILITY_SUMMARY.json` |
| — | Security S1–S6 (all controls pass) | ✅ frozen | `security/**` |
| — | 7B generation-only + attacks | ✅ frozen | `baselines/conjformer/qwen7b_*`, `attacks/*qwen7b*` |

### Headline results (real numbers)
- **Utility (SST-2)**: protected L12 vs plaintext-eq L5 mean gap **+0.0027** (within 0.01 margin).
- **GSM8K generation (0.5B, n=40)**: M0 base EM **0.025** (1/40), M1 plaintext LoRA EM **0.000** (0/40) —
  both at 0.5B's reasoning floor (within subset noise; extraction success 1.0, invalid 0.0); **M1 vs M2 gap = 0**
  (M2 = M1 by protected-training equivalence + frozen fp32 generation parity). We do NOT claim a GSM8K gain or
  LoRA superiority. See `lora_generation_gsm8k/{gsm8k_lora_eval.json, FINDINGS.md}`.
- **Ablation**: A2 (nonlinear-perm) is the correctness-critical component (SiLU-through-signed-mask err **4.16**
  vs permutation err **0**); A1 feature mask owns S1/S6 (strict inversion cos 0.002→**1.0** when removed);
  A4 monomial owns S3/S5 (conf-corr 0.009→**1.0**); A5 refresh owns cross-step KPA (static rel_err **1e-7**);
  A3 rank mask is **marginal** (factor-split only).
- **Overhead**: ours **2** TEE boundary crossings, **≤1.07×** GPU-TEE inference slowdown, vs ObfuscaTune **224**
  crossings; protected training = 2 trusted calls/opt-step, fixed ~114 ms enclave AdamW.

## 2. Remaining limitations (honest)
- **GSM8K absolute EM at 0.5B is capability-limited** (small model + short SFT); the design-relevant
  quantity is the M1-vs-M2 gap (~0), not the absolute score. Converged GSM8K LoRA is compute-infeasible
  here (measured L12 ≈ 74.6 h/seed).
- **7B protected-LoRA training not run** — no 7B checkpoint is reachable in this environment; scalability
  of the *transform* is supported by frozen 7B generation/security artifacts + size-independent 0.5B
  optimizer-exactness, plus a labeled cost projection (not a measurement).
- **Security is empirical + assumption-scoped** (S1–S6): orthogonal/permutation masks preserve norms/Grams/
  multisets and are invertible given paired plaintext; confidentiality rests on the TEE preventing
  paired-plaintext / per-example-gradient exposure + aggregation, not the algebraic mask alone.
- **Utility breadth**: one classification task (SST-2, 3 seeds, wide CI) + one generation task (GSM8K, 0.5B).
- **External privacy-LoRA baselines are N/A by threat model** (DP-LoRA/federated defend a different asset;
  inference-only STIP/ObfuscaTune/Amulet have no LoRA training) — used for generation/overhead only.

## 3. Paper claims SUPPORTED by evidence
1. Exact operator correctness of the masked forward (fp64 < 1e-8) and TDX-trusted optimizer (SGD/momentum/
   AdamW, top-1 = 1.0). [frozen]
2. **Privacy-preserving LoRA maintains downstream utility**: SST-2 L12≈L5 (+0.0027) + GSM8K M1≡M2 (gap 0 by
   equivalence). [P1 + frozen]
3. Each mask component's **contribution** is isolated (ablation A0–A5); the design is not over-parameterized
   (A3 rank mask marginal, honestly flagged). [P4]
4. **Efficiency**: single trusted session, 2 TEE crossings, ≤1.07× GPU-TEE inference overhead vs 224
   crossings for ObfuscaTune. [P5]
5. **Empirical security resistance** under the evaluated attacks S1–S6 (all positive controls pass), with
   explicit documented leaks. [frozen]
6. Real-hardware realization (A10 GPU + Intel TDX, verified attestation). [frozen]

## 4. Claims NOT (yet) supported / to avoid
- ❌ Any formal / information-theoretic / zero-leakage security claim (never made; masks are exact linear
   operators invertible under paired plaintext).
- ❌ Superiority of protected LoRA over plaintext LoRA in task accuracy (we claim *parity*, not superiority).
- ❌ Empirical 7B protected-LoRA fine-tuning results (deferred; environment-limited).
- ❌ Converged GSM8K reasoning gains at 0.5B (below capability floor; not claimed).
- ❌ Attention-score confidentiality (explicitly documented as leaked; partial layer-0 mitigation only).

## 5. Recommended final paper scope
- **Method**: private-base unified transformed package (exact fold) for generation + LoRA training behind a
  TDX boundary; single trusted session, permutation-compatible nonlinear handling, monomial logit mask,
  refreshed orthogonal feature masks + rank mask.
- **Utility**: SST-2 (converged, 3 seeds) as the primary equivalence result; GSM8K-0.5B generation as a
  downstream-generation datapoint with the M1≡M2 parity argument; state the 0.5B/short-SFT capability caveat.
- **Security**: S1–S6 as an empirical, scoped evaluation with all positive controls and documented leaks;
  the ablation as the "which component matters" evidence. Never overclaim.
- **Efficiency**: the 2-vs-224 TEE-crossing algorithmic comparison; clearly separate prototype transport
  latency from algorithmic overhead.
- **Scale**: present 7B as generation/fold + security validation (frozen) + a scalability argument and cost
  projection; list full 7B protected LoRA training as future work.
- **Positioning**: private-base weight confidentiality against an untrusted execution host with a TEE — a
  threat model no existing LoRA-privacy baseline shares (justify N/A baselines explicitly).

_Nothing in this round was committed by the assistant._
