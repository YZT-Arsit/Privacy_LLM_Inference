# Limitation statement (private-base AAAI submission)

We state the following limitations plainly. Throughout, security is claimed only as
**empirical resistance under the evaluated threat model and published attacks** — we make
**no** claim of zero leakage, impossible recovery, or information-theoretic security.

## Threat-model scope
- The evaluated attacker is honest-but-curious with the transformed GPU package and the
  protocol-observable tensors; it does not hold plaintext base weights, the original
  checkpoint, masks/inverses, TDX secrets, or trusted optimizer state. Results do not
  extend to a compromised TEE, a malicious trusted party, or side-channel attackers.
- The base model is treated as a **from-scratch private base**. Our security harness uses
  the real Qwen2.5-0.5B weights as the *secret* base for realistic representations but
  never exposes them to any attack function; the leakage geometry under test is
  weight-distribution invariant. We do not claim security for a base whose weights are
  publicly downloadable.

## Structural leakage we explicitly do NOT claim away
- **Orthogonal / permutation masks preserve norms, pairwise Grams, and value multisets**
  (S1, S3), and are **linearly invertible given paired plaintext** (S1 synthetic rel_err
  5.7e-7). Confidentiality of hidden states / weights therefore rests on the TEE
  preventing paired-plaintext exposure, not on the algebraic mask.
- **Gradient inversion** is a transparent orthogonal change of basis (S4, basis-invariance
  loss gap 9e-13); the mask adds no protection. Defense is batch aggregation + never
  exposing per-example gradients.
- **Attention scores** `Q_rope K_rope^T` are exposed on the untrusted GPU by the
  paired-feature construction (design_spec §G); the attention-fingerprint leak is not
  claimed hidden (partial mitigation = layer-0 TEE relocation only).
- **Input tokens at the embedding boundary** are recoverable from the (masked) embedding
  table for any deterministic embedding (S1 NN acc 1.0); input-token protection is a
  separate input-pad / layer-0-TEE mechanism, not the residual mask.
- **Permutation-only logit masking** leaks the full confidence/uncertainty profile (S3);
  only the proposed monomial variant perturbs it, at a bounded condition number (numerical
  range trade-off), and neither hides token identity better than the shared permutation.

## Experimental-scope limitations
- Utility is demonstrated on **SST-2** with 3 seeds; the 95% CI on the L12−L5 gap is wide,
  so we claim equivalence-in-mean, not strict statistical equivalence. GSM8K converged
  utility is compute-bound and deferred (measured L12 ≈ 74.6 h/seed).
- Security attacks are evaluated on **Qwen2.5-0.5B** (not 7B) on CPU; S5 targets a
  LoRA-style linear probe over frozen features (output-confidence channel), and S6 uses
  layer-0 KV with closed-set token classification — both are attacker-favourable
  simplifications chosen to make the positive control strong and the contrast clean.
- We do not reproduce external baselines (STIP / ObfuscaTune / Amulet) in this round; the
  comparison is to plaintext / identity / random-mask internal baselines only.

## System-overhead limitations
- The dominant protected-training cost is a fixed per-optimizer-step TDX round trip
  (~113 ms AdamW on the CPU-only TDX guest + ~44 ms CE network); it is inherent to
  effective-batch-16 protected training, not implementation overhead, and would benefit
  from a GPU/accelerated enclave.

## Honest bottom line
Under the evaluated threat model and published attacks, the private-base transformed model
**empirically resists** recovery of private representations, adapters, gradients, and
inference artifacts *when the TEE assumptions hold* (no paired-plaintext / per-example-
gradient exposure, aggregated updates). Its assurances are empirical and assumption-scoped,
and the algebraic masking alone is explicitly **not** the source of confidentiality.
