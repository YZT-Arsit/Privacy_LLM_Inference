# Security Evaluation (draft)

**Scope of the claim (never exceeded):** *under the evaluated threat model and published attack
methodologies, the private-base transformed model empirically resists recovery of private
weights, representations, LoRA adapters, gradients, and inference artifacts.* We make **no**
claim of impossible recovery, zero leakage, or information-theoretic security. Every attack
first succeeds on a positive control, then is run against the protected system; we report a
random baseline and the documented structural leaks.

## 6.1 Threat model (frozen)
Honest-but-curious attacker with the transformed GPU package and protocol-observable tensors;
it knows the architecture and public hyperparameters. It does **not** know the plaintext base
weights, the original checkpoint, the masks/inverses, TDX secrets, or the trusted optimizer
state. The attacker side never receives a public checkpoint or plaintext hidden states/weights.

## 6.2 Attacks and results (S1–S6)
All six positive controls pass (Table 3). Each attack maps to a published methodology (real
citations only): Mahendran & Vedaldi (CVPR'15) and Fredrikson et al. (CCS'15) for S1/S6; Hu et
al. LoRA (ICLR'22) for S2; Zhu et al. DLG (NeurIPS'19) and Zhao et al. iDLG for S4; Shokri et
al. (S&P'17) for S5.

- **S1 Representation inversion.** With no plaintext weights and no paired plaintext, inversion
  of a masked hidden state is under-determined (strict cosine 0.002 ≈ random). A *generous*
  known-pairs upper bound shows the orthogonal residual mask is exactly linearly invertible once
  ≥dim paired plaintext leaks (rel-err 5.7e-7) — so confidentiality rests on the TEE preventing
  such exposure, not on the mask being information-destroying. Norms and pairwise Grams are
  preserved exactly (documented structural leak, not claimed hidden).
- **S2 LoRA adapter recovery.** The masked product is recoverable but the plaintext ΔW is not
  (rel-err 1.41 ≈ √2, an uninformative orthogonal scramble) without the secret side masks;
  the positive control recovers plaintext ΔW at rel-err 2.4e-15.
- **S3 Logit leakage.** Permutation-only masking preserves the confidence profile *exactly*
  (multiset gap 0, confidence correlation 1.0); the proposed **monomial** mask removes this
  (correlation 0.009, KL 3.7) at a bounded condition number.
- **S4 Gradient inversion.** Masking is a transparent orthogonal change of basis: the plaintext
  DLG solution mapped through the mask matches the masked gradients with loss-gap 9e-13 and
  recovers the identical token in 100% of cases. Defense is batch aggregation (recovery → 0 at
  batch 8) and non-exposure of per-example gradients — **not** the mask.
- **S5 Membership inference.** The plaintext black-box MIA is non-random (AUC 0.614); membership
  leaks at the *trained weights* (generalization gap). Permutation-only masking keeps the signal
  readable (AUC 0.595); the monomial mask reduces it to chance (AUC 0.499).
- **S6 KV-cache inversion.** Plaintext KV yields token recovery well above chance (top-1 0.541
  vs 1.3e-3); the masked KV defeats a plaintext-calibrated attacker (top-1 0.003) but is
  orthogonal-invertible given masked pairs (top-1 0.536) — protection is mask secrecy / the TEE.

## 6.3 What we deliberately do NOT claim
Attention scores are exposed on the untrusted GPU by the paired-feature construction (the
attention-fingerprint leak); we document it and do not claim attention-score confidentiality.
Input tokens at the embedding boundary are recoverable from the masked embedding table (any
deterministic embedding); input-token protection is a separate input-pad / layer-0-TEE
mechanism. The recurring, honest theme is that the masks are **exact orthogonal/permutation
operators** whose confidentiality is real only under the TEE assumptions (no paired-plaintext or
per-example-gradient exposure, refreshed masks, aggregated updates).

## 6.4 Component attribution
The ablation (§5.5) attributes each attack surface to a component: feature mask → S1/S6,
monomial → S3/S5, refresh → cross-step KPA. This demonstrates *contribution*, not a security
proof; we state this explicitly.
