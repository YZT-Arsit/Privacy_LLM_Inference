# Limitations (draft)

We state limitations plainly; security is claimed only as *empirical resistance under the
evaluated threat model and published attacks*.

**Security is empirical and assumption-scoped.** The masks are exact orthogonal/permutation
operators that preserve norms, pairwise Grams, and value multisets, and are linearly invertible
given paired plaintext (S1 synthetic rel-err 5.7e-7). Confidentiality therefore rests on the TEE
preventing paired-plaintext and per-example-gradient exposure, plus batch aggregation — not on
the algebraic masking alone. We make no formal, information-theoretic, or zero-leakage claim.

**Documented residual leaks.** Attention scores are exposed on the untrusted GPU (partial
mitigation: layer-0 TEE relocation only); norms/Grams/multisets are preserved; input tokens at
the embedding boundary are recoverable from the masked table for any deterministic embedding.
These are recorded, not claimed away.

**Utility breadth.** Converged utility is shown on one classification task (SST-2, 3 seeds, wide
CI → equivalence-in-mean, not strict TOST). The downstream-generation result (GSM8K) is at
Qwen2.5-0.5B's reasoning floor; a converged GSM8K LoRA is compute-infeasible here (≈74.6 h/seed),
so we claim protected≡plaintext *parity*, not a reasoning gain or LoRA superiority.

**Scale.** Empirical results are at 0.5B. A full 7B protected-LoRA fine-tune was not run — no 7B
checkpoint is reachable in our environment. Scalability of the transform is supported by frozen
7B generation/fold + security artifacts and the size-independent 0.5B optimizer-exactness, plus a
labeled cost projection (not a measurement); full 7B training is future work.

**System overhead.** The dominant protected-training cost is a fixed per-optimizer-step TDX
round trip (~114 ms enclave AdamW on a CPU-only guest + ~44 ms CE network); it is inherent to
effective-batch-16 training and would benefit from a GPU/accelerated enclave. The per-token
decode overhead of the two-machine prototype (~9×) is a transport artifact, not algorithmic.

**Baselines.** No published privacy-preserving-LoRA method shares our threat model (base-weight
confidentiality against an untrusted execution host with a TEE); DP/federated LoRA defend a
different asset and inference-only obfuscation baselines do not train LoRA. We therefore compare
against plaintext LoRA + ablations, and use inference-only baselines for generation/overhead
only — with the N/A justification stated, not hidden.

**Attacker model.** We evaluate an honest-but-curious black-box/observational attacker with the
published attacks S1–S6. White-box, adaptive, side-channel, and compromised-TEE attackers are
out of scope.
