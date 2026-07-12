# Claim → basis → experiment → evidence (AAAI private-base)

Honest mapping. **Correctness** claims rest on exact algebraic identities (fp64-validated
to < 1e-8) — these are the only "theorem-like" guarantees. **Security** claims are
**empirical under the evaluated threat model and published attacks** — no formal-security
theorem is asserted. Wording is scoped throughout (no zero-leakage / impossible-recovery /
information-theoretic claims).

| # | Claim | Formal basis | Experiment | Evidence artifact |
|---|---|---|---|---|
| C1 | The untrusted GPU package holds only transformed `*_tilde` artifacts — no plaintext base weight / embedding / LM-head / RMSNorm-γ / bias, and no mask secret | Fail-closed packager + independent scan (structural, not a probabilistic claim) | build + scan private-base package | `private_package/plaintext_absence_scan.json` (clean); `src/pllo/deployment/private_base_package.py` |
| C2 | The masked forward on `H̃`/`W̃` reconstructs the plaintext op **exactly** (all 8 Linear families + RMSNorm-fold + RoPE-commuting Q/K + monomial logits) | Exact algebraic identity, fp64-validated **max fold err < 1e-8**; RoPE-commuting B verified < 1e-9 | `gate0_build_private_package.py` in-build validation | `private_package/build_validation.json` |
| C3 | Protected LoRA fine-tuning (L12 bf16) matches the plaintext-equivalent reference (L5 fp32) in **utility** on SST-2 | Empirical equivalence-in-mean vs a 0.01 accuracy margin | converged 3-seed SST-2 (L0 + L5×3 + L12×3) | `utility_dataplane/PHASE7_SST2_UTILITY_SUMMARY.json` (mean paired L12−L5 = **+0.0027**) |
| C4 | The TDX-side trusted optimizer is **exact** for SGD / momentum / AdamW under masking | Exact fp64 equivalence (top-1 = 1.0, all trust-domain counters 0) | full LoRA optimizer matrix + real H800+TDX gate | `full_lora_matrix/**`, `gate0_d4/**` (prior, completed) |
| C5 | Masked representations resist **representation inversion** by an attacker without plaintext weights or paired plaintext | Empirical (evaluated attacks) | S1 (linear/MLP/deep-MLP; strict + known-pairs) | `security/S1_representation_inversion/s1_results.json` |
| C6 | Plaintext LoRA `ΔW` is not recovered from the transformed adapters without the secret side masks | Empirical | S2 (SVD / optimization / functional) | `security/S2_lora_recovery/s2_results.json` |
| C7 | The **monomial** logit mask reduces the distributional (multiset/confidence) leakage that permutation-only preserves exactly | Empirical + exact multiset identity for perm-only | S3 (B0/B1/B2) | `security/S3_logit_leakage/s3_results.json` |
| C8 | The mask does not by itself defend **gradient inversion** (it is a transparent orthogonal basis change); defense is aggregation + non-exposure | Empirical + exact basis-invariance (loss gap 9e-13) | S4 (DLG/iDLG + batch sweep) | `security/S4_gradient_inversion/s4_results.json` |
| C9 | Membership signal originates at the trained weights; the protected output channel keeps it readable under perm-only but not under monomial | Empirical (Shokri shadow MIA) | S5 (B0/B1-perm/B1-monomial) | `security/S5_membership/s5_results.json` |
| C10 | Masked KV cache defeats a plaintext-calibrated attacker; it is orthogonal-invertible given masked pairs (protection = mask secrecy / TEE) | Empirical | S6 (linear/MLP KV decoders) | `security/S6_kv_cache/s6_results.json` |
| C11 | Real end-to-end run on **real** A10 GPU + **real** Intel TDX with verified attestation | Measured (hardware) | migration + converged utility on A10+TDX | `alicloud_a10_migration/**`, `utility_dataplane/**` |

## Cross-cutting honest caveat (stated once, applies to C5–C10)
Orthogonal / permutation masks preserve norms, Grams, and value multisets and are linearly
invertible given paired plaintext. Empirical confidentiality therefore **rests on the TEE
preventing paired-plaintext and per-example-gradient exposure, plus batch aggregation** —
not on the algebraic mask alone. This is consistent with `design_spec.md §G/§H`, which
explicitly refuses to claim attention-score confidentiality or non-orthogonal-mask security.
