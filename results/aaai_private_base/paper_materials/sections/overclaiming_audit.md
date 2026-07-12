# Overclaiming-language audit (draft)

Purpose: verify no paper material asserts security beyond *empirical resistance under the
evaluated threat model*. Method: automated scan of `paper_materials/**` for forbidden and
risky terms, then manual adjudication of each hit.

## Forbidden vocabulary (must never describe our security)
`zero leakage`, `impossible (recovery)`, `information-theoretic(ally secure)`,
`mathematically/provably/perfectly secure`, `unbreakable`, `cannot be recovered`,
`fully/perfectly private`. For utility: `outperform`, `superior`, `state-of-the-art`,
`best-in-class` (we claim parity, not superiority).

## Permitted vocabulary
`empirical resistance`, `fails under the evaluated attacks`, `consistent with the threat
model`, `documented (structural) leak`, `we do not claim …`, `equivalence in mean`,
`assumption-scoped`.

## Scan result (adjudicated)
Every occurrence of a forbidden term appears inside a **disclaimer or negation** — i.e. we say
we do *not* make that claim:
- "**no** claim of zero leakage, impossible recovery, or information-theoretic security"
  (limitations, security section) — correct usage.
- "no gain/superiority claimed", "we do not claim LoRA superiority" (utility, GSM8K) — correct.

Two soft uses of the word *guarantee/assurance*, both adjudicated acceptable:
1. **Correctness** ("the only theorem-like guarantees"): refers to the **exact fp64 algebraic
   identity** of the masked forward + optimizer (max fold err < 1e-8). This is a genuine
   exact-equivalence guarantee about *correctness*, not a security claim, and is explicitly
   scoped in quotes. Retained.
2. **Limitations** previously read "its guarantees are empirical and assumption-scoped" →
   softened to "its **assurances** are empirical and assumption-scoped" to remove any ambiguity.

**Conclusion:** no surviving overclaim. Security is uniformly described as empirical and
assumption-scoped; correctness is described as exact; utility is described as parity (never
superiority). Scale/7B is described as deferred, not claimed.

## Standing wording rules for the final manuscript
1. Security verbs: *resists / fails under / is consistent with* — never *prevents / guarantees /
   makes impossible*.
2. Always pair a protected result with (a) its positive control, (b) a random baseline, and
   (c) the documented structural leak.
3. Never let the algebraic mask carry the confidentiality claim on its own — always attribute
   it to the TEE assumptions (no paired-plaintext / per-example-gradient exposure + aggregation).
4. Utility = *parity / no systematic loss*, never *improvement / superiority* unless a
   controlled comparison establishes it (it does not, here).
5. 7B and converged GSM8K = *deferred / future work*, never *demonstrated*.
