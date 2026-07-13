# Profile E (exact) vs Profile N (native transformed-coordinate) — LOCAL gate

Source: `profile_n_gate.json` (CPU synthetic gate; the real-Qwen 1/10/50-step on A10+TDX is the
deferred Phase 3 hardware gate). Masks: **dense orthogonal** (primary) — AdamW's per-element
preconditioner is basis-dependent, so Profile N follows a genuinely different trajectory.

| step | metric | plaintext | Profile E (exact) | Profile N (native) |
|---|---|---|---|---|
| 50 | top-1 utility | 1.00 | 1.00 | **1.00** |
| 50 | final loss | 2.1e-5 | 2.1e-5 | **2.1e-5** |
| 50 | eff. ΔW rel-diff vs plaintext | 0 | **0 (exact)** | **0.96 (different)** |
| 50 | eff. ΔW cosine vs plaintext | 1.0 | 1.0 | **0.534** |
| 50 | trusted optimizer bytes (in TDX) | n/a | **49152** (master+m+v) | **0** |
| 1/10 | ΔW rel-diff vs plaintext (N) | — | 0 | **1.13 / 0.99** |

**Reading (honest):**
- **Profile E reproduces plaintext EXACTLY** (rel-diff 0) — it is the correctness/theorem reference.
- **Profile N is a DISTINCT optimizer** (ΔW rel-diff 0.96, cosine 0.534 vs plaintext — *not*
  equivalent) that nonetheless reaches **the same utility** (top-1 1.00, matched loss) while keeping
  **zero plaintext optimizer state in the TDX**.
- **Special finding:** under a *signed-permutation* transform, native transformed AdamW *coincides*
  with plaintext AdamW (rel-diff **1.24e-15**) because per-element m/v are permutation/sign
  equivariant; a *dense* transform breaks this. (`signed_permutation_equivariance` in the json.)

We do NOT claim Profile N equals standard plaintext AdamW; we claim comparable utility at lower
trusted footprint on this local gate, pending the hardware Phase 3 confirmation.
