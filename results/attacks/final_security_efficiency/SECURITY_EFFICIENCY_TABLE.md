# Final consolidated security + efficiency table (measured, Qwen2.5-7B-Instruct)

All numbers **measured on real Qwen2.5-7B-Instruct (H800)**. Security cells = leak / recovery rate in **[0,1], lower is safer**. Built by `scripts/attacks/build_final_security_efficiency_table.py` from the JSONLs in `measured_inputs/` — no hand-transcription.

## Block A — security discrimination

| attack probe | threat model | plaintext | STIP | ObfuscaTune | ours-static (signed-perm) | ours-fresh (fresh_signed_perm) |
|---|---|---|---|---|---|---|
| KPA (known-plaintext) | known_plaintext | 1 | 1 | 1 | 1 | 0 |
| multiset permutation leakage | closed (default) | 1 | 1 | 0 | 0 | 0 |
| NN embedding inversion | public-model | 0.969 | 0 | 0 | 0 | 0 |
| frequency / per-token norm | closed (default) | 1 | 1 | 1 | 1 | 1 |
| Gram weight recovery [worst-case] | weight-leakage | blocked | 1 | 0 | 1 | 1 |
| ArrowMatch weight align [worst-case] | weight-leakage | blocked | 1 | 0 | 0.506 | 0.5 |

**Reading block A.** KPA + multiset together separate all three defenses pairwise: STIP loses both, ObfuscaTune loses KPA only, ours-fresh resists both. The `frequency / per-token norm` row is ≈1.0 for **every** norm-preserving scheme (shared weak statistical leak, not a discriminator, disclosed not hidden). The two `[worst-case]` rows require the **public** base weights as a match anchor (`gram_weight_recovery` needs `model_weights['public']`; verified in code) — under a **proprietary model (weights + embedding table private)** they are **blocked**, so they form a worst-case band and do NOT enter the default verdict.

## Block C — efficiency (realistic hybrid: Qwen on GPU + TDX mask on CPU)

base GPU decode = **15.97 ms/token** (62.6 tok/s), ctx=512, bf16. Consistent convention: each `+ms` is the TEE-side cost added on top of the SAME real GPU decode.

| scheme | TEE-side +ms/tok | end-to-end ms/tok | tok/s | overhead | TEE crossings |
|---|---|---|---|---|---|
| plaintext | +0.000 | 15.97 | 62.6 | +0.0% | none |
| STIP | +0.000 | 15.97 | 62.6 | +0.0% | none |
| ObfuscaTune | +1.551 | 17.52 | 57.1 | +9.7% | 224x |
| ours-static (signed-perm) | +0.000 | 15.97 | 62.6 | +0.0% | 2x |
| ours-fresh (fresh_signed_perm) | +0.137 | 16.10 | 62.1 | +0.9% | 2x |
| ours-fresh-pad (dense, ref) | +682.9 | 698.9 | 1.4 | +4277% | 2x |
| ours-non-isometric (dense, ref) | +1744.5 | 1760.5 | 0.6 | +10925% | 2x |

**Reading block C.** ours-static and ours-fresh add ~0/+0.9% (static fold / O(D) signed-perm gen); ObfuscaTune adds +9.7% because it runs 224 nonlinearities/token *inside* the TEE (not a mask cost). Dense-fresh masks (fresh-pad / non-isometric) need an O(D^3) QR per token → prohibitive; listed for reference only.

## Verdict (default deployment: closed model, private weights + embedding table)

- **Gram/ArrowMatch**: blocked for all (no public anchor) → not a differentiator.
- **KPA**: ours-fresh RESIST (heldout deobf err 1.47, succ 0.0) vs ObfuscaTune BROKEN (5e-7) and STIP BROKEN (0.0). Weight privacy does NOT cover KPA — freshness does.
- **multiset / NN**: ours-fresh and ObfuscaTune both 0.0; STIP leaks multiset (1.0).
- **Efficiency**: ours-fresh +0.9% vs ObfuscaTune +9.7% (one order of magnitude).

→ Under the stated private-weights threat model, **ours-fresh (fresh_signed_perm) matches-or-beats ObfuscaTune on every applicable security probe and is ~10x cheaper in TEE overhead.** Two orthogonal secrets do the work: private weights neutralise weight-anchored mask recovery (Gram class); per-token freshness neutralises KPA. If the base model is ever PUBLIC, the worst-case band reactivates and ours-fresh's static signed-perm fold is Gram-broken (1.0) — this boundary is stated, not hidden.

