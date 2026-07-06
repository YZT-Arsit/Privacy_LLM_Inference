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

## Block A robustness — cross sequence-length / input-statistics

Real text (ag_news_small.jsonl), subdim 256. Captured at layer 0 (per-token, length-invariant by construction) AND a mid layer (attention-mixed, where length is a real variable). KPA success (1=broken, 0=resist); multiset leak (1=leak, 0=safe).

| seq_len | layer | KPA obf | KPA ours-static | KPA ours-fresh | mset STIP | mset obf | mset ours-fresh |
|---|---|---|---|---|---|---|---|
| 64 | embed | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 64 | mid(L14) | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 256 | embed | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 256 | mid(L14) | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 512 | embed | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| 512 | mid(L14) | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |

**Reading.** The verdict does not move with sequence length or input statistics, at both the per-token layer and the attention-mixed mid layer. These attacks are geometric/statistical, not semantic — so a single corpus + length sweep suffices; re-running across domains would buy duplicate numbers, not new information.

## Block C completeness — correctness / honest boundary cost / memory

- **Correctness (ours vs plaintext)**: signed-perm residual fold max|y'−y| = 3.10e-06 → **lossless** (fp32). STIP/ObfuscaTune not tested (also exact linear maps, not the discriminator).
- **Fresh boundary cost, consistent convention** (the e2e +0.9% counted mask generation only): gen 0.1002 + apply 0.0210 + unmask 0.0225 = **0.1438 ms/token (0.90% of decode)**. Static folds add 0 (mask is inside the weights). Even the corrected total stays an order below ObfuscaTune's 1.55 ms of in-TEE nonlinearities.
- **Memory**: folded weights are the SAME size as plaintext (0 extra). Mask buffers: static 30697.51 / fresh 30697.64 MB peak-with-model (O(D), KB-scale delta); a dense DxD mask adds ~105 MB — why dense-fresh is doubly impractical.

## Block B (motivation appendix) — prompt recovery across sensitive domains

Best-effort output-only optimization attack (BRE) through the real split downstream, on the repo's SYNTHETIC sensitive prompts (no real PII). Cell = token_recovery_top1 (higher = more recovered). NON-discriminative by design.

| domain | attack | plaintext | STIP | ObfuscaTune | ours-fresh |
|---|---|---|---|---|---|
| medical | bre | 0.719 | 0 | 0 | 0 |
| legal | bre | 0.875 | 0 | 0 | 0 |
| financial | bre | 0.812 | 0 | 0 | 0 |
| general | bre | 0.812 | 0 | 0 | 0 |

**Reading (motivation only).** Plaintext prompts leak (0.72–0.88); STIP, ObfuscaTune and ours all crush recovery to 0. The mask STRUCTURE does not separate the defenses against output-only attacks — only the PRESENCE of a mask matters. Discrimination between defenses lives in block A (KPA), not here. Best-effort implementation, not a full SOTA reproduction; motivation echo, not a ranking claim.

## Block D — second model family (Llama): cross-family reproduction

Different family than Qwen (LlamaForCausalLM vs Qwen2, vocab 128256 vs 152064). Security cells = leak/recovery (lower safer); ObfuscaTune shown as its KPA success (broken). Same runners, only `--model-name-or-path` changed.

### Llama-3.2-1B (hidden 2048, 16 layers, vocab 128256)

| probe | threat | STIP | ObfuscaTune | ours-static | ours-fresh |
|---|---|---|---|---|---|
| KPA | known_plaintext | 1 | 1 | 1 | 0 |
| multiset | closed | 1 | 0 | 0 | 0 |
| Gram [worst] | weight-leak | 1 | 0 | 1 | 1 |
| ArrowMatch [worst] | weight-leak | 1 | 0 | 0.51 | 0.508 |

Efficiency (base decode **8.92 ms/tok**): ours-fresh +1.0% vs ObfuscaTune +8.9% (ours-static +0.0%); dense-fresh +2203%.

### Llama-3-8B (hidden 4096, 32 layers, vocab 128256)

| probe | threat | STIP | ObfuscaTune | ours-static | ours-fresh |
|---|---|---|---|---|---|
| KPA | known_plaintext | 1 | 1 | 1 | 0 |
| multiset | closed | 1 | 0 | 0 | 0 |
| Gram [worst] | weight-leak | 1 | 0 | 1 | 1 |
| ArrowMatch [worst] | weight-leak | 1 | 0 | 0.512 | 0.504 |

Efficiency (base decode **17.48 ms/tok**): ours-fresh +0.9% vs ObfuscaTune +9.6% (ours-static +0.0%); dense-fresh +5293%.

**Reading.** Every block-A + efficiency conclusion reproduces on both Llama sizes: static masks KPA-broken, fresh KPA-resist; STIP multiset-leaks, others don't; Gram breaks STIP + ours (public weights), ObfuscaTune resists; ours-fresh ~10x cheaper than ObfuscaTune. The verdict is family- and scale-invariant. (8B KPA from run stdout; the confirmatory non_isometric row was skipped as its O(D^3) per-token QR at dim 4096 is the very cost the paper flags as prohibitive.)

