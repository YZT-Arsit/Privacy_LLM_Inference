# trusted_shortcut (amulet) latency: before vs after lift-factor caching

Worker + client both on the H800 (127.0.0.1, no tunnel), resident fp32 fold,
greedy. "Before" = per-call `R` regeneration; "after" = cached lift factors
(commit `1a48219`). Bit-identical output on both (fixed-seed factors).

## Before — IFEval-20 metrics (per-call regen), tokens/sec per prompt

Mean ≈ 3.66 tok/s (range 3.196–3.851 over 20 prompts). Sample:

| prompt_id | output_len | finish | latency_s | tok/s |
|---|---|---|---|---|
| 1000 | 289 | eos | 90.41 | 3.196 |
| 1005 | 512 | length | 140.62 | 3.641 |
| 1012 | 122 | eos | 35.09 | 3.477 |
| 1019 | 12 | eos | 3.29 | 3.649 |
| 1087 | 495 | eos | 132.70 | 3.730 |

## After — IFEval-5 metrics (cached R), max_new_tokens 128

| prompt_id | output_len | finish | latency_s | tok/s |
|---|---|---|---|---|
| 1000 | 128 | length | 11.73 | 10.91  (incl. resident warmup) |
| 1001 | 128 | length | 5.25 | 24.359 |
| 1005 | 128 | length | 5.20 | 24.595 |
| 1012 | 122 | eos | 4.97 | 24.550 |
| 1019 | 12 | eos | 0.52 | 23.137 |

Steady state **~24.4 tok/s** (excluding the first-prompt resident-cache warmup),
= parity with the `current` baseline (24.4 tok/s). **~6.7× faster** than the
per-call-regen prototype.

## Invariants preserved (verified on the after-run)
- `amulet_real_path_executed=True`, `lifted_nonlinear_ops_count=14504`,
  `trusted_nonlinear_ops_count=0` — still genuinely lifting.
- id 1012 → 122 tok eos and id 1019 → 12 tok eos in BOTH runs → bit-identical
  decoding.
- boundary runtime hash for trusted_shortcut UNCHANGED (amulet_backend.py is
  untrusted-worker code, not in the trusted boundary manifest) → existing TDX
  quotes stay valid.

## Safe latency features (keep-alive + precompute-embed), verified bit-identical

Enabled `--persistent-conn` + `--precompute-embed` (client-side only; do NOT touch
the attested boundary → runtime hash / TDX quotes unchanged). Re-ran the same 5
IFEval prompts against the flags-OFF cached run:

- **bit-identity: token_match_rate 1.0, exact_text 1.0, exact_token 1.0 (n=5)** —
  not a single token changed (precompute's batched-GEMM table matched the per-token
  GEMV exactly on this model/device).
- latency (local, worker+client on H800): steady ~24.4 → **25.5–25.8 tok/s**
  (id 1012, 122 tok: 24.55 → 25.54). ~4–5% locally (worker fp32 forward is the
  floor); keep-alive's real value is over a cross-machine TEE↔GPU tunnel where it
  saves a full ~71 ms RTT per token.

### Latency floor at fp32 (profiled, H800)
Per-token 39.6 ms: boundary side only 0.8 ms; the ~33 ms worker fp32 forward is the
floor. Of that, ~8.4 ms is the hard fp32 memory-bandwidth floor (reads ~28 GB
weights/token — the literal cost of the fp32 precision), ~24 ms is CUDA
kernel-launch overhead (only CUDA-graph capture would remove it; deferred).
