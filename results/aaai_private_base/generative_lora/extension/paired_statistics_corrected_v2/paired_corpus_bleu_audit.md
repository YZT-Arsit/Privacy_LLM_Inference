# Corrected paired corpus-BLEU audit (v2)

This report does not overwrite the earlier mean-sentence-BLEU analysis.

| Seed | G1 corpus BLEU | G2 corpus BLEU | Corpus Δ | Mean sentence-BLEU Δ | Paired bootstrap corpus-Δ 95% CI | Matched |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 63.98 | 65.88 | +1.90 | +0.413 | [+0.650, +2.942] | 500 |
| 7 | 63.93 | 64.63 | +0.70 | -0.113 | [-0.421, +1.805] | 500 |
| 2025 | 64.31 | 63.16 | -1.15 | -1.541 | [-2.476, +0.171] | 500 |

## Audit findings

- All three pairs contain exactly 500 matched IDs, with no duplicate or missing IDs.
- References, meaning representations, and input hashes are identical within every pair.
- The discrepancy is expected: corpus BLEU aggregates n-gram counts before applying the BLEU formula; averaging sentence BLEU is a different statistic.
- The workspace seed-1234 G2 JSONL is a truncated 524,288-byte copy. The verified cache copy exactly matches the completion-manifest size and SHA-256 and was used without modifying the frozen path.

## Exact generation hashes

- seed 1234 G1: `ec5a432031dcbfed61152e6a1b8299bfc55704568342e7890b412f49e9b7c76b`
- seed 1234 G2: `d3b5ad9820c13e473ee7a995480a9ea6b5b3f5a786a4ba3276774cf116e46e31`
- seed 7 G1: `0cfef06b7ace081b71303ca05f7709deb6562b521803f5dc9d9cff9b034aeb17`
- seed 7 G2: `70a93b9d50d4cb67037a04f70e0cf9a608ff93f8a5b64dd095e06cae686c2aec`
- seed 2025 G1: `744d1758db59cff389fa618e8682ea412f076be646257ace296e33da392906f2`
- seed 2025 G2: `1c7c7a7a12fc9845a9635b1634e8aa5c4fc3a0aabb1185ccfd566770de89fede`
