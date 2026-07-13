# Final G2 quality analysis

All cells use the official E2E test subset with 500 aligned sample IDs and identical multi-reference targets. Decoding is greedy with `max_new_tokens=96`; G1 seed 1234 is the primary paired baseline.

| Cell | BLEU | chrF | ROUGE-L | invalid% | repetition% | avg tokens |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 41.33 | 59.13 | 47.35 | 0.00 | 1.88 | 34.2 |
| G1 seed 1234 | 63.98 | 72.78 | 61.40 | 0.00 | 3.47 | 33.3 |
| G1 seed 7 | 63.93 | 72.41 | 61.31 | 0.00 | 3.05 | 33.2 |
| G1 seed 2025 | 64.31 | 72.43 | 61.58 | 0.00 | 3.37 | 33.1 |
| G2 L12 seed 1234 | 65.88 | 70.56 | 61.96 | 0.00 | 2.79 | 30.4 |

G1 across three seeds is `64.073 +/- 0.207` BLEU, `72.540 +/- 0.208` chrF, and `61.430 +/- 0.138` ROUGE-L (sample standard deviation). G2 differs from these means by +1.807 BLEU, -1.980 chrF, and +0.530 ROUGE-L.

Against the same-seed G1 baseline, G2 improves corpus BLEU by 1.90 and ROUGE-L by 0.56, while chrF decreases by 2.22. The paired sentence-level BLEU mean difference is +0.413 over 500 aligned examples (205 better, 200 worse, 95 tied). A deterministic 10,000-sample percentile bootstrap gives a 95% interval of `[-0.628, 1.464]`; because it crosses zero, the paired sentence-BLEU improvement is not statistically established by this single G2 seed.

G2 produces no invalid outputs and has lower measured bigram repetition than the G1 seed-1234 baseline (2.79% versus 3.47%), but also shorter outputs (30.4 versus 33.3 tokens). METEOR is recorded as `nltk_error` because the required local NLTK resource was unavailable; no METEOR value should be claimed.

The protected training path uses BF16 runtime copies with an authoritative FP32 master state in TDX, whereas G1 training is FP32. The result supports successful protected training and competitive generation quality, but it is not strict numeric-equivalence evidence. A transformed adapter package, fresh-process restart subset, additional protected seeds, complete cost report, and target ablation remain separate work.
