# G1/G2 Corpus and Diagnostic Metric Audit (v2)

Primary paper metrics are corpus BLEU, corpus chrF, and explicitly aggregated ROUGE-L. Mean sentence metrics are retained as secondary diagnostics.

ROUGE-L is the arithmetic mean over examples of 100 × the maximum whitespace-token LCS F1 across references.

Each paired 95% interval uses 10,000 matched-ID bootstrap replicates and recomputes the complete corpus metric.

## Primary corpus metrics

| Seed | Metric | G1 | G2 | G2−G1 | Paired 95% CI |
|---:|---|---:|---:|---:|---:|
| 7 | corpus_bleu | 63.934 | 64.630 | +0.695 | [-0.421, +1.805] |
| 7 | corpus_chrf | 72.413 | 72.158 | -0.256 | [-1.030, +0.545] |
| 7 | rouge_l | 61.311 | 60.813 | -0.498 | [-1.448, +0.456] |
| 1234 | corpus_bleu | 63.982 | 65.884 | +1.902 | [+0.650, +2.942] |
| 1234 | corpus_chrf | 72.783 | 70.559 | -2.224 | [-3.006, -1.423] |
| 1234 | rouge_l | 61.403 | 61.956 | +0.553 | [-0.288, +1.407] |
| 2025 | corpus_bleu | 64.312 | 63.162 | -1.149 | [-2.476, +0.171] |
| 2025 | corpus_chrf | 72.425 | 71.255 | -1.170 | [-2.055, -0.264] |
| 2025 | rouge_l | 61.577 | 60.576 | -1.001 | [-2.066, +0.073] |

## Secondary diagnostics

| Seed | Metric | G1 | G2 | G2−G1 | Paired 95% CI |
|---:|---|---:|---:|---:|---:|
| 7 | mean_sentence_bleu | 62.834 | 62.721 | -0.113 | [-1.219, +0.969] |
| 7 | mean_sentence_chrf | 72.630 | 72.209 | -0.421 | [-1.151, +0.300] |
| 7 | output_length | 33.230 | 32.112 | -1.118 | [-1.636, -0.604] |
| 7 | invalid_rate | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| 7 | repetition_bigram_fraction | 3.053 | 2.944 | -0.108 | [-0.412, +0.190] |
| 7 | repeated_output_rate | 50.600 | 53.400 | +2.800 | [-1.800, +7.400] |
| 1234 | mean_sentence_bleu | 62.926 | 63.339 | +0.413 | [-0.669, +1.463] |
| 1234 | mean_sentence_chrf | 72.941 | 70.795 | -2.146 | [-2.832, -1.456] |
| 1234 | output_length | 33.318 | 30.388 | -2.930 | [-3.446, -2.406] |
| 1234 | invalid_rate | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| 1234 | repetition_bigram_fraction | 3.469 | 2.791 | -0.678 | [-1.017, -0.342] |
| 1234 | repeated_output_rate | 52.400 | 45.200 | -7.200 | [-11.600, -2.600] |
| 2025 | mean_sentence_bleu | 63.172 | 61.631 | -1.541 | [-2.757, -0.345] |
| 2025 | mean_sentence_chrf | 72.555 | 71.386 | -1.169 | [-1.939, -0.379] |
| 2025 | output_length | 33.058 | 32.740 | -0.318 | [-0.830, +0.190] |
| 2025 | invalid_rate | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| 2025 | repetition_bigram_fraction | 3.373 | 4.206 | +0.833 | [+0.455, +1.207] |
| 2025 | repeated_output_rate | 52.600 | 60.000 | +7.400 | [+2.600, +12.000] |

## Across-seed descriptive statistics

| Metric | G1 mean ± SD | G2 mean ± SD | Mean Δ ± SD | Hierarchical exploratory 95% interval |
|---|---:|---:|---:|---:|
| corpus_bleu | 64.076 ± 0.206 | 64.559 ± 1.362 | +0.483 ± 1.537 | [-1.152, +1.878] |
| corpus_chrf | 72.540 ± 0.210 | 71.324 ± 0.802 | -1.217 ± 0.985 | [-2.245, -0.217] |
| mean_sentence_bleu | 62.977 ± 0.175 | 62.564 ± 0.865 | -0.414 ± 1.011 | [-1.580, +0.623] |
| mean_sentence_chrf | 72.709 ± 0.205 | 71.464 ± 0.710 | -1.245 ± 0.865 | [-2.151, -0.370] |
| rouge_l | 61.430 ± 0.135 | 61.115 ± 0.738 | -0.315 ± 0.793 | [-1.219, +0.611] |
| output_length | 33.202 ± 0.132 | 31.747 ± 1.218 | -1.455 ± 1.338 | [-2.873, -0.336] |
| invalid_rate | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 ± 0.000 | [+0.000, +0.000] |
| repetition_bigram_fraction | 3.298 ± 0.218 | 3.314 ± 0.777 | +0.016 ± 0.763 | [-0.647, +0.808] |
| repeated_output_rate | 51.867 ± 1.102 | 52.867 ± 7.414 | +1.000 ± 7.465 | [-6.868, +7.800] |

The hierarchical intervals are exploratory seed-and-example resampling intervals. They are not equivalence tests and do not establish strict equivalence.

All six inputs were selected through the canonical registry and verified by size and SHA-256 before parsing.
