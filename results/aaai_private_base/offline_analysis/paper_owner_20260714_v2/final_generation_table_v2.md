# Final Generation Table (v2)

Corpus metrics are paper-facing; all generation JSONL inputs were registry-selected and hash-verified.

| Method | Seed | Corpus BLEU | Corpus chrF | ROUGE-L | Length | Invalid % | Repetition bigram % | Paired G2−G1 BLEU CI | Provenance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G0 | — | 41.333 | 59.132 | 47.347 | 34.184 | 0.000 | 1.885 | — | MEASURED |
| G1 | 7 | 63.934 | 72.413 | 61.311 | 33.230 | 0.000 | 3.053 | — | MEASURED |
| G2 | 7 | 64.630 | 72.158 | 60.813 | 32.112 | 0.000 | 2.944 | [-0.421, +1.805] | MEASURED |
| G1 | 1234 | 63.982 | 72.783 | 61.403 | 33.318 | 0.000 | 3.469 | — | MEASURED |
| G2 | 1234 | 65.884 | 70.559 | 61.956 | 30.388 | 0.000 | 2.791 | [+0.650, +2.942] | MEASURED |
| G1 | 2025 | 64.312 | 72.425 | 61.577 | 33.058 | 0.000 | 3.373 | — | MEASURED |
| G2 | 2025 | 63.162 | 71.255 | 60.576 | 32.740 | 0.000 | 4.206 | [-2.476, +0.171] | MEASURED |

Across seeds, corpus BLEU is 64.076 ± 0.206 for G1 and 64.559 ± 1.362 for G2 (mean G2−G1 +0.483).
The paired confidence intervals test zero difference within each seed. They are not equivalence intervals.
