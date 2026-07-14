# Final paired analysis: G2_L12_s2025 minus G1_s2025

All statistics use the existing 500 aligned frozen E2E outputs; generation was not rerun.

| Metric | Mean delta | 95% paired bootstrap CI | Cohen dz | + / − / 0 |
|---|---:|---:|---:|---:|
| BLEU | -1.541 | [-2.738, -0.319] | -0.113 | 180/247/73 |
| chrF | -1.169 | [-1.961, -0.366] | -0.129 | 166/265/69 |
| ROUGE_L | -1.001 | [-2.051, 0.095] | -0.082 | 172/249/79 |
| length | -0.318 | [-0.822, 0.192] | -0.054 | 192/206/102 |
| repetition_pct | 0.833 | [0.452, 1.208] | 0.193 | 195/120/185 |
| invalid_pct | 0.000 | [0.000, 0.000] | N/A | 0/0/500 |

## Interpretation

G2 is competitive with G1 on the frozen evaluation set. The paired sentence-BLEU interval excludes zero.
chrF is lower for G2 with a paired interval excluding zero.
No equivalence or superiority claim is made.
