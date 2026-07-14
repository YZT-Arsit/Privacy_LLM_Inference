# Final paired analysis: G2_L12_s7 minus G1_s7

All statistics use the existing 500 aligned frozen E2E outputs; generation was not rerun.

| Metric | Mean delta | 95% paired bootstrap CI | Cohen dz | + / − / 0 |
|---|---:|---:|---:|---:|
| BLEU | -0.113 | [-1.201, 0.971] | -0.009 | 209/222/69 |
| chrF | -0.421 | [-1.144, 0.323] | -0.051 | 202/229/69 |
| ROUGE_L | -0.498 | [-1.458, 0.443] | -0.045 | 205/218/77 |
| length | -1.118 | [-1.630, -0.604] | -0.189 | 153/239/108 |
| repetition_pct | -0.108 | [-0.406, 0.201] | -0.031 | 144/145/211 |
| invalid_pct | 0.000 | [0.000, 0.000] | N/A | 0/0/500 |

## Interpretation

G2 is competitive with G1 on the frozen evaluation set. The paired sentence-BLEU interval includes zero, so no significant BLEU difference is established.
chrF is lower for G2; its paired interval does not establish a significant degradation.
No equivalence or superiority claim is made.
