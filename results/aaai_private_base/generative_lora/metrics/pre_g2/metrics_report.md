# Phase 6 — Generation Quality Metrics (E2E NLG)

Multi-reference. BLEU/chrF via sacrebleu; ROUGE-L local LCS-F1. Offline (Mac control-plane).

| Cell | BLEU | chrF | ROUGE-L | METEOR | invalid% | rep% | avg_len | n |
|---|---|---|---|---|---|---|---|---|
| G0 | 41.33 | 59.13 | 47.35 | nltk_error | 0.0 | 1.88 | 34.2 | 500 |
| G1_s1234 | 63.98 | 72.78 | 61.4 | nltk_error | 0.0 | 3.47 | 33.3 | 500 |
| G1_s2025 | 64.31 | 72.43 | 61.58 | nltk_error | 0.0 | 3.37 | 33.1 | 500 |
| G1_s7 | 63.93 | 72.41 | 61.31 | nltk_error | 0.0 | 3.05 | 33.2 | 500 |

**Paired per-example BLEU vs G1_s1234** (same sample_id):

- G0: mean Δ -21.831, median Δ -18.215, better/worse 51/449 of 500
- G1_s2025: mean Δ 0.246, median Δ 0.0, better/worse 152/127 of 500
- G1_s7: mean Δ -0.092, median Δ 0.0, better/worse 151/133 of 500
