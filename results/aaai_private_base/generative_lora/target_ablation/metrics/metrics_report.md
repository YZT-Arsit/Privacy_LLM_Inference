# Phase 6 — Generation Quality Metrics (E2E NLG)

Multi-reference. BLEU/chrF via sacrebleu; ROUGE-L local LCS-F1. Offline (Mac control-plane).

| Cell | BLEU | chrF | ROUGE-L | METEOR | invalid% | rep% | avg_len | n |
|---|---|---|---|---|---|---|---|---|
| ALL7 | 63.98 | 72.78 | 61.4 | unavailable_no_nltk | 0.0 | 3.47 | 33.3 | 500 |
| QV | 62.3 | 68.99 | 59.64 | unavailable_no_nltk | 0.0 | 2.59 | 30.6 | 500 |

**Paired per-example BLEU vs ALL7** (same sample_id):

- QV: mean Δ -2.506, median Δ -0.655, better/worse 186/260 of 500
