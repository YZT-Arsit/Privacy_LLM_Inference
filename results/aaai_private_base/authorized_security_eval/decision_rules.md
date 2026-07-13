# Pre-registered decision rules

Frozen before the authorized V0-versus-V2 aggregate results were available.

Primary comparisons use paired samples/seeds wherever possible. Confidence
intervals are 95% intervals; example-level metrics use a paired bootstrap with
10,000 resamples, and seed-level summaries report all seeds without treating
examples as independent seeds.

| Family | Metric | Meaningful-advantage margin | Pass rule |
|---|---|---:|---|
| Surrogate fidelity | BLEU, chrF, ROUGE-L | +2.0 points | upper 95% CI for V2−V0 ≤ 2.0 |
| Surrogate fidelity | exact/next-token agreement | +0.02 absolute | upper 95% CI ≤ 0.02 |
| Membership | ROC-AUC | +0.03 absolute | upper 95% CI ≤ 0.03 |
| Membership | TPR at 1% FPR | +0.02 absolute | upper 95% CI ≤ 0.02 |
| Membership | TPR at 0.1% FPR | +0.01 absolute | upper 95% CI ≤ 0.01 |
| Input inference | exact-token/top-1 reconstruction | +0.02 absolute | upper 95% CI ≤ 0.02 |
| Input inference | repeated-input link AUC | +0.03 absolute | upper 95% CI ≤ 0.03 |

A conclusion is `empirically black-box-like` only for a named asset/task/view,
budget and attack family whose primary rule passes. A failed or underpowered
cell is reported as such and is never converted into evidence of security.
