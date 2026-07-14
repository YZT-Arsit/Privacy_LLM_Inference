# Statistical method

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

Primary evaluation uses exactly `(S2,S3)→S1`, `(S1,S3)→S2`, and `(S1,S2)→S3`. Imputation, scaling, TF-IDF vocabularies, and C selection are fitted only within the two attack-training shadows. Group-disjoint sensitivity uses five deterministic `mr_field_set` partitions and excludes test-group IDs from both training shadows.

Metrics include ROC-AUC, balanced accuracy, average precision, max(TPR−FPR), TPR@1% FPR, and logistic calibration metrics. TPR@0.1% FPR is withheld because 500 negatives give 0.2% empirical resolution.

Paired uncertainty uses 2000 within-shadow percentile-bootstrap replicates, shared indices across compared views, and equal shadow weighting. Intervals are exploratory because there are only three shadows.
