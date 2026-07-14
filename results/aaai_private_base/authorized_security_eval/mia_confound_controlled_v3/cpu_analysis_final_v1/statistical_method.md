# Statistical method

Collection mode: SIMULATED_PROTOCOL_VIEW

Primary evaluation uses three leave-one-shadow-out folds. All imputation, scaling, TF-IDF vocabulary construction and C selection are fit only on training shadows. Internal tuning swaps the two training shadows. A stricter five-fold sensitivity holds out entire `mr_field_set` groups and removes those sample IDs from both training shadows.

Metrics are ROC-AUC, balanced accuracy, average precision, TPR at 1% FPR, and max(TPR−FPR). TPR at 0.1% FPR is not reported because 500 held-out negatives give an empirical resolution of 0.2%. Logistic Brier score is reported; SVM is not treated as probabilistic.

Paired uncertainty uses 2000 percentile-bootstrap replicates. Samples are resampled within each held-out shadow, the same indices are used for both compared views, and differences are aggregated equally over the three shadows. Intervals are exploratory because there are only three shadows.

