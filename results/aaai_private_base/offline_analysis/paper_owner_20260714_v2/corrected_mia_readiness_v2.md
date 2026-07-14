# Corrected MIA CPU Readiness (v2)

Status: **READY_FOR_HASHED_COLLECTIONS; NO SHADOW TRAINING LAUNCHED**.

The isolated evaluator `scripts/corrected_mia_evaluator_v2.py` implements:

- strict matched V0/V2 manifest, SHA-256, 1,000-sample, schema, and C0/eval/forward-only validation;
- identical sample-pool, query-ID, collection-path, duplicate-group, and semantic-group checks;
- a disallowed-field gate for membership/split/source/run/session/checkpoint/mode/step/epoch/tensor-presence fields;
- membership labels joined only from separate evaluator metadata;
- leave-one-shadow-run-out and semantic-group-isolated leave-samples-out evaluation;
- label-shuffle and within-member/within-nonmember pseudo-membership controls;
- imputation, scaling, univariate feature selection, and logistic-regression fitting inside a train-fold-only pipeline;
- all-feature and per-family ablations;
- semantic-group bootstrap confidence intervals;
- paired semantic-group bootstrap intervals for the primary matched V2-minus-V0 comparison;
- ROC-AUC, TPR at 1% and 0.1% FPR, empirical FPR resolution, and attack advantage.

Attack advantage is defined as `max_tau (TPR(tau) - FPR(tau))` on the stated test fold. This is an
empirical descriptive maximum, not a separately calibrated operating threshold. TPR at 0.1% FPR is
explicitly flagged as resolution-limited whenever a fold has fewer than 1,000 negatives.

The evaluator fails closed if any required bundle is missing or mismatched. Current corrected shadow
collections are absent, so no corrected MIA result is reported. The old source-confounded AUC=1 is
excluded from final-result use.
