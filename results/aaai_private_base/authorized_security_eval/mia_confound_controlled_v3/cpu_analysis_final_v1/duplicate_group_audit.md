# Duplicate and group audit

Collection mode: SIMULATED_PROTOCOL_VIEW

- Exact duplicate inputs: 0 rows in 0 groups.
- Exact duplicate references: 2 rows in 1 groups.
- Repeated exact meaning representations: 0.
- Near-duplicate MR pairs at token Jaccard ≥0.9: 155.
- Semantic template (`mr_field_set`) groups: 102 (sizes 1–45).
- Sample IDs present in all three shadows: 1,000 (intentional frozen-pool reuse).

No rows were removed. The primary LOSO is accompanied by a five-fold semantic-template-group-disjoint sensitivity in `per_fold_metrics.csv`.

Generated-output audit:
- Shadow 1: 9 duplicate groups, 19 affected rows, maximum group 3.
- Shadow 2: 9 duplicate groups, 18 affected rows, maximum group 2.
- Shadow 3: 14 duplicate groups, 30 affected rows, maximum group 3.
