# Duplicate and semantic-group audit

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

- Exact duplicate inputs: 0 rows in 0 groups.
- Exact duplicate references: 2 rows in 1 groups.
- Repeated exact meaning representations: 0 rows.
- Near-duplicate pairs at token Jaccard ≥0.9: 155; groups: 65; sizes: [9, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2].
- Semantic-template groups (`mr_field_set`): 102.
- No samples were removed. The group-disjoint sensitivity excludes every test semantic-template group from both attack-training shadows.

## V0 output duplication/template audit

- Shadow 1: 9 exact-output groups (19 rows; max 3); 53 repeated-template groups (118 rows; max 4).
- Shadow 2: 9 exact-output groups (18 rows; max 2); 44 repeated-template groups (94 rows; max 4).
- Shadow 3: 14 exact-output groups (30 rows; max 3); 47 repeated-template groups (102 rows; max 5).
