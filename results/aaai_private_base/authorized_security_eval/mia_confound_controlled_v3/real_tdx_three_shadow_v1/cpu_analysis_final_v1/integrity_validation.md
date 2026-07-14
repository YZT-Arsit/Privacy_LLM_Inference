# Integrity validation

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

Status: **PASS**

- Registered artifacts recomputed: 78; mismatches: 0.
- Each shadow: 1,000 rows, 500 members, 500 nonmembers, exact V0/V2/index join, and 610 ordered features.
- `hidden.final.last_l2`, prohibited metadata, membership labels, sample IDs, and trusted session secrets are absent from feature matrices and the registered handoff.

| Shadow bundle | Inventory SHA-256 | Adapter/membership/schema/collection gates |
|---|---|---|
| `results/aaai_private_base/authorized_security_eval/mia_confound_controlled_v3/real_tdx_three_shadow_v1/shadow_1` | `02d4a8e2aecb2f0884bbf9667f9c27ce021d22181547b7bc1023c443406aecd1` | PASS |
| `results/aaai_private_base/authorized_security_eval/mia_confound_controlled_v3/real_tdx_three_shadow_v1/shadow_2` | `ec0d3e096d1cab0d88f48dc901cd3e9ac8e0dbed7eb352150eb595cf88d63035` | PASS |
| `results/aaai_private_base/authorized_security_eval/mia_confound_controlled_v3/real_tdx_three_shadow_v1/shadow_3` | `359653fefdf99198788aeef1ffe1b79fb320a1e5ae41e4032b2d303b38922789` | PASS |
