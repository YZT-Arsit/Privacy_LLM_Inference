# Real-TDX three-shadow collection validation

Status: **PASS** (`REAL_TDX_THREE_SHADOW_COLLECTION_COMPLETE`).

Each shadow contains exactly 1,000 rows (500 members / 500 nonmembers), 610 fp32 features, a matched V0 output, a complete sample-ID join, valid fresh TDX attestation, zero silent fallbacks, and no disjoint member/nonmember ranges.

| Shadow | Wall time (s) | Decode calls | Shuffle AUC | Pseudo AUC | Top univariate symmetric AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 631.3 | 27577 | 0.514668 | 0.474384 | 0.558084 |
| 2 | 644.9 | 28022 | 0.507780 | 0.470124 | 0.537920 |
| 3 | 623.4 | 27033 | 0.482964 | 0.469688 | 0.557444 |

Joint gates: all passed. The joint top univariate symmetric AUC is 0.518211; mean shuffle AUC is 0.501804; mean pseudo-membership AUC is 0.471399; disjoint-range count is 0.

The shadow-identity diagnostic reaches 1.0 accuracy, but the matrix contains no run/session/shadow metadata. The separation is therefore documented as legitimate model-specific behavior and does not invalidate collection.

The final paper-facing MIA classifier was not run in this collection session.
