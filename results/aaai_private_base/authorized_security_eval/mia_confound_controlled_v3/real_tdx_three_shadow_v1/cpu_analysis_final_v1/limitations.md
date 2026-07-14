# Limitations

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

- Only three shadow models are available; cross-shadow uncertainty and confidence intervals are exploratory.
- The same 1,000 identities occur across shadows; the semantic-group-disjoint sensitivity is therefore required.
- One source population is present, so a balanced source classifier is not identifiable.
- TPR@0.1% FPR is unsupported with 500 held-out nonmembers.
- Shadow identity is detectable from legitimate model behavior, although runtime identity metadata is absent.
- This evaluation does not establish zero leakage, formal privacy, information-theoretic security, equivalence, or universal MIA resistance.
