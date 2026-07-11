# D2 attack families (registered; run against the immutable real package)

Main families (bounded, realistic): spectrum-only recon, self-Gram/eigendecomp,
alternating orthogonal factorization, generalized Procrustes (anchored), FastICA
(+ second BSS if a verified impl exists), two-sided BSS, full residual-graph chain
alignment, Q/K paired-mask constraints, GQA constraints, gate/up shared-perm
constraints, RMSNorm constraints, embedding/LM-head structural constraints,
endpoint-anchor propagation, input/output/joint bridge learning, functional extraction.

## MANDATORY: tied embedding / LM-head cross-view family (correction 5)
- cross-view orthogonal alignment (E_embed_tilde vs W_lm_tilde share source E)
- embedding/LM-head joint factorization
- vocabulary-side bridge recovery
- endpoint recovery using tied-weight constraints
- T0 (no reference) / T2 (endpoint or 1/8/32 pairs) / T3 (matched reference)
- positive control P5': synthetic tied-weight package with planted masks; attack must
  recover the relation when an anchor is supplied.
Report: tied_view_constraint_improves_attack, embedding_basis_recovery,
final_residual_basis_recovery, vocabulary_bridge_recovery, functional_recovery_gain.

Attacks with no verified implementation are recorded as UNAVAILABLE, never as
success/failure. JADE only if a tested impl exists.
