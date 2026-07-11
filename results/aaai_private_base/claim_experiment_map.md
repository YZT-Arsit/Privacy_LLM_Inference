# Claim -> experiment map (Gate 0)

| claim | experiment(s) | artifact |
|---|---|---|
| private base-weight confidentiality (empirical, no crypto claim) | D2 T0-T3 + positive controls | gate0_d2/go_no_go.md |
| masked operator fwd/bwd exact on real Qwen | D3 | gate0_d3/summary.md |
| GPU package holds no plaintext | PACKAGE scans | private_package/plaintext_absence_scan.json |
| protected unified 1-step aligns plaintext under real TDX | D4 | gate0_d4/ |
| protected 10-step aligns plaintext | POST_D4_10STEP | gate0_post_d4_10step/ |
| protected adapter deployable+useful in same package | DEPLOY_SMOKE | lora_deployment_smoke/ |
| training semantics == plaintext (full) | PHASE 8 L1-L8 | (post Gate 0) |
| downstream utility parity | PHASE 9 | (post Gate 0) |
| generation+LoRA share one package | PHASE 10 provenance | (post Gate 0) |
| tied embedding/LM-head does NOT ease weight recovery | D2 tied-view cross-attack | gate0_d2/bridge_learning.csv |

Non-claims: no cryptographic security, no zero-leakage, no attention-score confidentiality.

## Counter trust-domain boundary (correction 1)
Plaintext-materialization counters are TRUST-DOMAIN-SCOPED. The trusted diagnostic
oracle and trusted packager legitimately load plaintext (counters may be nonzero,
honestly recorded). The security claim concerns ONLY the untrusted protected worker
(`untrusted_worker_*` = 0). Local D3 has no untrusted worker ->
`untrusted_worker_counters_applicable=false`, `protected_operator_plaintext_shortcuts=0`.
Do not fabricate zero counters for a component that was not executed.
