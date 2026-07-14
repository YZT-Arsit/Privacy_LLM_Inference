# Offline Analysis and Paper Status (v2)

Status: **CPU WORK COMPLETE; GPU-DEPENDENT CELLS PENDING**.

Completed in this isolated bundle:

- canonical generation registry, SHA-256 allowlist, and corrupted-copy notice;
- registry-gated corpus BLEU, corpus chrF, explicit ROUGE-L, diagnostics, and 10,000-replicate paired bootstrap audit;
- append-only, strict terminal-manifest T1–T4 ingestion pipeline;
- T1 terminal ingestion with T2–T4 left pending;
- corrected-MIA schema/evaluation pipeline and CPU self-test;
- final generation, target-ablation, unified-cost, claim-evidence, limitations, artifact-index, and reproducibility v2 files.

Pending external evidence:

- matched G1 BF16;
- full terminal T2, T3, and T4 bundles;
- all three corrected shadow adapters and matched V0/V2 C0 collections;
- deployable transformed-adapter closure;
- matched ObfuscaTune-style system run.

No three-seed interval is presented as an equivalence test. The historical source-confounded MIA
AUC=1 is not used as a final MIA result.

This session performed no GPU/TDX action, no shared-registry write, no active-job script edit, and no commit.
