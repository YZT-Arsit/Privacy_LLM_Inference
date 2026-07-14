# Reproducibility Audit (v2)

- Canonical generation inputs are selected by logical ID through `canonical_generation_artifact_registry.json`.
- Size and SHA-256 are verified against `canonical_hash_allowlist.json` before JSON parsing; mismatch fails closed.
- The damaged and verified G2 seed-1234 copies were not overwritten.
- Corpus BLEU/chrF bootstrap replicates resample matched IDs and recompute full corpus scores from exact additive sufficient statistics.
- ROUGE-L aggregation is an explicitly documented arithmetic mean of per-example maximum-reference LCS F1.
- Target ingestion is append-only and admits only 750-step, 500-generation, finite, attested, hash-verified bundles; smoke runs are excluded.
- Corrected MIA preprocessing is fit inside the attack-training fold; labels remain outside collected feature records.
- No GPU/TDX action, shared-registry write, active-job script edit, or commit is performed by these tools.
- The ingestion ledger contains an initial T1 observation and a provenance-only corrected T1 event from the idempotence/schema repair; the latest JSON snapshot is canonical.
