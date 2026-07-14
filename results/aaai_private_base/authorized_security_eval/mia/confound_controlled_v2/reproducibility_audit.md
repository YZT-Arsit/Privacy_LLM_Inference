# Reproducibility Audit

- Candidate pool and membership assignments are deterministic and hash-bound.
- Membership is evaluator metadata and is excluded from feature files.
- Every sample changes membership state across the three planned shadows.
- Attack preprocessing is fit on attack-training partitions only.
- Planned primary split is leave-shadow-run-out with semantic/duplicate grouping.
- Existing source hashes are recorded in `source_hashes_before.sha256`.
- The local seed-1234 G2 JSONL was found truncated to exactly 524,288 bytes. Paired analysis uses
  a read-only isolated copy retrieved from the completed remote artifact and verified against the
  frozen completion-manifest SHA-256 `d3b5ad9820c13e473ee7a995480a9ea6b5b3f5a786a4ba3276774cf116e46e31`.
- No GPU/TDX work was launched during offline preparation.
