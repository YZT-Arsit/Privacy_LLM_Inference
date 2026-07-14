# V2 feature provenance report

**Historical feature-source result: DIAGNOSTIC_ONLY. It is not privacy leakage evidence.**

The two frozen corpora contain 500 and 500 records but have **0 shared sample IDs**. The source label is exactly aliased with train/test split, sample-ID namespace, query-file hash, and content distribution. Run ID, adapter hash, transformed-package root, capture schema, dtype, device, and one-prefill loop are equal.

| Family | Columns | Schema allowed | Invalid | Diagnostic only | Max source AUC | Disjoint |
|---|---:|---:|---:|---:|---:|---:|
| attention | 480 | 384 | 96 | 384 | 1.0000 | 194 |
| kv | 192 | 192 | 0 | 192 | 1.0000 | 131 |
| hidden | 25 | 25 | 0 | 25 | 1.0000 | 23 |
| logit | 10 | 10 | 0 | 10 | 1.0000 | 8 |
| combined | 707 | 611 | 96 | 611 | 1.0000 | 356 |

Normalized attention argmax-position features are marked INVALID because they retain raw positional coordinates. All otherwise invariant numerical columns remain DIAGNOSTIC_ONLY until the same frozen samples are collected twice through the identical path.

Batch size is inferred as one from the capture loop but was not recorded in the profiles. No session ID is present. Raw metadata contains `split`, sample-ID namespace, run ID, adapter hash, and package root; none is permitted in the corrected feature matrix.
