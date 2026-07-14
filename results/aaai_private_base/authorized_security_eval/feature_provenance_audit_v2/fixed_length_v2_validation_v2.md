# Fixed-length corrected V2 validation

**Validation: PASS.**

Two captures use the same 77 sample IDs, exact effective length 61, batch size 1, no padding, fp32, identical package/adapter/query/code path. The released matrix has 611 invariant features and excludes 96 positional-coordinate features plus all identifiers.

| Family | Aligned source AUC | Shuffled labels | Shuffled mapping |
|---|---:|---:|---:|
| attention | 0.5000 | 0.5000 | 0.5000 |
| kv | 0.5000 | 0.5000 | 0.5000 |
| hidden | 0.5000 | 0.5000 | 0.5000 |
| logit | 0.5000 | 0.5000 | 0.5000 |
| combined | 0.5000 | 0.5000 | 0.5000 |

Maximum paired feature difference: `0.0`. Train/test sample-ID overlap: 0. This validation tests collection-source confounding only; it is not a privacy or MIA result.
