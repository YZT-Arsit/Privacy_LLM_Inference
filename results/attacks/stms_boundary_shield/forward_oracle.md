# STMS forward-matching oracle

## Candidate-set (16 same-length)

| case | top1 | true_rank | true_resid | margin |
|---|---|---|---|---|
| baseline | True | #1/16 | 1.17e-14 | 8.72e-04 |
| orthogonal | True | #1/16 | 1.22e-15 | 3.41e-04 |
| nonorthogonal | False | #8/16 | 6.95e-01 | 5.34e-07 |
| blocklocal | True | #1/16 | 1.55e-16 | 3.41e-04 |

## Sensitive-token top1 (templates)

| case | sensitive_top1 |
|---|---|
| baseline | 1.00 |
| orthogonal | 1.00 |
| nonorthogonal | 0.00 |
| blocklocal | 1.00 |

- baseline uses the Procrustes/Gram oracle; STMS cases use the spectrum oracle (mask-agnostic).
