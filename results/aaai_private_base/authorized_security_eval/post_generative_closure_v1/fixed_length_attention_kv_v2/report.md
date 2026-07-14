# Fixed-length attention/KV audit

**View: SIMULATED_PROTOCOL_VIEW.**

The cohort contains 25 records per source, all at sequence length 51. The target below is collection source, not membership; membership results are withheld pending completed corrected shadow captures.

| View | Features | Source AUC | Label-shuffle AUC |
|---|---:|---:|---:|
| shape_only | 1 | 0.5000 | 0.5000 |
| attention | 480 | 1.0000 | 0.5033 |
| kv | 192 | 1.0000 | 0.5400 |
| hidden | 25 | 1.0000 | 0.6600 |
| logit | 9 | 1.0000 | 0.7233 |
| combined_metadata_free | 706 | 1.0000 | 0.5133 |

No source file, run/session/checkpoint identifier, train/eval mode, step/epoch, tensor-presence flag, split, label, or member field is selected. Any above-chance source AUC therefore diagnoses numerical collection/source confounding in the simulated captures; it must not be reported as MIA evidence.
