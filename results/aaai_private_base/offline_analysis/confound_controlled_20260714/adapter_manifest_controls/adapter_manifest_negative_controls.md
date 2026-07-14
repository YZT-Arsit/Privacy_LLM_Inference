# Adapter Manifest Negative Controls

| Test | Expected accept | Actual accept | Harness pass |
|---|---:|---:|---:|
| valid | True | True | True |
| wrong_pool | False | False | True |
| wrong_membership | False | False | True |
| wrong_checkpoint | False | False | True |
| wrong_shadow | False | False | True |
| training_mode | False | False | True |
| training_transcript_mixed | False | False | True |
| label_in_features | False | False | True |
| runtime_metadata_in_features | False | False | True |
| missing_adapter_hash | False | False | True |
