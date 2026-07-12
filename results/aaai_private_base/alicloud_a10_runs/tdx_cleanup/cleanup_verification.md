# TDX secure cleanup verification

All verified AFTER archives hashed (PHASE 11). Preserved attestation evidence to Mac BEFORE deletion.

| check | result |
|---|---|
| temporary keys removed (a10-to-tdx-prod, h800-direct-tdx) | 0 remain |
| removed key rejected | Permission denied (verified from A10) |
| session-key / gamma / labels files in /tmp | 0 |
| tensor dumps (logits/dlogits/grads/corr) in /tmp | 0 (152 removed) |
| experiment processes (persistent/bench/gate2/correction) | 0 |
| temporary listeners (18091) | 0 |
| management key (skp-*) | intact |
| /dev/tdx_guest + base env | preserved |
| attestation quotes/appraisal/report_data | preserved to Mac attestation_evidence/ |

Did NOT remove: TDX base env, system packages, management SSH config, durable reports, unrelated files.
