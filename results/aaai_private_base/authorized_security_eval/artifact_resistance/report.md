# Protected artifact misuse-resistance evaluation

All fixtures are researcher-owned. No external endpoint or authentication bypass is used.

Authorized transformed-package restart: 500 rows, token/text/hash agreement 100.0%, attestation A/B true.

| Condition | Status | Current evidence |
|---|---|---|
| R0 | PENDING_MEASUREMENT | package-only initialization/forward test not yet run |
| R1 | PENDING_MEASUREMENT | runtime-without-boundary forward test not yet run |
| R2 | PENDING_MEASUREMENT | meaningful-output test without TDX-only mapping not yet run |
| R3 | COMPLETE_VERIFIED | invalid local diagnostic boundary is refused before protected execution |
| R4 | COMPLETE_VERIFIED | wrong run/session HMAC, replay, recovery, and counters passed |
| R5 | COMPLETE_VERIFIED | all configured mismatch controls rejected |

The stage remains partial until R0, R1, R2 are measured independently.
