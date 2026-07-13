# Incomplete cells at takeover

| Cell | Status | Missing evidence/work |
|---|---|---|
| G2 exact protected seed 1234, 750 steps | PARTIAL | Running as `e2e_L12_s1234_full-1783933686-49d8be`; full durable log, final artifact, frozen training counters and adapter hash pending. |
| G2 transformed adapter package | NOT_STARTED | Manifest, safetensors, tensor hashes, provenance and fail-closed negative controls. |
| G2 fresh-process full generation | NOT_STARTED | 500 direct test generations from package in a new process. |
| G2 restart reproducibility subset | NOT_STARTED | Second new process and exact frozen-subset comparison. |
| Main metrics and paired G2-G1 | NOT_STARTED | Aggregate/per-example/paired/bootstrap artifacts. |
| Core cost report | PARTIAL | G0/G1 generation and smoke counters exist; full G2 training/generation/handoff costs absent. |
| Minimal T1 vs T4 target ablation | NOT_STARTED | Validation-only contrast and state/size/cost accounting. |
| Final paper tables/claims/limitations | NOT_STARTED | Must be built only from verified cells. |
| G3 native transformed LoRA | NOT_APPLICABLE | Optional and cannot delay mandatory G0/G1/G2 path. |
