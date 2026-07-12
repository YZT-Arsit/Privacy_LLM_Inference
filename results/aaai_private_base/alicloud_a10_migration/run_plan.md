# A10+TDX migration run plan

Target: reproduce the validated private-base O1-C LoRA direct-transport stack on Alibaba A10 (172.30.25.154)
+ TDX (172.30.25.153), private VPC data plane only, Mac control-plane only. Nothing committed.

- P0 provenance freeze (HEAD d271ef61, source-bundle 09068b38, package root bfd578b8 == pinned).
- P1 private-net preflight: PASS (RTT 0.143ms, MTU 8500, 0% loss, direct eth0).
- P2 throughput gate: PASS (~170 down / ~182 up MB/s, both >= 100).
- P4 sync validated code + immutable package + bundles; hash-compare (halt on mismatch).
- P5 A10 env: torch(cu128)+transformers+safetensors; checkpoint fetch + sha verify (88c14255...); bf16 smoke.
- P6 deploy persistent private transport (TDX forced-command service on 172.30.25.153; A10 resident runner).
- P7 one-step L10 BF16 regression vs H800/ferried/plaintext; require all security counters 0.
- P8 ten-step gates: L10 SGD, L11 momentum, L12 AdamW(1+10).
- P9 resume deferred 50-step matrix (only if gates+throughput pass).
- P10 SST-2 then GSM8K utility (L0/L5/L12, 3 seeds).
- P11 collect+hash+archive; P12 secure TDX cleanup; P13 optional A10 cleanup.

Tolerances: A10 vs H800 NOT required bitwise-identical; require algorithmic + task-trajectory alignment
within preregistered BF16 tolerances. Gate to proceed to long runs: both dirs >= 20 MB/s (PASSED at ~170+).
