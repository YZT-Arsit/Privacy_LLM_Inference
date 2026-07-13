# Minimum critical path

1. Validate and pull G0/G1 evidence; freeze exact source/runtime hashes without rerunning.
2. Resolve the FP32 frozen-contract versus BF16 protected-runner discrepancy and preserve training counters separately from generation counters.
3. Launch uniquely named `e2e_L12_s1234_full` for 750 steps with durable log/job registry/checkpoint metadata.
4. Export the transformed adapter package and execute all fail-closed negative controls.
5. Run full 500-example protected generation in a fresh process, then a second-process frozen-subset reproducibility check.
6. Compute quality metrics, paired G2-G1 differences, bootstrap CI and measured costs.
7. Complete the minimal q+v versus all-seven validation ablation without delaying the main cell.
8. Build paper tables, claim-evidence matrix, limitations and final partial/complete status.

The first paper-facing launch is blocked until step 2 is resolved.
