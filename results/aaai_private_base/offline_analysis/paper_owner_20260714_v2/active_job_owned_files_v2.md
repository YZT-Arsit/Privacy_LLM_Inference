# Active-Job-Owned Files (read-only snapshot)

Snapshot basis: the shared live registry at `2026-07-14T15:26:36+08:00` reported T2 and T3
running and T4 pending behind the T2 scheduler. These files and path families are excluded from
offline writes.

## T2 — `e2e_T2_qkvo_s1234_protected`

- `results/aaai_private_base/generative_lora/protected_runs/e2e_T2_qkvo_s1234_protected.*`
- `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_T2_qkvo_s1234_protected.jsonl`
- `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_T2_qkvo_s1234_protected.jsonl.profile.json`
- `results/aaai_private_base/generative_lora/extension/metrics/t2_qkvo_s1234_protected/**`
- remote log `/root/pllo_pb/g2_e2e_T2_qkvo_s1234_protected.log`

## T3 — `e2e_T3_mlp_s1234_protected`

- `results/aaai_private_base/generative_lora/protected_runs/e2e_T3_mlp_s1234_protected.*`
- `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_T3_mlp_s1234_protected.jsonl`
- `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_T3_mlp_s1234_protected.jsonl.profile.json`
- `results/aaai_private_base/generative_lora/extension/metrics/t3_mlp_s1234_protected/**`
- remote log `/root/pllo_pb/g2_e2e_T3_mlp_s1234_protected.log`

## T4 — `e2e_T4_all7_s1234_protected` (pending scheduler)

- `results/aaai_private_base/generative_lora/protected_runs/e2e_T4_all7_s1234_protected.*`
- `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_T4_all7_s1234_protected.jsonl`
- `results/aaai_private_base/generative_lora/generations/g2_protected_e2e_T4_all7_s1234_protected.jsonl.profile.json`
- `results/aaai_private_base/generative_lora/extension/metrics/t4_all7_s1234_protected/**`
- `results/aaai_private_base/generative_lora/extension/monitor/t4_scheduler_heartbeat.json`

The shared `live_job_registry.json`, active scheduler script, driver scripts, logs, sessions,
counters, adapters, generations, results, attestations, and terminal manifests remain owned by the
GPU execution owner. This offline session performs only read-only admission checks after completion.
