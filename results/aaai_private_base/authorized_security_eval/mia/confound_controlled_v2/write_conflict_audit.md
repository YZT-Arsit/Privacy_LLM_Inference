# Write-Conflict Audit

- Active T1 and T2 remote logs, drivers, sessions, adapters, and result paths are immutable.
- `generative_lora/extension/live_job_registry.json` and `experiment_queue.json` are immutable.
- Existing G0/G1/G2, target-ablation, and V2 MIA artifacts are immutable.
- Existing untracked paired-statistics files are owned by another workflow and are untouched.
- New writes are restricted to `mia/confound_controlled_v2/`, the isolated offline-analysis root,
  and new versioned scripts under `scripts/authorized_security/mia_confound_controlled/`.
- No active-job script is modified by this analysis.
