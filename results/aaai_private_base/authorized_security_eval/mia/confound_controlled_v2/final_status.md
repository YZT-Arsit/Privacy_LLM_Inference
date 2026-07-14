# Final Status

`CONFOUND_CONTROLLED_MIA_PARTIAL`

Completed offline: takeover audit, invalid-cohort preservation, frozen-feature attribution,
negative-control code/results, same-pool construction, adapter-manifest guards, paired G1/G2
statistics with bootstrap intervals, active-log parsing, cost table, ObfuscaTune fidelity preparation,
and paper/claim templates.

Active unrelated jobs: see `active_job_snapshot.md`.

Prepared but not launched: three shadow LoRA jobs, matched C0 V0/V2 collection, leave-shadow-run-out attack.

Missing corrected cells: 3 shadow adapters; 3×1,000 V0 outputs; 3×1,000 V2 C0 records; corrected controls and CIs.

Estimated resource: approximately 17–18 GPU/TDX hours. No device was idle at takeover.

Next safe command remains CPU-only:

```bash
python3 scripts/authorized_security/mia_confound_controlled/validate_prepared_shadows_v1.py   --root results/aaai_private_base/authorized_security_eval/mia/confound_controlled_v2/shadow_runs
```

No launch script is implemented during offline-only preparation; GPU/TDX execution remains fail-closed.
