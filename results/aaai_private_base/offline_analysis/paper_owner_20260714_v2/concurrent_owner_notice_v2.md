# Concurrent Owner Notice (v2)

During this offline session, the shared live registry changed externally from the earlier v1 snapshot
(`updated_at` `2026-07-14T09:36:12+08:00`) to v2 (`updated_at`
`2026-07-14T15:26:36+08:00`). The v2 registry identifies T2 and T3 as running and T4 as pending
under a scheduler. This offline session did not write either registry version.

Repository `HEAD` is `ec21c37` (`2026-07-14T15:22:57+08:00`, author `Zhaoting Yang`). No commit
was created by this offline session.

The following non-isolated paths also appeared or changed concurrently and are not outputs of this
offline owner:

- `results/aaai_private_base/generative_lora/extension/metrics/t1_qv_s1234_protected/**`;
- `results/aaai_private_base/generative_lora/extension/monitor/**`;
- `results/aaai_private_base/generative_lora/extension/paired_statistics_corrected_v3_chrf/**`;
- `scripts/generative_lora/audit_paired_corpus_chrf_v1.py`;
- `scripts/generative_lora/schedule_t4_after_t2.sh`.

The independently generated corpus-chrF audit agrees with this bundle's point estimates. Its
bootstrap endpoints differ slightly because it uses 20,000 replicates and a different deterministic
random stream; this bundle uses 10,000 replicates consistently across all metrics.
