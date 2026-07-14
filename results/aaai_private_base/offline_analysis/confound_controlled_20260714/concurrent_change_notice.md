# Concurrent Repository Change Notice

- The analysis session began from Git commit `0b8ab1c`.
- During the CPU-only analysis, repository `HEAD` advanced externally to commit `9f6f7c9`
  (`2026-07-14 15:11:54 +08:00`, author `Zhaoting Yang`).
- This analysis session did not create a commit.
- T3 protected-run files appeared or changed concurrently. They were treated as externally owned
  active-job artifacts and were not read for conclusions, edited, stopped, restarted, or removed.
- The modified `paper_table_template.md`, the adapter-manifest template, and the three-seed method
  summary are intentional isolated CPU-analysis outputs created after or across that external commit.
- The shared live-job registry and experiment queue were not modified by this analysis session.

This notice preserves the ownership boundary for interpreting `git status`; it is not an experiment
result and does not alter any reported metric.
