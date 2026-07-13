# Extended matrix dependency graph

```text
Frozen G0/G1/G2-s1234
  ├── adapter package + negative controls ──> restart A/B
  │                                             └──> T1 plaintext (already queued)
  ├── frozen config/schedules ──> G2-s7 ──> G2-s2025 ──> multi-seed statistics
  │                                                        └──> BF16-matched controls
  │                                                               └──> precision comparisons
  ├── T4 plaintext/protected reuse ─────────────────────────────────────┐
  └── T1 plaintext + T2/T3 plaintext + T1/T2/T3 protected ─────────────┴──> target Pareto

Existing ObfuscaTune code ──> fidelity audit ──> one-seed LoRA baseline (only if faithful)

All verified cells + measured profiles/counters
  └──> common cost table ──> security-interface audit ──> final paper tables/status
```

Native optimizer, 7B short gate and SAMSum remain downstream optional gates and cannot delay mandatory cells.
