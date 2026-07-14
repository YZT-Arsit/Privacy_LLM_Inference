# Unified Frozen-Artifact Cost Table

No latency measurement was run. Training wall is the recorded G1 wall or the sum of frozen G2 per-step walls; total combines training and the matching 500-sample generation.

| Method | Seed | Train h | Generation min | Total h | tok/s | TDX train calls | TDX decode calls | Adapter MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1_plaintext | 7 | 0.089 | 8.44 | 0.230 | 32.80 | 0 | 0 | 16.82 |
| G2_protected_L12 | 7 | 5.479 | 6.53 | 5.587 | 41.01 | 2252 | 16556 | 16.88 |
| G1_plaintext | 1234 | 0.089 | 8.39 | 0.229 | 33.10 | 0 | 0 | 16.82 |
| G2_protected_L12 | 1234 | 5.475 | 6.22 | 5.578 | 40.73 | 2251 | 15694 | 16.88 |
| G1_plaintext | 2025 | 0.089 | 8.54 | 0.231 | 32.30 | 0 | 0 | 16.82 |
| G2_protected_L12 | 2025 | 5.633 | 6.69 | 5.745 | 40.81 | unavailable_counter_not_local | 16870 | 16.88 |
