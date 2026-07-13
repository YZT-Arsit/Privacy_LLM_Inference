# Extended matrix resource budget

Snapshot: 2026-07-14 00:00 CST.

- A10: one NVIDIA A10, 23,028 MiB VRAM; 4,799 MiB in use by restart B at snapshot; 145 GiB disk free.
- TDX: service active on the private data plane; 28 GiB disk free.
- No second GPU is available. All GPU jobs must be serialized.
- Cloud billing balance is not observable from either server and no billing API is configured; runtime planning therefore uses measured wall time and available disk, not a monetary balance.

| Cell group | Count | Estimated time/cell | Estimated total | TDX required |
|---|---:|---:|---:|---|
| Missing G2 seeds | 2 | 5.6 GPU h | 11.2 h | yes |
| G1 BF16-matched | 3 | 0.25–0.35 GPU h | 0.75–1.05 h | no |
| Missing plaintext target cells (T2/T3; T1 already queued) | 2 | 0.25–0.35 GPU h | 0.5–0.7 h | no |
| Missing protected target cells (T1/T2/T3; T4 frozen) | 3 | 5.6 GPU h | 16.8 h | yes |
| ObfuscaTune-style seed 1234 | 1 | 5–7 GPU h after audit | 5–7 h | expected yes |
| CPU packaging/statistics/tables | many | parallel | <2 h aggregate | no |

Estimated mandatory single-GPU critical path: approximately 30–36 hours. Native, 7B and second-task cells are gated and may remain N/A.
