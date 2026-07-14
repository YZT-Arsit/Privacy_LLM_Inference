# Target Ablation Table (v2)

Only full terminal cells are populated. Completed smoke runs are deliberately excluded.

| Cell | Status | Targets | BLEU | chrF | ROUGE-L | invalid% | repetition% | Trainable params | Adapter bytes | Trusted A/B factors | GPU-local factors | TDX optimizer state bytes | Raw bytes/direction/step | Trusted roundtrip s/step | Trusted compute | Transport-only | Step s | tok/s | Attestation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T1 | TERMINAL | q_proj+v_proj | 64.08 | 68.34 | 60.28 | 0.0 | 2.25 | 540672 | 2192735 | 48/24 | 24 | 6193152 | 2064384 | 24.075661829312644 | MISSING | MISSING | 25.56530533333333 | 51.4 | PASS |
| T2 | PENDING | q_proj+k_proj+v_proj+o_proj | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING/MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| T3 | PENDING | gate_proj+up_proj+down_proj | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING/MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| T4 | PENDING | q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING/MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

Provenance tags are stored per field in the JSON snapshot and CSV. Trainable and factor counts are MEASURED by enumerating the hash-verified adapter. TDX state and raw bytes/direction/step are PROJECTED from tensor ownership and exclude service/framing overhead. Trusted compute and transport-only time are MISSING because the terminal T1 run retained roundtrip timing but not the profiling split.
