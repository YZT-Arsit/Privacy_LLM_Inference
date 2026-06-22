# Stage 5.8 -- Lookup Nonlinear Cost Proxy

## 1. Experiment Scope

We compare the current compatible SwiGLU nonlinear island against a finite-domain lookup-style SwiGLU proxy. The goal is paper-grade *cost* comparison only. No secure lookup, garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic protocol is implemented. This is a lookup cost proxy, not a secure lookup implementation.

Workload: `batch_size=1`, `seq_len=128`, `intermediate_size=11008`, `num_layers=1`, `num_tables_policy=per_layer_shared`.

## 2. Threat Model and Non-Claim

Honest-but-curious cloud accelerator. The current compatible island is faster and lower-memory but preserves permutation-invariant activation statistics. Lookup-style nonlinear protection may improve value hiding, but this stage evaluates only table-size and memory-access costs. No formal, cryptographic, or semantic security is claimed. No real TEE or GPU wall-time is measured.

## 3. Current Compatible SwiGLU Island Cost

- SiLU ops: `1,409,024`
- Multiply ops: `1,409,024`
- Online read bytes (G + U): `11,272,192` (10.75 MiB)
- Online write bytes (A): `5,636,096` (5.38 MiB)
- Online memory bytes: `16,908,288` (16.12 MiB)
- Table preprocessing bytes: `0`
- Known leakage: `permutation_invariant_statistics_preserved`

## 4. Lookup-style SwiGLU Proxy Cost

For a `b`-bit quantized binary SwiGLU lookup:
```
  table_entries        = 2^(2 * b)
  table_bytes          = table_entries * entry_bytes
  num_lookups          = batch_size * seq_len * intermediate_size
  online_lookup_bytes  = num_lookups * entry_bytes
  preprocessing_bytes  = table_bytes * num_tables
  per_channel_table_bytes = table_bytes * intermediate_size  (impractical_proxy_only)
```

| method | bit_width | table_entries | table_bytes | preprocessing_bytes | online_lookup_bytes | per_channel_table_bytes |
|---|---|---|---|---|---|---|
| `lookup_swiglu_proxy_4bit` | 4 | 256 | 512 (512.00 B) | 512 (512.00 B) | 2,818,048 (2.69 MiB) | 5,636,096 (5.38 MiB) |
| `lookup_swiglu_proxy_6bit` | 6 | 4,096 | 8,192 (8.00 KiB) | 8,192 (8.00 KiB) | 2,818,048 (2.69 MiB) | 90,177,536 (86.00 MiB) |
| `lookup_swiglu_proxy_8bit` | 8 | 65,536 | 131,072 (128.00 KiB) | 131,072 (128.00 KiB) | 2,818,048 (2.69 MiB) | 1,442,840,576 (1.34 GiB) |

## 5. Table Size Scaling

| bit_width | table_entries | table_bytes |
|---|---|---|
| 4 | 256 | 512 (512.00 B) |
| 6 | 4,096 | 8,192 (8.00 KiB) |
| 8 | 65,536 | 131,072 (128.00 KiB) |

## 6. Online Lookup Bandwidth

Online lookup bandwidth grows linearly with `num_lookups = batch_size * seq_len * intermediate_size`. Table preprocessing bandwidth grows as `2^(2b) * entry_bytes * num_tables`. Per-channel tables are reported only as `impractical_proxy_only`.

| method | online_lookup_bytes | preprocessing_bytes |
|---|---|---|
| `lookup_swiglu_proxy_4bit` | 2,818,048 | 512 |
| `lookup_swiglu_proxy_6bit` | 2,818,048 | 8,192 |
| `lookup_swiglu_proxy_8bit` | 2,818,048 | 131,072 |

## 7. CPU Microbenchmark

Microbench workload: `batch_size=1`, `seq_len=128`, `intermediate_size=1024`, `repeats=10`. This uses ordinary CPU memory lookup only -- not a secure lookup primitive.

| method | mean_ms | median_ms | std_ms |
|---|---|---|---|
| `compatible_swiglu_island_current` | 0.0744 | 0.0716 | 0.0054 |
| `lookup_swiglu_proxy_4bit` | 0.1522 | 0.1504 | 0.0159 |
| `lookup_swiglu_proxy_6bit` | 0.1397 | 0.1402 | 0.0045 |
| `lookup_swiglu_proxy_8bit` | 0.1258 | 0.1271 | 0.0084 |

## 8. Security / Cost Interpretation

- The current compatible island has zero table preprocessing bytes but preserves permutation-invariant activation statistics; its security profile is `lightweight_correctness_preserving_proxy_evaluated_not_formal`.
- The lookup proxy has table-size cost growing as `2^(2b)`. Its `security_potential` is `stronger_value_hiding_if_combined_with_secure_lookup_protocol`. Its `implemented_security` is `none_cost_proxy_only`.
- `formal_security_claim` = `False`
- `cryptographic_lookup_implemented` = `False`
- `recommended_use` = `cost-baseline-and-future-work-motivation`

## 9. Limitations

- This is a CPU-only cost proxy; no real TEE or GPU wall-time is measured.
- No secure lookup, garbled circuit, MPC, FHE, Tabula, or FLUTE protocol is implemented; the microbenchmark uses ordinary CPU memory lookup.
- The current compatible island preserves permutation-invariant activation statistics; lookup-style protection could close this leakage channel but only if combined with a real cryptographic lookup protocol.
- Per-channel tables are reported as impractical_proxy_only to make the bandwidth cost explicit; they are not evaluated in the microbenchmark.
- Bit widths above 8 are not microbenchmarked because the table size grows as 2^(2b); a 12-bit table already has 16M entries and is reported for cost only.
- No formal, cryptographic, or semantic security is claimed.

## 10. Next Stage Plan

Stage 5.8 produces a cost baseline. A future stage could (i) integrate a real secure lookup primitive (e.g. Tabula-style 2-PC) and validate correctness on a small model; (ii) explore mixed designs that keep the compatible island for hot paths and use lookup only at designated boundary tensors; (iii) extend the cost model to multi-layer / multi-head workloads. None of these are implemented in Stage 5.8.

## Honesty phrases (verbatim)

- This is a lookup cost proxy, not a secure lookup implementation.
- No garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic lookup protocol is implemented.
- Lookup-style nonlinear protection may improve value hiding, but this stage evaluates only table-size and memory-access costs.
- The current compatible island is faster and lower-memory but preserves permutation-invariant activation statistics.
- No formal, cryptographic, or semantic security is claimed.
- No real TEE or GPU wall-time is measured.

`formal_security_claim`: `False`

`cryptographic_lookup_implemented`: `False`

