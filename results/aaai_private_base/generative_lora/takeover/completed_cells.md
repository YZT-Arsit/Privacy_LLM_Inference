# Completed cells at takeover

| Cell | Status | Evidence | Validation note |
|---|---|---|---|
| P0 official E2E data | COMPLETE_VERIFIED | `data/dataset_manifest.json`, `preprocessing_report.json`, immutable split/tensor/schedule hashes | Official sizes recorded; deterministic subset/template/tokenizer hashes present. |
| G0 base generation | COMPLETE_VERIFIED | A10 `g0_base_test.jsonl` + profile + durable combined log; local copy present | 500/500 greedy outputs parse, unique sample IDs, no invalid outputs; model/source/runtime hashes bound by profile and takeover manifest. |
| G1 seed 1234 | COMPLETE_VERIFIED | Local pulled adapter, meta, training log, 500-output generation | 750 steps; 336 safetensor tensors parse; r=8, alpha=16, seven targets; remote/local hashes match. |
| G1 seed 2025 | COMPLETE_VERIFIED | Local pulled adapter, meta, training log, 500-output generation | 750 steps; 336 safetensor tensors parse; r=8, alpha=16, seven targets; remote/local hashes match. |
| G1 seed 7 | COMPLETE_VERIFIED | Local pulled adapter, meta, training log, 500-output generation | 750 steps; 336 safetensor tensors parse; r=8, alpha=16, seven targets; remote/local hashes match. |
| G2 5-step smoke | COMPLETE_VERIFIED | `protected_runs/e2e_L12_smoke.*`, 10 protected generations, attestation | Real A10+TDX path, no silent fallback, attestation verified. Diagnostic only. |
| G2 exact protected seed 1234 | COMPLETE_VERIFIED | `protected_runs/e2e_L12_s1234_full.json`, adapter, frozen training counters, completion manifest and durable log | 750/750 finite steps; 750 trusted AdamW steps; attestation and package root verified; no silent fallback. BF16 runtime copies with FP32 TDX master state. |
| G2 fresh-process full generation | COMPLETE_VERIFIED | `generations/g2_protected_e2e_L12_s1234_full.jsonl` + profile + generation counters | 500/500 parseable unique aligned samples; greedy FP32 generation; 15,694 TDX decode calls; attestation verified; no invalid outputs or silent fallback. |
| Main G2 quality metrics | COMPLETE_VERIFIED | `metrics/final_g2/` aggregate, per-example, paired, bootstrap and analysis artifacts | G2 BLEU 65.88; paired sentence-BLEU mean delta +0.413 with 95% bootstrap CI [-0.628, 1.464]. METEOR unavailable (`nltk_error`). |
| TDX enablement | COMPLETE_VERIFIED | live kernel/CPU/device check | Kernel `5.10.134-19.3.al8.x86_64`, `tdx_guest`, `/dev/tdx_guest`. |

No completed cell above may be rerun unless validation discovers a technical invalidity.
