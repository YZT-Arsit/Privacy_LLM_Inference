# Claim → evidence matrix (verified against frozen artifacts)

| Paper claim | Evidence artifact | sha256 | status |
|---|---|---|---|
| Exact masked-forward correctness (fp64 < 1e-8) | `private_package/build_validation.json` | ec264bf677b7 | ✅ verified |
| TDX trusted optimizer exact (SGD/momentum/AdamW) | `full_lora_matrix/** , gate0_d4/**` | dir | ✅ verified |
| Protected LoRA preserves utility (SST-2 L12≈L5 +0.0027) | `PHASE7_SST2_UTILITY_SUMMARY.json` | 892952e1aea7 | ✅ verified |
| Protected==plaintext downstream generation (GSM8K M1≡M2 gap 0) | `gsm8k_lora_eval.json` | b08f2d430e36 | ✅ verified |
| Empirical security resistance S1–S6 (controls pass) | `security_run_manifest.json` | 82bdd6407cf1 | ✅ verified |
| Component contributions isolated (ablation) | `ablation_results.json` | 54dd06d06510 | ✅ verified |
| Efficiency: 2 vs 224 TEE crossings, ≤1.07× GPU-TEE | `obfuscatune_latency_h800.json` | cc209f8ae636 | ✅ verified |
| Real A10 GPU + Intel TDX (attested) | `alicloud_a10_migration/**` | dir | ✅ verified |
| 7B generation/fold + attacks (scale evidence, reused) | `baselines/conjformer/qwen7b_generate_fp32.json` | d05231eb48c6 | ✅ verified |

**Unsupported / deferred (must NOT be claimed):** formal/zero-leakage/information-theoretic security; LoRA task-accuracy superiority; empirical 7B protected-LoRA training; GSM8K reasoning gains at 0.5B; attention-score confidentiality.