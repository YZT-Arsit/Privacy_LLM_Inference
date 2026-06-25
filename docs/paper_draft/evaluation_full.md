# Full evaluation (E1–E13) — consolidated scaffold

This is the master evaluation section. The **numeric tables are generated** from
the real experiment JSONs by `scripts/run_e13_final_evaluation_report.py`, which
re-infers deployment truth and takes paper claims only from the overclaim-refusing
validator (`scripts/validate_paper_claims.py`). Until the next real H800/TDX
session is run, the generated tables will be labeled `dry_run` / not
`paper_ready`; this scaffold documents the structure, the regeneration command,
and the fixed limitations.

## Regenerate

```
python scripts/run_e13_final_evaluation_report.py \
  --correctness-json outputs/qwen7b_folded_full_decode_probe.json \
  --correctness-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \
  --correctness-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --e3-json outputs/e3_remote_decode_scaling.json \
  --e4-json outputs/e4_setup_cost.json \
  --e5-json outputs/e5_final_comparison.json \
  --e8-json outputs/e8_lora_final_report.json \
  --e9-json outputs/e9_mmlu_tdx_attested.json \
  --e9-json outputs/e9_gsm8k_tdx_attested.json \
  --e10-json outputs/e10_lora_utility.json \
  --latency-json outputs/e12_latency_baselines.json \
  --security-negative-json outputs/security_negative_tests.json \
  --output-json outputs/e13_final_evaluation.json \
  --output-md   docs/paper_draft/evaluation_full.md
```

(The command overwrites this file with the generated version. Keep this scaffold
in version control as the template.)

## Tables produced

1. **Correctness** — per decode run: `tokens_exact_match`, `token_match_rate`,
   `allclose`, `audit_passed`, with the run's `dry_run` label. Sources: E1/E2
   local correctness + decode scaling, E5 no-LoRA comparison, the TDX-attested
   no-LoRA decode, and the folded-LoRA decode.
2. **Public task utility preservation** — E9 per dataset/backend: `metric_value`
   (accuracy / numeric_exact_match / macro_f1 / rouge_l), `num_examples`,
   `paper_ready`. Recommended cost-controlled subsets: MMLU 200–500, GSM8K
   100–200, BoolQ 200–500, AG News/SST-2 500–1000 (see
   `docs/runbooks/PUBLIC_BENCHMARK_PREPARATION.md`).
3. **LoRA utility preservation** — E10: base vs plaintext-LoRA vs folded-LoRA
   metric, `lora_gain_preserved_ratio`, `folded_lora_preserves_gain`,
   `utility_preserved` (= gain preserved AND security clean).
4. **Security audit matrix** — per run: `worker_has_mask_secrets`,
   `worker_has_raw_lora`, `tee_used_on_gpu`, `gpu_visible_plaintext_fields`,
   `leaked_secret_fields`, `audit_passed`.
5. **Security negative tests** — the 14 deliberately-broken cases that MUST be
   detected (mask-secret leak, plaintext input_ids, raw-LoRA/optimizer/training in
   the package, hash mismatches, attestation report_data/mr_td/stale-binding
   mismatch, labels/recovered-logits in the transcript).
6. **Deployment truth** — per result file: `gpu_real`, `tee_real`, `tee_type`,
   `attestation_verified`, `runtime_hash_bound`, `lora_enabled`, `boundary_mode`
   (so no run is described as more than it is).
7. **Latency / overhead** — E12 per backend: total latency, latency/token,
   tokens/s, `overhead_vs_plaintext_h800`, peak GPU memory. (`--output-tex`
   emits a paper-ready LaTeX table.)
8. **Setup / provisioning cost** — folded package size (F32), setup/load time,
   embedding artifact size, folded-LoRA package size, decode-latency overhead.
9. **Supported paper claims** — from the claim validator: supported vs
   unsupported claim classes + overclaim risks. Dry-run/fixtures never back a
   real-deployment claim; no-LoRA never backs a LoRA claim; non-attested never
   backs a TDX-attested claim; synthetic-LoRA never backs real-adapter utility;
   `production_ready_serving` stays unsupported.
10. **Limitations** — see below (always stated).

## Limitations

1. HTTP/SSH tunnel transport is a research prototype, not production transport.
2. The TDX-attested no-LoRA deployment has been validated; the LoRA attested real
   run must be separately validated after server restart.
3. E7 is a minimal private LoRA update prototype, not full GPU-offloaded private
   LoRA training.
4. Public benchmark subsets are used for cost-controlled utility validation; full
   benchmark scaling is future work / optional.
5. The boundary embedding artifact contains trusted secrets and must remain inside
   the TDX guest.
6. The folded package currently stores F32 folded operators for numerical
   fidelity.
