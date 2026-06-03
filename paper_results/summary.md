# Stage 7.5 — Paper Artifact Summary

_This is the consolidated paper-side summary of Stage 1 → 7.4. No new ops, no new attacks; pure aggregation of ``outputs/*.json``. **No real TEE wall-time, no formal / cryptographic / semantic security claims.**_

## 1. Artifact Inventory

- Total artifacts surveyed: **29**
- Present: **29**
- Missing: **0**
- By slot: cpu_paper=5, inference=11, lora=11, prior_work=2

See `paper_results/markdown/artifact_inventory.md`.

## 2. Correctness Summary

- Rows: **19**
See `paper_results/markdown/correctness_summary.md` / `paper_results/latex/correctness_summary.tex`.

## 3. Security Proxy Summary

- Rows: **14**
- Risk distribution: high=4, low=3, medium=1, n/a=1, needs_more_evaluation=5
See `paper_results/markdown/security_proxy_summary.md` / `paper_results/latex/security_proxy_summary.tex`.

## 4. Runtime Summary (local emulation only)

**This is local emulation, NOT real TEE wall-time.** No real sleep, no real runtime gating.
- `plain_synthetic_linear` (X W): mean = **0.002 ms**, median = 0.002 ms, std = 0.000 ms, repeats=5.
- `plain_lora_forward` (plain_rank_r): mean = **0.007 ms**, median = 0.007 ms, std = 0.001 ms, repeats=5.
- `masked_lora_forward` (fresh_masks_fresh_u_with_pad): mean = **0.260 ms**, median = 0.262 ms, std = 0.017 ms, repeats=5.
- `masked_lora_backward` (fresh_masks_fresh_u_with_pad): mean = **0.116 ms**, median = 0.114 ms, std = 0.005 ms, repeats=5.
- `rank_padded_lora_forward` (paired_cancellation_dummy): mean = **0.282 ms**, median = 0.270 ms, std = 0.026 ms, repeats=5.
- `multi_layer_lora_training_step` (synthetic_tile): mean = **4.606 ms**, median = 4.604 ms, std = 0.070 ms, repeats=5.
- `modern_decoder_model_wrapper` — _skipped_: modern_decoder_wrapper is opt-in (include_modern_decoder_wrapper=False)
See `paper_results/markdown/measured_runtime.md` / `paper_results/latex/measured_runtime.tex`.

## 5. LoRA Training Summary

- Rows: **11**
See `paper_results/markdown/lora_training_summary.md`.

## 6. Limitations

- Aggregated limitation rows: **211**
See `paper_results/markdown/limitations_summary.md`.
Recurring themes: no formal / cryptographic / semantic security; no real TEE wall-time; padded_rank still visible; optimizer / loss remain trusted; no PEFT integration; no hardware side-channel evaluation; no full Qwen / TinyLlama LoRA fine-tuning.

## 7. Claims Audit

- Status counts: proxy_supported=5, supported=8, unsupported=8
See `paper_results/markdown/paper_claims_audit.md` / `paper_results/latex/paper_claims_audit.tex`.

## 8. Missing Artifacts

_All registered artifacts are present._

## 9. Toy Task Results (CPU only)

- Tasks: **3**
  - `token_parity_classification` loss_diff=0.0 accuracy_diff=0.0 token_match_rate=1.0 allclose=True
  - `first_last_token_relation` loss_diff=1.1102230246251565e-16 accuracy_diff=0.0 token_match_rate=1.0 allclose=True
  - `next_token_toy_lm` loss_diff=8.881784197001252e-16 accuracy_diff=0.0 token_match_rate=1.0 allclose=True
See `paper_results/markdown/toy_task_summary.md` / `paper_results/latex/toy_task_summary.tex`.

## 10. Baseline Comparison (CPU only)

- Variants: **10**
- Risk distribution (proxy-derived): high=3, low=1, medium=1, needs_more_evaluation=5
See `paper_results/markdown/baseline_comparison_summary.md`.

## 11. Mitigation Ablation (CPU only)

- Rows: **16**
- Role distribution: experimental_optin=1, metadata_timing=2, security_critical=13
See `paper_results/markdown/ablation_summary.md`.

## 12. Robustness and Stability (CPU only)

- Experiments: **7**
  - `modern_decoder_synthetic_forward` trials=720 allclose_rate=1.0 max_error_p95=1.942890293094024e-16 failure_count=0
  - `kv_cache_append` trials=720 allclose_rate=1.0 max_error_p95=2.220446049250313e-15 failure_count=0
  - `nonlinear_island` trials=720 allclose_rate=1.0 max_error_p95=0.0 failure_count=0
  - `lora_forward` trials=720 allclose_rate=1.0 max_error_p95=4.055783486833775e-15 failure_count=0
  - `lora_backward` trials=720 allclose_rate=1.0 max_error_p95=4.529709940470639e-14 failure_count=0
  - `rank_padded_lora` trials=720 allclose_rate=1.0 max_error_p95=7.732703366514215e-14 failure_count=0
  - `multilayer_lora` trials=720 allclose_rate=1.0 max_error_p95=5.717648576819556e-15 failure_count=0
See `paper_results/markdown/stability_summary.md`.

## 13. CPU Runtime Completion (local emulation only)

- Rows: **96** across **12** components.
_This is CPU local trusted-runtime emulation, NOT real TEE wall-time and NOT GPU throughput._
See `paper_results/markdown/cpu_runtime_completion.md`.

## 14. Direct Prior-Work Primitive Comparison (CPU only)

- Rows: **11**
- exact_primitive_implemented=True: **6**, full_system_reproduced=True: **2** (only the two ours rows), cost_model_only=True: **4**, arithmetic_skeleton_only=True: **1**.
See `paper_results/markdown/direct_prior_work_comparison.md`.

## 15. Deployable Runtime API Validation (Local CPU only)

- Rows: **10**
- transcript_sanitized=True: **10/10**, raw_secret_leaked=True: **0**, backend=`local_cpu`.
_Local CPU backend only; real TEE / GPU backend NOT implemented_.
See `paper_results/markdown/ours_runtime_api_validation.md`.

## 16. Next Paper-Writing Plan

- Draft the system-model + threat-model sections using `paper_claims_audit.md` (unsupported claims must NOT appear as guarantees).
- Draft the correctness theorem section as empirical verification over the per-stage `correctness_summary` rows; do NOT make formal proof claims.
- Draft the security evaluation section from `security_proxy_summary` + `lora_training_timing_proxy.json`. Always word as 'proxy-evaluated', never 'secure'.
- Draft the experiment section by interleaving the consolidated CSV / Markdown / LaTeX tables + the `paper_results/figures/` PDFs / PNGs; keep the 'this is local emulation' disclaimer in the runtime caption.
