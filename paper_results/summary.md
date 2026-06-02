# Stage 7.5 — Paper Artifact Summary

_This is the consolidated paper-side summary of Stage 1 → 7.4. No new ops, no new attacks; pure aggregation of ``outputs/*.json``. **No real TEE wall-time, no formal / cryptographic / semantic security claims.**_

## 1. Artifact Inventory

- Total artifacts surveyed: **22**
- Present: **22**
- Missing: **0**
- By slot: inference=11, lora=11

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
- `plain_lora_forward` (plain_rank_r): mean = **0.008 ms**, median = 0.008 ms, std = 0.001 ms, repeats=5.
- `masked_lora_forward` (fresh_masks_fresh_u_with_pad): mean = **0.263 ms**, median = 0.263 ms, std = 0.011 ms, repeats=5.
- `masked_lora_backward` (fresh_masks_fresh_u_with_pad): mean = **0.118 ms**, median = 0.117 ms, std = 0.003 ms, repeats=5.
- `rank_padded_lora_forward` (paired_cancellation_dummy): mean = **0.270 ms**, median = 0.267 ms, std = 0.008 ms, repeats=5.
- `multi_layer_lora_training_step` (synthetic_tile): mean = **4.684 ms**, median = 4.685 ms, std = 0.084 ms, repeats=5.
- `modern_decoder_model_wrapper` — _skipped_: modern_decoder_wrapper is opt-in (include_modern_decoder_wrapper=False)
See `paper_results/markdown/measured_runtime.md` / `paper_results/latex/measured_runtime.tex`.

## 5. LoRA Training Summary

- Rows: **11**
See `paper_results/markdown/lora_training_summary.md`.

## 6. Limitations

- Aggregated limitation rows: **161**
See `paper_results/markdown/limitations_summary.md`.
Recurring themes: no formal / cryptographic / semantic security; no real TEE wall-time; padded_rank still visible; optimizer / loss remain trusted; no PEFT integration; no hardware side-channel evaluation; no full Qwen / TinyLlama LoRA fine-tuning.

## 7. Claims Audit

- Status counts: proxy_supported=5, supported=8, unsupported=8
See `paper_results/markdown/paper_claims_audit.md` / `paper_results/latex/paper_claims_audit.tex`.

## 8. Missing Artifacts

_All registered artifacts are present._

## 9. Next Paper-Writing Plan

- Draft the system-model + threat-model sections using `paper_claims_audit.md` (unsupported claims must NOT appear as guarantees).
- Draft the correctness theorem section as empirical verification over the per-stage `correctness_summary` rows; do NOT make formal proof claims.
- Draft the security evaluation section from `security_proxy_summary` + `lora_training_timing_proxy.json`. Always word as 'proxy-evaluated', never 'secure'.
- Draft the experiment section by interleaving the consolidated CSV / Markdown / LaTeX tables + the `paper_results/figures/` PDFs / PNGs; keep the 'this is local emulation' disclaimer in the runtime caption.
