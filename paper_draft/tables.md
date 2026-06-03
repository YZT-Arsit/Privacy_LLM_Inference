# Tables

This appendix lists the canonical tables to be inserted into the paper body. Source files live under `paper_results/markdown/`, `paper_results/csv/`, and `paper_results/latex/`. The LaTeX pass (Stage 7.6b) will pull from the `.tex` files directly.

## Table 1 — Threat model and leakage boundary

- Source: hand-authored from Section 3 (`paper_draft/system_and_threat_model.md`).
- Columns: `party`, `holds`, `observes`, `protected_assets`, `allowed_leakage`, `out_of_scope`.
- Rows: trusted-side controller; untrusted GPU.

## Table 2 — Architecture coverage

- Source: `paper_results/markdown/artifact_inventory.md` + `outputs/cross_architecture_summary.json` (`architectures` field).
- Columns: `architecture`, `nonlinear_components`, `mask_families`, `kv_cache`, `wrapper_artifact`.
- Rows: GPT-2 (LayerNorm + GELU + MHA), encoder-only (BERT-style), encoder-decoder (T5-style), modern decoder (RMSNorm + SwiGLU + RoPE + GQA).
- Mark: which architecture has a model-level wrapper, which has block-level only, which has tensor-level only.

## Table 3 — Correctness summary

- Source: `paper_results/markdown/correctness_summary.md` (19 rows) and `paper_results/latex/correctness_summary.tex`.
- Columns: `stage`, `component`, `architecture`, `scope`, `metric`, `value`, `allclose`, `artifact_path`, `notes`.
- All 19 rows must be reproduced verbatim; do *not* re-classify `allclose=None` rows as `True`.

## Table 4 — Workload and cost summary

- Source: `paper_results/markdown/workload_summary.md` (6 rows) and `paper_results/latex/workload_summary.tex`.
- Columns: `method`, `architecture`, `integration_level`, `boundary_calls`, `trusted_compute`, `gpu_compute`, `preprocessing_cost`, `measured_wall_time_ms`, `projected_wall_time_ms`, `wall_time_source`, `artifact_path`.
- Make sure the caption notes: `wall_time_source = projected_from_op_counts` is *projected*, not measured; `measured` rows are measured on tiny-gpt2 only.

## Table 5 — Security proxy summary

- Source: `paper_results/markdown/security_proxy_summary.md` (14 rows) and `paper_results/latex/security_proxy_summary.tex`.
- Columns: `stage`, `attack_family`, `target`, `strategy`, `metric`, `value`, `risk_level`, `recommendation`, `artifact_path`, `claim_supported`.
- Caption must state: *every* `claim_supported` row says `proxy-supported only`. Do not re-word as `proven`.

## Table 6 — LoRA training summary

- Source: `paper_results/markdown/lora_training_summary.md` (11 rows) and `paper_results/latex/lora_training_summary.tex`.
- Columns: `stage`, `training_scope`, `num_layers`, `num_lora_modules`, `true_rank`, `padded_rank`, `optimizer`, `loss_diff`, `grad_error`, `update_error`, `rank_hidden_from_shape`, `risk_level`, `artifact_path`.
- Caption must state: `rank_hidden_from_shape = True` means *true rank* hidden from shape only; padded rank still visible.

## Table 7 — Claims audit / limitations

- Source: `paper_results/markdown/paper_claims_audit.md` (3 buckets: 8 supported, 5 proxy_supported, 8 unsupported) and `paper_results/latex/paper_claims_audit.tex`.
- Columns: `status`, `claim`, `safe_wording`, `unsafe_wording_to_avoid`, `evidence_artifacts`, `notes`.
- Caption must state: every row in the `unsupported` bucket must NOT appear as a positive claim in the body.

## Optional Table A1 — Measured runtime (local emulation)

- Source: `paper_results/markdown/measured_runtime.md` (7 rows, 1 skipped) and `paper_results/latex/measured_runtime.tex`.
- Caption must begin: *"This is local runtime emulation, not real TEE wall-time."* (verbatim).
- Columns: `component`, `variant`, `num_warmup`, `num_repeats`, `mean_ms`, `median_ms`, `std_ms`, `min_ms`, `max_ms`, `device`, `dtype`, `wall_time_source`, `skipped_with_reason`, `notes`.

## Optional Table A2 — Artifact inventory

- Source: `paper_results/markdown/artifact_inventory.md` (22 rows).
- Columns: `slot`, `artifact_name`, `artifact_path`, `status`, `json_error`, `size_bytes`, `top_level_keys`.
- For a final paper the inventory belongs in an artifact appendix, not the body.
