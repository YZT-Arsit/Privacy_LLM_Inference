# 8. Evaluation

Our evaluation is structured around seven research questions, each pinned to a specific subset of the paper-side artifact registry. We use exactly the consolidated tables produced by the Stage 7.5 paper-artifact pipeline (`paper_results/markdown/*.md`, `paper_results/latex/*.tex`, `paper_results/figures/*.png`). We do **not** introduce new experiments in this paper draft; we report the artifacts as they exist.

## 8.1 RQ1: Does masked execution preserve correctness across model components and architectures?

**Setup.** We run the masked wrapper against a plain reference on:

- GPT-2 model-level wrapper (`sshleifer/tiny-gpt2`), full forward + prefill + decode_step + greedy generation.
- Modern decoder-only block wrapper (RMSNorm + SwiGLU + RoPE + GQA), synthetic config and tiny HF config.
- Modern decoder-only model wrapper, prefill + decode_step + greedy generation.
- Compatible nonlinear islands across decoder-only / encoder-only / encoder-decoder / modern-decoder architectures.

**Metric.** `allclose` against the plain reference; per-component `max_abs_error`.

**Result.** The GPT-2 model wrapper reproduces the plain reference output token-for-token in the tested configurations. The modern-decoder block-level and model-level wrappers each reproduce the plain reference output token-for-token in the tested configurations. The cross-architecture compatible-island smoke matches its plain reference. **Table 3 (correctness summary)** lists all 19 component rows; **Figure 5 (`correctness_error_summary.png`)** plots the per-component errors.

**Interpretation.** The algebraic identities of Theorems 1–6 (Section 6) hold on every tested architecture to numerical precision.

**Limitation.** Empirical equality on tested configurations only; no formal proof of universal correctness. Modern-decoder model wrappers use synthetic or tiny-HF configurations, not production-scale Qwen / TinyLlama / LLaMA.

## 8.2 RQ2: Does the system support decoder-only generation with KV cache?

**Setup.** Greedy generation with KV cache append, per-step decode, RoPE rotation, and grouped-query attention. Evaluated for both GPT-2 (multi-head attention) and the modern decoder wrapper (GQA + RoPE).

**Metric.** Sequence-level exact match of decoded tokens versus plain reference; per-step KV cache invariant check.

**Result.** Token-for-token exact match across the tested decode lengths in `outputs/gpt2_generation_correctness.json` and `outputs/modern_decoder_model_wrapper_smoke.json`. KV cache append invariant holds at every step.

**Interpretation.** Right masking + paired `(N_Q, N_K)` + token-axis cache append is sufficient to preserve generation while keeping `K` / `V` in masked space.

**Limitation.** Decoder-only with greedy sampling only. Sampling-based decoding (top-k / top-p / temperature) is implemented at the controller level but is not exercised in the security artifacts. We do not evaluate beam search.

## 8.3 RQ3: How much boundary-call and online-compute overhead does the method introduce?

**Setup.** We compare three method profiles on GPT-2:

- `plain_hf_gpu` — plain reference, no obfuscation.
- `tslp_trusted_nonlinear_baseline` — boundary-mask-only baseline that keeps every nonlinear layer on the trusted side (Amulet-style reference cost model).
- `ours_current` — the integrated wrapper used by the inference path.

**Metric.** Boundary-call count, trusted compute, GPU compute, preprocessing cost, projected wall-time, and (where measured) measured wall-time. See **Table 4 (workload summary)** and `paper_results/figures/boundary_call_reduction.png`.

**Result.** Boundary calls drop from `36` for `ours_current` to `16` for `ours_compatible_nonlinear_islands` and to `4` for an `ours_ideal_gpu_nonlinear` projection that runs every nonlinear layer in an operator-compatible island. The measured wall-time for `ours_current` on tiny-gpt2 is `6.20 ms`; `plain_hf_gpu` measures `2.83 ms` on the same hardware.

**Interpretation.** Compatible nonlinear islands reduce trusted-GPU round-trips by `4.5x` from `36` to `4` on tiny-gpt2. The remaining gap is consistent with the preprocessing cost of mask sampling and the trusted compensation arithmetic.

**Limitation.** These numbers are local-emulation projections on a tiny config (n_layer=2, n_embd=2, n_head=2). They are illustrative, not deployment numbers. Wall-time for the projected rows is `projected_from_op_counts`, not measured.

## 8.4 RQ4: How effective are the mitigation bundles against proxy attacks?

**Setup.** We instantiate the proxy attackers of Section 7 against three mitigation configurations:

- `fresh_perm_only`: per-call mask only, no sandwich, no pad.
- `fresh_perm_plus_sandwich`: per-call mask + dense sandwich.
- `fresh_perm_plus_sandwich_plus_pad`: full mitigation bundle.

**Metric.** Per-attacker accuracy / linkability AUC / distinguishability; consolidated into a risk-level matrix in **Table 5 (security proxy summary)** and **Figure 6 (`security_risk_matrix.png`)**.

**Result.** The full mitigation bundle reduces every adaptive-inverter accuracy to "close to random chance" status in the tested configurations. Specific rows: blackbox query attacker `low` distinguishability via exact-token-match invariance; ridge / MLP / signature / Sinkhorn attackers `needs_more_evaluation` (i.e., the attackers we ran do not succeed, but we do not claim no attacker can). The timing classifier under `proxy_equalized` constant-time mode is `low` (chance accuracy).

**Interpretation.** Under the named proxy attackers, the full bundle behaves as intended. The artifacts also flag the bundle as a *required* mitigation; partial bundles (just permutation, no sandwich) leak more.

**Limitation.** Proxy attackers only. Not a formal-security claim. Specific high-risk rows (gradient rank inference, stronger-dummy spectral inference) are reported as `high` and not re-classified.

## 8.5 RQ5: Can the LoRA private training path preserve forward / backward / update correctness?

**Setup.** We run synthetic single-linear and synthetic multi-layer LoRA training:

- Stage 7.0: single-linear masked LoRA forward, SGD.
- Stage 7.1: single-linear masked LoRA backward, SGD.
- Stage 7.2: rank-padded forward + backward, SGD.
- Stage 7.3: multi-layer LoRA training step across 2 layers and 14 modules.
- Stage 7.4: five stronger dummy distributions on the same single-linear tile.

**Metric.** `max_loss_diff`, `max_grad_a_real_err`, `max_grad_b_real_err`, `max_update_err`, `allclose` against the plain reference. See **Table 6 (LoRA training summary)** and **Figure 7 (`lora_training_errors.png`)**.

**Result.** Across all 11 LoRA-side rows, the masked path matches the plain reference to float64 precision (`max_grad_a_real_err` in the `1e-15` to `2e-15` range, `max_update_err` in the `1e-17` to `1e-16` range, `allclose=True`).

**Interpretation.** The algebraic identities of Theorems 7–9 hold to numerical precision across the tested synthetic tiles and the multi-layer decoder configuration.

**Limitation.** Synthetic tiles only. No production fine-tune of Qwen / TinyLlama / LLaMA. Loss and optimizer remain trusted-side. PEFT / DeepSpeed / vLLM / FlashAttention are not integrated.

## 8.6 RQ6: How much leakage remains from rank, timing, and metadata?

**Setup.** Spectral-rank-inference, gradient-rank-inference, dummy-strategy-classifier, cross-layer linkage, and cost-model training-timing classifier evaluated across Stages 7.2 / 7.3 / 7.4.

**Metric.** Per-attacker risk level; classifier accuracy; AUC. See `paper_results/figures/rank_inference_risk.png` and `paper_results/figures/timing_proxy_before_after.png`.

**Result.**
- Spectral rank inference under `paired_cancellation_dummy`: `needs_more_evaluation` (worst-case).
- Gradient rank inference: `high` under the same dummy strategy.
- Stronger-dummy spectral inference (Stage 7.4): worst-case `high`; recommendation `spectrum_matched_dummy / mixed_dummy_ensemble`.
- Stronger-dummy gradient inference: worst-case `high`.
- Dummy-strategy classifier (Stage 7.4): accuracy `0.476` vs chance `0.143`, risk `medium`.
- Cross-layer linkage with fresh masks per module + paired cancellation: `low`.
- Training-step cost-model timing classifier with `proxy_equalized`: `0.5124` accuracy, risk `low`.

**Interpretation.** Constant-time-mode timing leakage and cross-layer linkage are well-controlled in the tested configurations. Spectral / gradient rank inference is **not** controlled and the artifact says so plainly. We do not claim the padded rank is hidden, and we do not claim rank inference is impossible.

**Limitation.** Spectral-inference high-risk rows are honest open problems, surfaced as Stage 7.6 / 7.7 future work (heterogeneous padded rank, stronger spectral hardening).

## 8.7 RQ7: What is the measured local runtime overhead?

**Setup.** `time.perf_counter` benchmarks of six primitives:

- `plain_synthetic_linear` (`X W`, baseline).
- `plain_lora_forward` (rank-r LoRA forward, no masking).
- `masked_lora_forward` (Stage 7.0 forward).
- `masked_lora_backward` (Stage 7.1 backward).
- `rank_padded_lora_forward` (Stage 7.2 with `paired_cancellation_dummy`).
- `multi_layer_lora_training_step` (Stage 7.3 single training step).
- `modern_decoder_model_wrapper` — *opt-in only, skipped by default* with reason `include_modern_decoder_wrapper=False`.

`num_warmup = 2`, `num_repeats = 5`, `device = cpu`, `dtype = float64`, `wall_time_source = measured_local_emulation`.

**Metric.** Mean / median / std / min / max in milliseconds. See **Table 7 (measured runtime)** and **Figure 8 (`measured_runtime_summary.png`)**.

**Result.** Mean times (illustrative):
- `plain_synthetic_linear`: `0.002 ms`.
- `plain_lora_forward`: `0.008 ms`.
- `masked_lora_forward`: `0.263 ms`.
- `masked_lora_backward`: `0.118 ms`.
- `rank_padded_lora_forward`: `0.270 ms`.
- `multi_layer_lora_training_step`: `4.684 ms`.

**Interpretation.** Masking adds roughly an order of magnitude in absolute time over a tiny baseline, dominated by mask-sampling and per-call boundary arithmetic at this scale. Multi-layer training step dominates as expected.

**Limitation. This is local runtime emulation, not real TEE wall-time.** No real sleep, no real runtime gating, `time.perf_counter` only. Workload tiles are small for test stability — absolute numbers are illustrative. The modern-decoder model-wrapper benchmark is opt-in and recorded as `skipped` with reason `include_modern_decoder_wrapper=False` when not engaged. No PEFT / DeepSpeed / vLLM / FlashAttention integration. Reports publish timing statistics only; raw tensors, adapters, gradients, and masks are never emitted.

## 8.8 Cross-cutting observations

- The full mitigation bundle (fresh per-call mask + dense sandwich + boundary pad) is *required*; partial bundles leak more under the same attackers.
- Stage 7.4 stronger dummy distributions improve cross-layer linkage to `low` but do **not** improve worst-case rank inference; the paper reports this honestly rather than as a "rank hiding" success.
- Local-emulation runtime is a coarse signal — its primary use is to confirm that none of the trusted-side primitives are catastrophically slow at this scale; it is not a deployment latency claim.
