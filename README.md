# privacy_llm_obfuscation

`privacy_llm_obfuscation` is a PyTorch prototype for validating mask/pad obfuscated execution in privacy-preserving large-model inference.

The prototype models two execution domains:

- **Trusted side / Simulated TEE**: owns plaintext inputs, masks, pads, LoRA adapters, compensation generation, and output recovery.
- **Untrusted GPU side**: sees only obfuscated inputs, transformed weights/adapters, compensation tensors, and obfuscated outputs.

This stage uses Python classes to simulate the trusted boundary and keeps interfaces ready for a later real TEE backend.

## Security Boundary

`SimulatedTEE` is a Python simulation of trusted-side execution. It can access plaintext inputs, one-time pads, masks, mask inverses, private LoRA adapters, compensation terms, and output recovery state because those values model what would live inside a real trusted boundary.

`UntrustedGPUExecutor` is a simulation of the untrusted GPU side. It only receives obfuscated inputs, transformed base weights, transformed LoRA adapters, `bias_tilde`, and compensation tensors. It does not generate masks or pads, does not create compensation, and does not recover plaintext outputs.

This prototype validates algebraic correctness only. It does not claim to provide real security isolation, side-channel resistance, memory isolation, attestation, or production TEE guarantees.

Stage 2 uses trusted LayerNorm as an engineering simplification. This is not the final security design. It is used to isolate and validate end-to-end Transformer correctness before implementing a fully obfuscated LayerNorm protocol.

Stage 2 uses trusted GELU as an engineering simplification. The MLP linear layers are obfuscated, but GELU is evaluated after recovering the Linear1 output inside `SimulatedTEE`.

Stage 3 keeps the same trusted LayerNorm and trusted GELU simplifications. It validates prefill/decode, KV cache masking, and greedy generation correctness, but it still does not implement real TEE isolation.

Stage 4 adds a HuggingFace model-zoo abstraction for plain external model loading and inspection. It does not implement obfuscated GPT-2 yet; it only validates that GPT-2-like HuggingFace models can be loaded, inspected, run forward, and used for plain greedy generation.

Stage 4.5 adds GPT-2 HuggingFace `Conv1D` adapter validation. It verifies that GPT-2 `Conv1D` modules follow the project's row-vector linear convention, validates fused `attn.c_attn` Q/K/V splitting, and emits a GPT-2 linear mapping report. It still does not replace GPT-2 modules or implement obfuscated GPT-2.

Stage 4.6 adds a GPT-2 single-block obfuscated wrapper correctness check. It reads one HuggingFace GPT-2 block, uses fused `c_attn` with block-diagonal Q/K/V masks, keeps residual branches in a shared hidden mask space, and compares recovered hidden states against the original HF block. This stage still does not replace the HF model, implement KV cache, or run GPT-2 generation.

Stage 4.6 also audits one-time pad compensation for the single-block wrapper. In `use_pad=True` mode, `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, and `mlp.c_proj` each use `X_tilde = (X - T) N_in` with compensation `C_T = T W N_out`.

Stage 4.7 adds a GPT-2 model-level obfuscated wrapper. It composes per-block `ObfuscatedGPT2BlockWrapper` executions and applies a diagonal vocab output mask `N_vocab = diag(scale)` to the LM head logits. The HuggingFace model is never modified. `use_pad=True` enables Conv1D pad compensation in every block. LM head pad is not added in this stage (vocab dimension is large; vocab output mask is applied instead). KV cache and generation are not implemented in this stage. Trusted LayerNorm and trusted activation shortcuts are carried forward from Stage 4.6.

Stage 4.8 adds GPT-2 prefill/decode KV cache correctness. It introduces an internal `ObfuscatedGPT2KVCache` data structure (one `ObfuscatedGPT2LayerCache` per transformer block) holding GPU-visible obfuscated `key_tilde` / `value_tilde` tensors plus TEE-managed per-head Q/K/V masks. `ObfuscatedGPT2ModelWrapper.prefill()` runs the prompt through the obfuscated blocks and emits a session cache; `decode_step()` consumes the cache, appends the new token's masked K/V, and advances `cache.seq_len` so the decode position id never restarts from 0. The K/V masks are sampled once at prefill and reused across all decode steps in the session, preserving `K_tilde = K N_K` and `V_tilde = V N_V` invariants and the per-head `N_Q N_K^T = I` constraint that keeps attention scores in plaintext. The HuggingFace model is not modified, HF `past_key_values` is used only as a plaintext reference, and no `generate()` / sampling / beam search is wired in.

Stage 4.9 adds GPT-2 greedy generation correctness. `ObfuscatedGPT2ModelWrapper.generate_greedy(input_ids, max_new_tokens)` is built directly on top of Stage 4.8's `prefill()` and `decode_step()`: the first new token is the argmax of `prefill_logits[:, -1, :]`, and each subsequent new token is the argmax of the next `decode_step()` recovered logits. No HuggingFace `generate()` call, no sampling, no temperature, no beam search, no top-k / top-p, and no EOS early-stop is wired in. A plaintext reference path is implemented as a hand-written HF greedy loop (still avoiding `model.generate()`) so per-step logits can be captured and compared, and `token_match_rate` / `sequence_exact_match` are computed against that reference. The internal `ObfuscatedGPT2KVCache` continues to satisfy `K_tilde = K N_K` and `V_tilde = V N_V` after generation, and the HuggingFace model is not modified.

Stage 4.10 is a reproducibility report stage rather than a new wrapper. `scripts/run_experiment_summary.py` aggregates every per-stage correctness JSON (Stages 1, 1-lora, 2, 3-cache, 3-gen, 4.6, 4.7, 4.8, 4.9) into a single triple of artifacts under `outputs/`: `experiment_summary.json` (machine-readable), `experiment_summary.csv` (one row per `(stage, use_pad)` pair), and `experiment_summary.md` (paper-ready Markdown including stage coverage, per-stage trusted-shortcut limitations, side-by-side `use_pad=true` / `use_pad=false` metrics, and the reproducibility command). With `--rerun`, the script first re-executes each upstream correctness script (both pad variants where applicable) into `outputs/_summary_runs/`, so the comparison columns are populated from fresh runs.

Stage 5.0 turns the project from "engineering correctness" toward "paper-grade experiments." It adds two new experiment harnesses under `src/pllo/experiments/`:

* **Attention probe** (`attention_probe.py`): validates six attention invariants on GPT-2 — `Q_tilde K_tilde^T ≈ Q K^T`, softmax-probability invariance, `A V_tilde ≈ (A V) N_V`, `AttnOut_tilde ≈ AttnOut N_res`, prefill cache invariants, and decode-step cache append invariants. It deliberately reuses the existing `gpt2_attention_wrapper` helpers and the per-head mask generation in `pllo.ops.attention`, so any numerical drift it sees also affects the production wrapper. `scripts/run_attention_experiments.py` sweeps `batch_size ∈ {1, 2}`, `seq_len ∈ {4, 8, 16}`, `decode_steps ∈ {1, 2, 4}`, and `use_pad ∈ {true, false}` (36 cells by default) and emits `outputs/attention_experiments.{json,csv,md}`.

* **Workload profiler** (`workload_profiler.py`): produces a TEE/GPU cost-model comparison for five execution strategies — `plain_hf_gpu` (measured), `tslp_trusted_nonlinear_baseline` (projected), `ours_current` (measured), `ours_ideal_gpu_nonlinear` (projected upper bound), and `amulet_style_reference` (projected reference, not a re-implementation). Stage 5.0.1 calibrated the cost model so every method reports four explicit slices of workload — **preprocessing trusted cost** (one-off weight obfuscation, amortised), **online boundary crossings** (true trusted ↔ untrusted round trips, counted via documented per-method formulas such as `3L + 2` for TSLP or `4L + 1` for ours_current), **online trusted compute** (LayerNorm / GELU / sampling / recovery FLOPs running inside the trusted side), and **online GPU obfuscated compute** (masked linears, attention, LM head matmul). Internal Python bookkeeping such as mask-state creation and pad compensation generation is counted as trusted compute, **not** as a boundary call — so the boundary count is no longer inflated by implementation-level overhead. The JSON output additionally carries `module_breakdown`, `interaction_breakdown` (per-interaction slice across `input_masking`, `trusted_layernorm`, `trusted_gelu`, `lm_head_recovery`, `sampling`, `preprocessing_weight_obfuscation`), `paper_metrics` (boundary-call / transfer / trusted-compute reductions vs TSLP, GPU offload ratio, per-forward boundary-call formulas), and an explicit Limitations section warning that this is a **simulated** TEE cost model, not real SGX. `scripts/run_workload_profile.py` writes `outputs/workload_profile.{json,csv,md}`.

## Mathematical Invariants

The first-stage tests validate the following row-vector invariants:

```text
X_tilde = X N_in
X_tilde = (X - T) N_in
Y_tilde = Y N_out
Y_hat = Y_tilde N_out^{-1}
```

For padded execution, trusted-side compensation cancels the contribution of `T` before recovery.

## Current Scope

Implemented in this first stage:

- PyTorch package skeleton.
- Invertible right-mask generation.
- Fresh one-time pad generation.
- Obfuscated execution for standard linear layers.
- Obfuscated execution for LoRA linear layers with independent low-rank branches.
- Pad compensation correctness checks.
- Unit tests.
- Correctness scripts that emit JSON metrics.

Implemented in Stage 2:

- A self-written tiny decoder-only Transformer.
- Plain full-sequence forward pass.
- Obfuscated full-sequence forward pass.
- Obfuscated Q/K/V/O projections, MLP linears, and LM head.
- End-to-end recovered logits correctness checks.
- Token-wise top-1 match metrics for logits.

Implemented in Stage 3:

- Plain and obfuscated prefill/decode APIs for the tiny decoder-only Transformer.
- Plain and obfuscated K/V cache containers.
- Persistent per-layer, per-head K/V cache masks within a generation session.
- K/V cache invariant checks for `K_tilde = K N_K` and `V_tilde = V N_V`.
- Greedy generation correctness with token sequence matching.

Implemented in Stage 4:

- `model_zoo` abstraction for external model loaders.
- HuggingFace loader for `sshleifer/tiny-gpt2`, `distilgpt2`, and `gpt2`-style causal LMs.
- Tokenizer loading through HuggingFace `AutoTokenizer`.
- Plain GPT-2 forward and greedy generation smoke scripts.
- GPT-2 module inspection and structural spec reporting.
- Recognition of HuggingFace GPT-2 `Conv1D` projection modules.

Implemented in Stage 4.5:

- GPT-2 `Conv1D` to internal row-vector Linear adapter helpers.
- Equivalence checks for `hf_conv1d(x)` and `x @ W_internal + b_internal`.
- Fused `attn.c_attn` Q/K/V weight and bias splitting.
- Mapping report for `c_attn`, `c_proj`, `mlp.c_fc`, `mlp.c_proj`, embeddings, and `lm_head`.

Implemented in Stage 4.6:

- GPT-2 single-block obfuscated wrapper.
- Fused `attn.c_attn` block-diagonal Q/K/V mask strategy.
- Obfuscated `attn.c_proj`, `mlp.c_fc`, and `mlp.c_proj` through Conv1D-as-linear adapters.
- Trusted LayerNorm and trusted activation shortcuts for block-level correctness isolation.
- Single-block recovered hidden-state correctness script.
- Pad compensation audit report for each GPT-2 Conv1D path in the single-block wrapper.

Implemented in Stage 4.7:

- GPT-2 model-level obfuscated wrapper chaining all transformer blocks.
- Diagonal vocab output mask `N_vocab = diag(scale)` on the LM head.
- Full-forward recovered-logits correctness script with `top1_match_rate` and pad/lm_head audit reports.

Implemented in Stage 4.8:

- `ObfuscatedGPT2KVCache` and `ObfuscatedGPT2LayerCache` data structures with per-layer K/V masks and `[batch, heads, seq, head_dim]` cache shape.
- `ObfuscatedGPT2ModelWrapper.prefill()` and `decode_step()` with internal cache (HF `past_key_values` is used only as a plaintext reference).
- Per-head K/V mask reuse across all decode steps in a session; per-head `N_Q N_K^T = I` constraint preserved.
- Position id advances from `cache.seq_len` and never restarts from 0.
- Cache invariant metrics `K_tilde ≈ K N_K`, `V_tilde ≈ V N_V` and prefill/decode logits correctness script.

Implemented in Stage 4.9:

- `ObfuscatedGPT2ModelWrapper.generate_greedy(input_ids, max_new_tokens)` built on top of `prefill()` + `decode_step()`.
- Token-level alignment with a hand-written plaintext HF greedy loop (no `model.generate()` call): `token_match_rate`, `sequence_exact_match`, and per-step logits `allclose` / `top1_match_rate` checks.
- Cache invariant continues to hold after the full generation session.

Implemented in Stage 4.10:

- `scripts/run_experiment_summary.py` aggregates every stage's correctness JSON into `outputs/experiment_summary.{json,csv,md}` (stage coverage, trusted-shortcut limitations, side-by-side pad-variant metrics, and reproducibility command).
- `--rerun` re-executes each upstream correctness script for both `use_pad=true` and `use_pad=false` into `outputs/_summary_runs/` so the comparison columns are reproducible from a single command.

Implemented in Stage 5.0:

- `src/pllo/experiments/attention_probe.py` — six-invariant attention correctness probe with prefill / decode coverage and a 36-cell parameter sweep.
- `src/pllo/experiments/workload_profiler.py` — TEE/GPU cost-model comparison for `plain_hf_gpu`, `tslp_trusted_nonlinear_baseline`, `ours_current`, `ours_ideal_gpu_nonlinear` with per-module breakdown and explicit measured-vs-projected flags.
- `src/pllo/experiments/experiment_registry.py` + `report_utils.py` — sweep registry, method registry, cost-model constants, shared JSON / CSV / Markdown emitters.
- `scripts/run_attention_experiments.py` and `scripts/run_workload_profile.py` — driver scripts emitting `outputs/attention_experiments.{json,csv,md}` and `outputs/workload_profile.{json,csv,md}`.

Calibrated in Stage 5.0.1:

- Workload profiler cost model split into four explicit categories: preprocessing trusted cost (amortised), online boundary crossings, online trusted compute, online GPU obfuscated compute.
- Per-method boundary-call formulas documented (`3L + 2` for TSLP, `4L + 1` for `ours_current`, `1` for `ours_ideal` / `amulet_style_reference`) and emitted alongside the numeric results.
- Mask-state creation, pad compensation generation, and other internal trusted-side bookkeeping are now attributed to `online_trusted_compute_ops`, not boundary calls.
- Added `amulet_style_reference` as a projected reference cost model (explicitly marked `implemented: false`, not a re-implementation of any published system).
- New `interaction_breakdown` and `paper_metrics` JSON sections plus a "main online bottleneck" interpretation row.

Implemented in Stage 6.0:

- New `src/pllo/architectures/` package — `architecture_types.py` (`ArchitectureType` enum, `ArchitectureModelSpec`, `AttentionKindSpec`), `architecture_registry.py` (default + fallback model ids per architecture, `AutoModelFor*` class hints), `architecture_inspector.py` (auto-load + classify), `attention_taxonomy.py` (causal / bidirectional / cross-attention reference specs with required invariants), and `encoder_only_spec.py` / `encoder_decoder_spec.py` (BERT / T5 / BART module-path metadata).
- `scripts/run_architecture_coverage.py` emits `outputs/architecture_coverage.{json,csv,md}` covering decoder-only (`sshleifer/tiny-gpt2`), encoder-only (`hf-internal-testing/tiny-bert` with fallbacks), and encoder-decoder (`hf-internal-testing/tiny-random-t5`). Models that fail to load are recorded as `skipped` rather than crashing the report.
- Attention taxonomy documents the three required invariants — `Q_tilde K_tilde^T = Q K^T` everywhere, `K_cache_tilde = K_cache N_K` for causal self-attention, and `Q_dec_tilde K_enc_tilde^T = Q_dec K_enc^T` for cross-attention.
- This stage is a **scaffold only** — no obfuscated forward / cache / generation path exists for BERT / T5 / BART yet; those are deferred to Stages 6.1 (encoder-only probe) and 6.2 (cross-attention probe).

Implemented in Stage 6.1:

- `src/pllo/experiments/encoder_attention_probe.py` — bidirectional self-attention probe for BERT-style encoder-only models. Pulls per-layer `nn.Linear` `query` / `key` / `value` / `output.dense` modules into the project's row-vector `[d_in, d_out]` convention, builds independent per-projection mask states with the same `N_Q N_K^T = I` constraint Stages 4.6 / 5.0 use for GPT-2, and validates the 10 invariants enumerated in the task spec (`Q_tilde = Q N_Q`, `K_tilde = K N_K`, `V_tilde = V N_V`, the QK constraint, the score / softmax / V-aggregation invariants, the output-projection invariant `Y_tilde = Y N_out`, and Q/K/V/O pad compensation under `use_pad=true`).
- Both all-ones and per-batch padding attention masks are tested for every cell — the additive mask is added in the same trusted-side step for the obfuscated path as for the plain reference, so padding does not perturb the algebraic invariants.
- `scripts/run_encoder_attention_experiments.py` sweeps `batch_size ∈ {1, 2}`, `seq_len ∈ {4, 8, 16}`, `use_pad ∈ {true, false}` (12 cells × 2 mask kinds = 24 metric rows) and writes `outputs/encoder_attention_experiments.{json,csv,md}`. Cells that fail to load are recorded as `skipped` rather than crashing the script.
- This stage continues to **not** implement BERT obfuscated forward, LayerNorm / GELU / FFN obfuscation, MLM head handling, pooler / classification heads, encoder-decoder cross-attention, or real TEE security — those are explicit Limitations in the report.

Implemented in Stage 6.2:

- `src/pllo/experiments/cross_attention_probe.py` — encoder-decoder cross-attention probe for T5- and BART-style models. Decoder hidden states feed Q while encoder memory feeds K/V, so the input mask space for Q is independent of the input mask space for K/V. The probe validates the same Q/K/V mask invariants, the per-head `N_Q_dec N_K_enc^T = I` constraint, the `Q_dec_tilde K_enc_tilde^T ≈ Q_dec K_enc^T` score invariant under both all-ones and padding encoder masks, the V-aggregation invariant `AttnProb V_enc_tilde ≈ (AttnProb V_enc) N_V_enc`, and the output projection `Y_dec_tilde = Y_dec N_dec_out` (with Q/K/V/O pad compensation under `use_pad=true`).
- New probe-level `EncoderMemoryCache` dataclass captures plain and obfuscated K/V plus the masks that produced them, and validates `K_enc_tilde ≈ K_enc N_K_enc` / `V_enc_tilde ≈ V_enc N_V_enc`. This is **not** a generation-runtime cache — it is a probe structure only.
- Projection helpers handle `bias=None` (T5 attention) and `bias!=None` (BART attention) uniformly.
- `scripts/run_cross_attention_experiments.py` sweeps `batch_size ∈ {1, 2}`, `dec_seq_len ∈ {1, 4}`, `enc_seq_len ∈ {4, 8, 16}`, `use_pad ∈ {true, false}` (24 cells × 2 encoder-mask kinds = 48 metric rows) and writes `outputs/cross_attention_experiments.{json,csv,md}`. Cells whose model fails to load are recorded as `skipped`.
- This stage continues to **not** implement full T5/BART obfuscated forward, decoder self-attention cache, encoder-decoder generation, LayerNorm / FFN / activation obfuscation, LM head, relative position bias, or real TEE security.

Implemented in Stage 5.3e:

- `src/pllo/ops/mitigation_bundles.py` — `VALID_MITIGATION_BUNDLES = ("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad")`, `DEFAULT_MITIGATION_BUNDLE = "fresh_perm_only"`, `RECOMMENDED_DEFAULT_ON_BUNDLE = "fresh_perm_plus_sandwich_plus_pad"`, `normalize_mitigation_bundle(...)`, `MitigationBundleDescriptor`, `describe_mitigation_bundle(...)`, `bundle_metadata(bundle, *, use_pad, online_extra_matmul_count=0)`. Each descriptor records `fresh_permutation_enabled`, `dense_sandwich_enabled`, `boundary_pad_required`, `activation_pad_forbidden`, `activation_input_form="ZP"`, `island_view_lifetime="short_lived"`, `post_island_dense_mask`, `default_on_candidate_under_stage_5_4`, `risk_level_from_stage_5_4`, `default_on_recommendation`, `security_profile_detail`. The math of Stage 5.2a's `run_gelu_mlp_island` / `run_swiglu_mlp_island` is unchanged across bundles — the bundle is the contract over the per-call fresh-mask sampling that the existing API already produces.
- `src/pllo/ops/nonlinear_islands.py` — `run_gelu_mlp_island` and `run_swiglu_mlp_island` now accept `mitigation_bundle=None` (default `"fresh_perm_only"`) and emit `mitigation_bundle_metadata` in their result dicts. `online_extra_matmul_count` stays at 0; `activation_input_form` is `"ZP"`; pad is never pushed through the activation.
- `src/pllo/hf_wrappers/{gpt2_block_wrapper,gpt2_model_wrapper}.py` — both wrappers accept `mitigation_bundle` (default `"fresh_perm_only"`). The block wrapper threads it into `island_report` (new flags: `mitigation_bundle`, `dense_sandwich_enabled`, `fresh_permutation_enabled`, `boundary_pad_enabled`, `boundary_pad_required`, `activation_input_form`, `activation_pad_forbidden`, `island_view_lifetime`, `post_island_dense_mask`, `default_on_candidate_under_stage_5_4`, `risk_level_from_stage_5_4`, `default_on_recommendation`, `mitigation_bundle_metadata`, `security_profile_detail`). The model wrapper threads it into `island_summary` (aggregated across blocks: `mitigation_bundle`, `dense_sandwich_enabled`, `fresh_permutation_enabled`, `boundary_pad_enabled`, `default_on_candidate_under_stage_5_4`).
- `src/pllo/experiments/{encoder_ffn_island_probe,encoder_decoder_ffn_island_probe,modern_decoder_probe}.py` — each probe config accepts `mitigation_bundle` (default `"fresh_perm_only"`); the bundle is threaded to the underlying island call and propagated to per-cell + global metadata. The modern decoder probe also publishes `mitigation_applies_to = "nonlinear islands (RMSNorm / SwiGLU). RoPE and GQA probes are mask-mathematics independent of the mitigation bundle."`.
- `src/pllo/experiments/adaptive_island_attacker.py` — new `STRATEGIES` entry `fresh_perm_plus_sandwich_plus_pad`: per-session `(X − T) @ N_left @ P @ N_right`. The mitigation summary gains `recommended_default_on_bundle`, `recommended_default_on_bundle_status`, `recommended_default_on_bundle_risk_level`, plus per-row `is_recommended_default_on_bundle`. Tested headline (hidden=64): full bundle lands at `risk=low`, `default_on_recommendation=acceptable_with_mitigation`, permutation top1 ≈ 1/H ≈ 0.016, linear inverter `rel_l2` > 1.0 — comparable to `dense_sandwich` and far below `fixed_permutation` / `boundary_pad_only_activation_view`.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands` gains `mitigation_bundle_selectable=True`, `default_mitigation_bundle="fresh_perm_only"`, `recommended_default_on_bundle="fresh_perm_plus_sandwich_plus_pad"`, `recommended_default_on_status="acceptable_with_mitigation_under_adaptive_proxy"`, `dense_sandwich_supported=True`, `boundary_pad_required=True`, `fresh_permutation_required=True`, `security_profile_detail_with_full_bundle="adaptive-proxy-mitigated, not formal"`. Top-level `wrapper_integration_status.ours_compatible_nonlinear_islands` mirrors. **Unchanged**: `implemented=False`, `wall_time_source="projected_from_op_counts"`, `full_runtime_integrated=False` — Stage 5.3e is mitigation selectability, not a runtime promotion.
- `src/pllo/experiments/cross_architecture_summary.py` — new `mitigation_bundle_support` field with per-architecture rows (`architecture`, `integration_level`, `fresh_perm_only`, `fresh_perm_plus_sandwich_plus_pad`, `use_pad_supported`, `dense_sandwich_enabled`, `online_extra_matmul_count`, `default_on_candidate`, `security_profile`). The script emits a **Mitigation Bundle Support** table after the existing integration-status section.
- Smoke scripts gain `--mitigation-bundle {fresh_perm_only, fresh_perm_plus_sandwich_plus_pad}` and `--both-bundles` flags: `scripts/run_gpt2_model_compatible_island_smoke.py`, `scripts/run_modern_decoder_probe.py`, `scripts/run_cross_architecture_compatible_island_smoke.py`. `scripts/run_nonlinear_island_experiments.py` accepts `--both-bundles` as a documentation acknowledgement (the island math is identical across bundles — the bundle is metadata over the same per-call fresh-mask sampling). All scripts default to `fresh_perm_only` so existing callers keep their old behaviour byte-for-byte.
- New tests:
  - `tests/test_mitigation_bundles.py` (11 tests) — enum, normalizer, descriptor invariants.
  - `tests/test_dense_sandwich_nonlinear_islands.py` (8 tests) — GELU / SwiGLU island correctness under both bundles, metadata invariants, `island math is bundle-invariant`.
  - `tests/test_gpt2_dense_sandwich_integration.py` (10 tests) — GPT-2 block / model wrapper accepts bundle, trusted mode unaffected, full bundle forward + greedy correctness, summary metadata.
  - `tests/test_cross_architecture_dense_sandwich.py` (6 tests) — BERT / T5 / modern decoder probes with full bundle, cross-architecture smoke `--both-bundles`, summary JSON + Markdown surface.
  - `tests/test_adaptive_attacker_dense_sandwich_bundle.py` (10 tests) — `fresh_perm_plus_sandwich_plus_pad` registered + low-risk decision + close-to-dense-sandwich recovery + required mitigations + markdown phrase.
- Stage 5.3e does **not** flip `implemented` to `True`; does **not** change default mode (`nonlinear_mode` stays `"trusted"`); does **not** change default bundle (`mitigation_bundle` stays `"fresh_perm_only"`); does **not** claim formal security; security is `adaptive-proxy-mitigated, not formal` when the full bundle is enabled. It is **not** a real TEE measurement. The Stage 5.4 mitigation table now has a directly-selectable implementation, but production deployment still requires explicit opt-in via both `nonlinear_mode="compatible_islands"` AND `mitigation_bundle="fresh_perm_plus_sandwich_plus_pad"` AND `use_pad=True`.

Implemented in Stage 6.4:

- `src/pllo/architectures/architecture_registry.py` — new `modern_decoder_only` candidate list (`hf-internal-testing/tiny-random-LlamaForCausalLM`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct`) plus a `MODERN_DECODER_FAMILY_MAP`. The new key is a *subtype* of `decoder_only` in `ARCH_KEY_TO_TYPE`; existing taxonomy (decoder_only / encoder_only / encoder_decoder) is unchanged.
- `src/pllo/experiments/rope_probe.py` — LLaMA / Qwen-style RoPE (`apply_rope` with `[..., S, D]` layout, even-`head_dim` requirement) plus two probes:
  - **Probe A (REQUIRED)** — post-RoPE masking invariant `Q_tilde K_tilde^T ≈ Q_rope K_rope^T` with per-head `N_Q N_K^T = I`. Passes at `max_abs_error ≈ 4e-6`.
  - **Probe B (feasibility / negative result)** — pre-RoPE mask commutation `RoPE(Q N) ?= RoPE(Q) N` across three families: `dense_invertible` (does NOT commute), `orthogonal` (does NOT commute), `block_diagonal_rotation` (a per-pair 2D rotation in the same RoPE planes — DOES commute). Reported as a negative-result probe; system correctness only requires Probe A.
- `src/pllo/experiments/gqa_probe.py` — synthetic tensor-level GQA / MQA / MHA probe. Mask per kv-head; per q-head `N_Q[i] = N_K[i // group_size]^{-T}`; verify `Q_tilde K_tilde_rep^T ≈ Q K_rep^T` (score path) and `AttnProb V_tilde_rep ≈ (AttnProb V_rep) N_V_rep` (value path). Mask dimension is `head_dim`, not `hidden_size` and not `num_heads`. Tested across MQA (kv=1), GQA (kv=2), MHA (kv=4).
- `src/pllo/experiments/modern_decoder_probe.py` — orchestrator composing RMSNorm orthogonal island (reuses Stage 5.2a `run_rmsnorm_orthogonal_island`) + SwiGLU paired-permutation island (reuses Stage 5.2a `run_swiglu_mlp_island`) + the two probes above. `attempt_real_model_load=False` by default — synthetic tensors only, so `pytest` never depends on network downloads. `--attempt-real-model-load` walks the registered candidates; any failure silently falls through to synthetic.
- `scripts/run_modern_decoder_probe.py` — runner writing `outputs/modern_decoder_probe.{json,csv,md}`. Markdown contains: Experiment Scope, Model Loading Status, Modern Decoder Architecture Spec, RMSNorm Orthogonal Island Probe, SwiGLU Paired-Permutation Island Probe, RoPE-Aware Attention Probe, GQA / MQA KV Shape Probe, Workload / Integration Status, Security Caveats from Stage 5.4, Limitations, Next Stage Plan.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands.wrapper_integration_status` gains `qwen_or_modern_decoder = "implemented_probe_level"` + `modern_decoder_probe = "implemented"`. `measured_integration_scope` bumps to `"cross_architecture_plus_modern_decoder_probe_level"`. `implemented` / `wall_time_source` / `full_runtime_integrated` are unchanged.
- `src/pllo/experiments/cross_architecture_summary.py` — `compatible_island_integration_status` now appends a fourth row labelled `architecture_type="modern_decoder_only"` (logical type stays `decoder_only`) with `model_family="qwen_like / llama_like / synthetic_modern_decoder"`, `integration_level="probe_level"`, `nonlinear_mode_available=["trusted","compatible_islands"]`, `online_extra_matmul_count=0`, plus a `modern_decoder_probe_status` field. The markdown table emits four rows for `decoder_only` / `encoder_only` / `encoder_decoder` / `modern_decoder_only`.
- `tests/test_rope_probe.py` (9 tests), `tests/test_gqa_probe.py` (8 tests), `tests/test_modern_decoder_probe.py` (15 tests) — synthetic-only, no HF downloads; cover RoPE shape / norm preservation / odd-`head_dim` rejection / zero-position identity / post-mask invariant / pre-mask commutation expectations / GQA score+value invariants for MQA / GQA / MHA / mask-dimension = head_dim / non-divisible heads rejected / orchestrator integration-level / workload + cross-arch markdown updates.
- Stage 6.4 does **not** claim full Qwen / TinyLlama integration. No model-level wrapper, no LM head, no generation runtime, no KV cache. `compatible_islands` stays gated behind the Stage 5.3a `nonlinear_mode` feature flag — default `"trusted"`. The probe inherits the Stage 5.4 mitigation table: fixed permutation is `unsafe_default_on`; the recommended default-on candidate is `fresh permutation + dense sandwich + pad at Linear boundaries`.

Implemented in Stage 6.4b:

- `src/pllo/model_zoo/modern_decoder_spec.py` — `ModernDecoderBlockSpec` + `inspect_modern_decoder_block`. Walks `model.model.layers[i]` (LLaMA / Qwen2 / TinyLlama convention), resolves `input_layernorm` / `self_attn.{q,k,v,o}_proj` / `post_attention_layernorm` / `mlp.{gate,up,down}_proj` paths, classifies family as `llama_like` / `qwen_like` / `tinyllama` / `unknown` (raises with explicit reason if the layout is not recognised). Also exposes `extract_linear_row_weights` (transposes HF `[out, in]` to row-vector `[in, out]`, tolerates `bias=None`) and `extract_rmsnorm_params` (handles both `LlamaRMSNorm.variance_epsilon` and torch `RMSNorm.eps`). Spec records hidden / intermediate / num_attention_heads / num_key_value_heads / head_dim / `rope_theta` / `rope_scaling_kind` / `attention_variant` (mha / mqa / gqa).
- `src/pllo/hf_wrappers/modern_decoder_block_wrapper.py` — block-level obfuscated forward. `ModernDecoderBlockWeights` is a JSON-friendly weight container (`from_synthetic` for pytest, `from_hf_block` for real HF LLaMA / Qwen). `plain_block_forward` runs a reference forward built from the extracted weights (no HF block forward call required). `ObfuscatedModernDecoderBlockWrapper` runs the compatible-islands path: orthogonal residual mask `N_res` applied to the residual stream; RMSNorm γ folded into adjacent q/k/v and gate/up projections via `fold_rmsnorm_affine_into_linear`; per-head Q/K/V masks applied **after** RoPE with `N_Q N_K^T = I` (Stage 6.4 invariant); GQA / MQA handled via `repeat_kv` after masking with one mask pair per kv-head; SwiGLU MLP path runs through `run_swiglu_mlp_island` with freshly-sampled `n_in_island` / `perm` / `n_out_island` per call. Default `nonlinear_mode="trusted"`, default `mitigation_bundle="fresh_perm_only"`.
- `src/pllo/experiments/modern_decoder_block_probe.py` — orchestrator. `ModernDecoderLoadConfig.attempt_real_model_load` defaults to `False` so pytest never downloads. With `--attempt-real-model-load`, walks the registered `modern_decoder_only` candidates; any failure (network, missing weights, unsupported layout) silently falls through to a synthetic LLaMA-shape block. Sweeps `(mitigation_bundle × use_pad)` and emits `per_run` rows with `max_abs_error`, `relative_l2_error`, `allclose`, mitigation metadata, and the four sub-path handling statuses (`rmsnorm_status`, `rope_attention_status`, `gqa_status`, `swiglu_status`).
- `scripts/run_modern_decoder_block_smoke.py` — runner writing `outputs/modern_decoder_block_wrapper_smoke.{json,md}`. CLI: `--attempt-real-model-load`, `--model-id`, `--local-files-only`, `--use-synthetic-fallback` / `--no-synthetic-fallback`, `--use-pad {true,false,both}`, `--mitigation-bundle …`, `--both-bundles`, `--nonlinear-mode {trusted,compatible_islands}`. Markdown contains: Experiment Scope, Model Loading Status, Modern Decoder Block Spec, Plain Reference vs HF Block Status, RMSNorm Handling, RoPE-Aware Attention Handling, GQA / MQA Handling, SwiGLU Compatible Island Handling, Mitigation Bundle Results, Limitations, Next Stage Plan.
- `src/pllo/experiments/workload_profiler.py` — `wrapper_integration_status.qwen_or_modern_decoder` promotes from `"implemented_probe_level"` → `"implemented_block_level"`, and a new key `modern_decoder_block_wrapper = "implemented"` is added. `measured_integration_scope` bumps to `"cross_architecture_plus_modern_decoder_block_level"`. **Unchanged**: `implemented=False`, `wall_time_source="projected_from_op_counts"`, `full_runtime_integrated=False` — Stage 6.4b is block-level correctness, not a runtime promotion.
- `src/pllo/experiments/cross_architecture_summary.py` — `_STATUS_TO_LEVEL` learns a new `implemented_block_level → block_level` mapping. The `modern_decoder_only` row's `integration_level` is `"block_level"` whenever the workload status is `implemented_block_level`, and the row gains a `modern_decoder_block_wrapper_status` field plus a `block_level_correctness_artifact` pointer to `outputs/modern_decoder_block_wrapper_smoke.json`. The Markdown emitter renders a dedicated "Modern Decoder Block-Level Integration (Stage 6.4b)" callout under the integration table.
- `tests/test_modern_decoder_spec.py` (10 tests), `tests/test_modern_decoder_block_wrapper.py` (17 tests), `tests/test_modern_decoder_block_smoke.py` (4 tests) — synthetic-by-default; the real-HF tests skip if `transformers` or the tiny-random-LLaMA checkpoint is unavailable. Coverage: spec field resolution (LLaMA / Qwen / TinyLlama / unknown), `bias=False` Linear extraction, RMSNorm `eps` vs `variance_epsilon`, plain-reference shape, compatible-islands allclose for both bundles × both `use_pad`, `online_extra_matmul_count==0`, mitigation metadata (`dense_sandwich_enabled`, `default_on_candidate_under_stage_5_4`, etc.), intermediate-branch allclose (attention / h_mid / MLP), MHA / GQA / MQA shape sweep, default mode / default bundle preserved, invalid mode / bundle raises, smoke JSON + Markdown contents.
- Headline Stage 6.4b numbers (synthetic block, `hidden=64`, `intermediate=128`, `q_heads=4`, `kv_heads=2`, `head_dim=16`, dtype `float32`):

| bundle | use_pad | max_abs_error | rel_l2_error | allclose | dense_sandwich | default_on_candidate |
|---|---|---|---|---|---|---|
| `fresh_perm_only` | False | ≈ 2e-6 | ≈ 4e-7 | True | False | False |
| `fresh_perm_only` | True | ≈ 2e-6 | ≈ 5e-7 | True | False | False |
| `fresh_perm_plus_sandwich_plus_pad` | False | ≈ 2e-6 | ≈ 4e-7 | True | True | False (needs pad) |
| `fresh_perm_plus_sandwich_plus_pad` | True | ≈ 2e-6 | ≈ 5e-7 | True | True | **True** |

  Real `hf-internal-testing/tiny-random-LlamaForCausalLM` (hidden=16, head_dim=4) — all four runs allclose at `max_abs_error ≤ 8e-7`.

- Stage 6.4b does **not** implement a full Qwen / TinyLlama model-level wrapper, does **not** implement generation / `decode_step` / KV cache runtime, does **not** implement a tokenizer or LM head path, does **not** flip `implemented` to `True`, does **not** change the default `nonlinear_mode` ("trusted"), does **not** change the default `mitigation_bundle` ("fresh_perm_only"), does **not** claim formal security, and is **not** a real TEE measurement. RoPE is handled by post-RoPE per-head masking; mask-before-RoPE dense commutation is not assumed. RMSNorm γ folds into adjacent linears so the norm core runs in the orthogonal `N_res` residual mask space. At the SwiGLU boundary the residual `N_res` is removed (trusted-side; `N_res` is orthogonal so this is exact) and the SwiGLU island applies its own freshly-sampled `N_in` / `P` / `N_out` per the Stage 5.3e bundle. `online_extra_matmul_count = 0` is preserved end-to-end.

Implemented in Stage 6.4c:

- `src/pllo/cache/modern_decoder_kv_cache.py` — `ModernDecoderLayerKVCache` + `ModernDecoderKVCache` + `init_empty_modern_decoder_kv_cache`. Per-layer cache holds masked `K_tilde = K @ N_K` / `V_tilde = V @ N_V` (shape `[B, num_kv_heads, seq, head_dim]`) plus the per-kv-head dense mask stack `n_k_stack` / `n_v_stack` / `n_v_inv_stack` so the decoder can mask new tokens the same way prefill did. Append along the seq axis is a plain `torch.cat` — the mask material is constant per generation session. `summary_dict()` is JSON-safe (shapes, layer counts, SHA-256-prefix mask fingerprints; never the raw tensors).
- `src/pllo/hf_wrappers/modern_decoder_model_wrapper.py` — `ModernDecoderModelWeights` (with `from_synthetic` and `from_hf_model`, the latter walking `model.model.embed_tokens` / `model.model.layers` / `model.model.norm` / `model.lm_head` and handling tied embeddings) + `ObfuscatedModernDecoderModelWrapper` with:
  - `full_forward(input_ids) -> (logits, report)` — embeds, stacks Stage 6.4b block wrappers, runs final RMSNorm + optionally-masked LM head (`W_lm_tilde = N_in_lm^{-1} @ W_lm @ N_vocab_lm`; logits recovered with `@ N_vocab_lm^{-1}`).
  - `prefill(input_ids)` — inline attention path per layer that samples per-layer fresh `N_K` / `N_V` and stores them in the returned `ModernDecoderKVCache`. Returns `{"logits_recovered", "logits_plain", "kv_cache", "plain_layer_caches", "report"}`.
  - `decode_step(next_ids, kv_cache, position, *, plain_layer_caches=None)` — reuses cached `N_K` / `N_V` to mask the new K/V, applies position-aware RoPE (`_apply_rope_at(..., position_offset=position)`), concatenates onto the masked cache, runs the SwiGLU island for MLP, computes next-token logits.
  - `greedy_generate(input_ids, max_new_tokens=3)` — hand-written greedy loop (no HF generate, no beam, no top-k/p) that wraps `prefill` + `decode_step` and compares each step against a plain reference token-for-token.
  - Plain references: `plain_model_forward`, `plain_prefill`, `plain_decode_step` — used by tests and the wrapper itself for correctness checking.
  - Default `nonlinear_mode='trusted'` / `mitigation_bundle='fresh_perm_only'`; `compatible_islands` + full bundle + `use_pad=True` must all be explicit.
- `src/pllo/experiments/modern_decoder_model_probe.py` — `ModernDecoderModelWrapperConfig` + `run_modern_decoder_model_probe`. Drives the wrapper across (bundle × use_pad) sweeps, runs `full_forward` / `prefill` / `decode_step` / `greedy_generate` and assembles a JSON-safe report. Reuses Stage 6.4b's `_try_load_real_block` for the best-effort HF loading path with synthetic fallback.
- `scripts/run_modern_decoder_model_smoke.py` — runner writing `outputs/modern_decoder_model_wrapper_smoke.{json,md}`. CLI: `--attempt-real-model-load`, `--model-id`, `--local-files-only`, `--use-synthetic-fallback`, `--max-layers 2`, `--use-pad {true,false,both}`, `--mitigation-bundle …`, `--both-bundles`, `--nonlinear-mode {trusted,compatible_islands}`, `--max-new-tokens 3`, `--collect-traces`. Markdown contains: Experiment Scope, Model Loading Status, Model-Level Wrapper Configuration, Full Forward Correctness, Prefill / Decode-Step Correctness, Decode-Step (one token after prefill), Greedy Generation Correctness, KV Cache Invariants, RoPE / GQA Handling, Mitigation Bundle Results, Trace Hook Status, Limitations, Next Stage Plan. Defensive `_strip_tensors` walker ensures raw tensors never reach the JSON.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands.wrapper_integration_status` adds `qwen_or_modern_decoder = "implemented"` (promoted from `"implemented_block_level"`) + `modern_decoder_model_wrapper = "implemented"`. Per-method record + top-level mirror also gain `modern_decoder_generation_status = "greedy_generation_implemented"`, `modern_decoder_kv_cache_status = "implemented"`, `modern_decoder_model_smoke_artifact = "outputs/modern_decoder_model_wrapper_smoke.json"`. `measured_integration_scope` is bumped to `"cross_architecture_plus_modern_decoder_model_level"`. **Unchanged**: `implemented=False`, `wall_time_source="projected_from_op_counts"`, `full_runtime_integrated=False` (real TEE wall-time is still unmeasured).
- `src/pllo/experiments/cross_architecture_summary.py` — `_STATUS_TO_LEVEL` already mapped `"implemented" → "model_level"` from Stage 5.3c. The `modern_decoder_only` row now reports `integration_level="model_level"`, plus new fields `modern_decoder_model_wrapper_status`, `modern_decoder_generation_status`, `modern_decoder_kv_cache_status`, and `model_level_correctness_artifact`. Row limitations rewrite themselves for the model-level case (focus on `not a real TEE deployment`, `greedy only`, real-loading opt-in).
- `tests/test_modern_decoder_kv_cache.py` (6 tests), `tests/test_modern_decoder_model_wrapper.py` (13 tests, parametrized → 16 cases), `tests/test_modern_decoder_generation.py` (10 tests), `tests/test_modern_decoder_model_smoke.py` (5 tests). Synthetic-by-default. Coverage: cache shape / append / `K_tilde` + `V_tilde` append invariant / GQA `repeat_kv` masked score invariant / JSON-safety; full-forward allclose for both bundles × both use_pad; default mode `trusted`; compatible mode must be explicit; LM head recovery allclose; final norm status metadata; trace hook off by default; invalid bundle / mode raises; real-LLaMA full forward allclose (skip if unavailable); prefill top1 / cache length / decode top1 / decode cache append invariant / greedy sequence exact match for both bundles × both use_pad; RoPE position metadata; MHA / GQA / MQA cache metadata.
- Headline Stage 6.4c numbers (synthetic, 2 layers, hidden=32, intermediate=64, q_heads=4, kv_heads=2, head_dim=8, vocab=64, prompt=6, max_new_tokens=3, dtype `float32`):

| bundle | use_pad | full_forward allclose | prefill allclose | decode_step top1 | greedy sequence_exact_match |
|---|---|---|---|---|---|
| `fresh_perm_only` | False | ✓ (max_err ≈ 7e-7) | ✓ (max_err ≈ 9e-7) | 1.000 | True |
| `fresh_perm_only` | True | ✓ | ✓ | 1.000 | True |
| `fresh_perm_plus_sandwich_plus_pad` | False | ✓ | ✓ | 1.000 | True |
| `fresh_perm_plus_sandwich_plus_pad` | True | ✓ | ✓ | 1.000 | True |

  Real `hf-internal-testing/tiny-random-LlamaForCausalLM` (2 layers extracted): full_forward `max_abs_error ≈ 3e-6`, greedy `sequence_exact_match=True` for `max_new_tokens=3`.

- Stage 6.4c does **not** measure real TEE wall-time (`wall_time_source="projected_from_op_counts"` and `full_runtime_integrated=False` are unchanged), does **not** implement beam search / top-k / top-p sampling, does **not** implement batched variable-length prompts, does **not** implement LoRA training, does **not** implement real-time tokenizer wiring (the smoke script feeds random `input_ids`), does **not** claim formal / semantic security, does **not** change the default `nonlinear_mode` ("trusted"), does **not** change the default `mitigation_bundle` ("fresh_perm_only"), does **not** require real Qwen2.5-0.5B / TinyLlama-1.1B loading (those are `--attempt-real-model-load` opt-in). Inter-layer hidden states are recovered to plain space between blocks — the attacker view is constrained by Stage 6.4b's intra-block masks plus the new LM-head dense/vocab masks. Stage 5.5 already validated the block-level activation visibility surface; Stage 6.4c reuses the same block wrapper math + per-call fresh-mask sampling for prefill / decode, so the attacker conclusions from Stage 5.5 carry forward.

Implemented in Stage 5.5:

- `src/pllo/hf_wrappers/modern_decoder_block_wrapper.py` — additive `ObfuscatedModernDecoderBlockWrapper.forward_with_traces(x)` returns `(recovered_y, report, traces)`. `traces` is an in-memory dict of `{name: tensor}` exposing both plaintext references and attacker-visible obfuscated tensors at every Linear / SwiGLU / attention boundary: `boundary_input_{plain,visible}` (block residual), `q_{plain,visible}` / `k_{plain,visible}` / `v_{plain,visible}` (post-RoPE per-head), `gate_{plain,visible}` / `up_{plain,visible}` / `swiglu_intermediate_{plain,visible}` (SwiGLU island column-permuted), `post_island_{plain,visible}` (down-projection with dense `N_out_island`), `final_{plain,visible}` (residual `N_res` masked). `forward(x)` is unchanged (it now delegates to `forward_with_traces(collect_traces=False)`); existing 25 Stage 6.4b wrapper tests still pass byte-for-byte.
- `src/pllo/experiments/real_activation_trace.py` — `RealActivationTraceConfig` + `collect_real_activation_traces(config)`. Drives the block wrapper across many "sessions" (each session = one call, masks/permutations refresh between sessions), flattens per-tensor outputs into 2D `[N, D]` datasets, and emits a JSON-safe `trace_summary` containing only shapes, sample counts, scalar statistics, and SHA-256-prefix fingerprints — **never the raw tensor**. The in-memory `traces` dict is consumed only by the in-process attacker.
- `src/pllo/experiments/real_activation_attacker.py` — `RealActivationAttackConfig` + `run_real_activation_attacks(config)`. Per (tensor × bundle) runs the Stage 5.4 attackers reused verbatim: ridge linear inverter (`_fit_linear_inverter`), small MLP inverter (Adam + ReLU + MSE), signature-matching + Sinkhorn soft-assignment permutation recovery (`recover_permutation_by_signature` + `_soft_assignment_top1`), plus a linkability proxy (per-row `cos(V[i], X[i])`, mean pairwise cosine, mean linkability rank). Risk classification (`low` / `medium` / `high`) thresholds: `linear_rel_l2 < 0.2` ⇒ high; `0.2 ≤ linear_rel_l2 < 0.6` ⇒ medium; otherwise low. Permutation recovery and linkability cosine feed into the same classifier with separate thresholds. The two Stage 5.3e bundles (`fresh_perm_only`, `fresh_perm_plus_sandwich_plus_pad`) produce numerically identical traces under the Stage 6.4b wrapper (both use the same per-call fresh `N_in / perm / N_out` sampling inside `run_swiglu_mlp_island`), so deltas are 0.0 by construction; the bundle label distinguishes security posture, not numerical visibility. Optional `fixed_permutation_debug` bundle pins the wrapper's per-session RNG so masks/permutations are identical across sessions — included only as a sanity reference to demonstrate that the attacker DOES recover when freshness is removed (linear `rel_l2 → 0`, risk → `high`).
- `scripts/run_real_activation_attacks.py` — runner writing `outputs/real_activation_attacks.{json,csv,md}`. CLI: `--num-samples`, `--attacker-steps`, `--include-fixed-debug`, `--bundle …` (repeated), `--attempt-real-model-load`, `--model-id`, `--no-use-pad`. CSV is long-format (`section,tensor_name,mitigation_bundle,attack,metric,value,notes`); Markdown contains: Experiment Scope, Model Loading Status, Trace Collection Summary, Target Tensor Inventory, Linear Inverter on Real Activations, Small MLP Inverter on Real Activations, Permutation Recovery on Real Activations, Linkability on Real Activations, Mitigation Bundle Comparison, Per-Bundle Headline, Comparison with Stage 5.4 Synthetic Adaptive Attacker, Recommendation, Limitations, Next Stage Plan. A defensive `_strip_traces` walker ensures raw `torch.Tensor` values never reach the JSON; the runner also asserts no `tensor(...)` substring or numeric array of length ≥ 32 made it through.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands` gains `real_activation_attacker_status="implemented"`, `real_activation_attacker_scope="modern_decoder_block_level"`, `real_activation_attacker_artifact="outputs/real_activation_attacks.json"`, `security_profile_detail_with_real_activation="real-activation-adaptive-proxy-evaluated, not formal"`. **Unchanged**: `security_profile` stays `"proxy-evaluated, not formal"`, `implemented=False`, `wall_time_source="projected_from_op_counts"`, `full_runtime_integrated=False`.
- `src/pllo/experiments/cross_architecture_summary.py` — `modern_decoder_only` row gains `real_activation_attacker_status`, `real_activation_attacker_scope`, `real_activation_attacker_artifact` fields.
- `tests/test_real_activation_trace.py` (10 tests), `tests/test_real_activation_attacker.py` (12 tests) — synthetic-by-default; no network. Coverage: synthetic fallback / `candidates_tried == []`, trace_summary contains gate/up/swiglu_intermediate/post_island, JSON-safe (no `tensor(`, no overlong arrays), in-memory tensors typed as `torch.Tensor`, per-tensor feature-dim matches block_spec (gate/up/swiglu = intermediate_size; q/k/v = head_dim; post_island/boundary = hidden_size), full-bundle final allclose, both Stage 5.3e bundles present in `target_tensor_results`, permutation recovery only on permutation-relevant tensors, full bundle's per-tensor risk_level ≤ fresh-only's risk_level, `fixed_permutation_debug` shows attacker recovery (linear `rel_l2` drops, risk → `high`), Markdown contains required honesty phrases ("real-activation adaptive proxy attacks, not formal security proofs", "Dense sandwiching reduces tested recovery but does not imply semantic security", "No real TEE isolation is evaluated").
- Headline Stage 5.5 numbers (synthetic block, `hidden=64`, `intermediate=128`, `q_heads=4`, `kv_heads=2`, `head_dim=16`, 512 samples, 200 attacker steps, dtype `float32`):

| tensor | bundle | linear rel_l2 | MLP rel_l2 | perm top1 | random chance | linkability cos | risk |
|---|---|---|---|---|---|---|---|
| gate | `fresh_perm_only` | ~1.87 | ~1.10 | ~0.02 | 1/128 = 0.008 | ~0.01 | **low** |
| gate | `fresh_perm_plus_sandwich_plus_pad` | ~1.87 | ~1.10 | ~0.02 | 1/128 | ~0.01 | **low** |
| gate | `fixed_permutation_debug` (reference) | **~0.00** | ~0.11 | ~0.02 | 1/128 | ~0.00 | **high** |
| up | `fresh_perm_only` | ~1.84 | ~1.10 | ~0.02 | 1/128 | ~0.03 | **low** |
| up | `fresh_perm_plus_sandwich_plus_pad` | ~1.84 | ~1.10 | ~0.02 | 1/128 | ~0.03 | **low** |
| swiglu_intermediate | full bundle | ~1.89 | ~1.10 | ~0.02 | 1/128 | ~0.00 | **low** |
| post_island | full bundle | ~1.10 | ~1.05 | n/a | n/a | ~0.00 | **low** |
| q / k / v | full bundle | ~1.05–1.10 | ~1.05 | n/a | n/a | ~0.05 | **low** |

  All 9 target tensors → `low` risk under the full bundle. `default_on_recommendation_full_bundle = "acceptable_with_mitigation_under_real_activation_proxy"`.

- Stage 5.5 does **not** claim formal / semantic security, does **not** flip `implemented` to `True`, does **not** change the default `nonlinear_mode` ("trusted") or default `mitigation_bundle` ("fresh_perm_only"), does **not** evaluate black-box query attacks, does **not** evaluate side-channel attacks, and is **not** a real TEE measurement. Random hidden-state inputs to the block are NOT the same as real tokenizer-driven activations; the limitation is logged. `security_profile` stays `"proxy-evaluated, not formal"`; the added detail label `"real-activation-adaptive-proxy-evaluated, not formal"` is additive metadata only. The two Stage 5.3e bundles produce numerically identical traces under the current wrapper — Stage 5.5 confirms this as a known property of the math, not a limitation of the attacker.

Implemented in Stage 5.5b:

- `src/pllo/hf_wrappers/modern_decoder_model_wrapper.py` — additive `return_traces=False` keyword on `prefill(...)` and `decode_step(...)`. When `True`, the wrapper accumulates per-layer `{name_plain, name_visible}` dicts covering the same attacker boundary inventory as Stage 5.5 (`boundary_input`, `q`, `k`, `v`, `gate`, `up`, `swiglu_intermediate`, `post_island`, `final`) and exposes them via `per_layer_traces` in the returned dict. Correctness math is unchanged (`return_traces=False` is the default and all Stage 6.4c tests pass byte-for-byte).
- `src/pllo/experiments/real_token_trace.py` — `RealTokenTraceConfig` + `collect_real_token_traces(config)`. Optional HF `AutoTokenizer.from_pretrained` load (`attempt_tokenizer_load=True`) with deterministic synthetic-token fallback (`token_source="synthetic_token_ids"`, `tokenizer_status="unavailable"|"not_requested"`). Drives `ObfuscatedModernDecoderModelWrapper.prefill(return_traces=True)` and `decode_step(return_traces=True)` across a small built-in prompt set (`DEFAULT_PROMPTS` — 8 short prompts; `num_prompts` parameter caps the size), then flattens per-layer plain/visible tensors into 2D `[N, D]` datasets and emits a JSON-safe `trace_summary` containing only shapes, sample counts, scalar statistics, and SHA-256-prefix fingerprints — **never raw tensors**. Both `prefill` and `decode` scopes are accumulated.
- `src/pllo/experiments/real_token_activation_attacker.py` — `RealTokenActivationAttackConfig` + `run_real_token_activation_attacks(config)`. Reuses Stage 5.5's attacker family verbatim: ridge linear inverter, two-layer MLP inverter, signature + Sinkhorn permutation recovery, linkability proxy. Per (bundle × scope × tensor) it records `risk_level ∈ {low, medium, high}` plus a `default_on_recommendation`. **Inter-block plain flag**: `boundary_input` and `final` are plain at the Stage 6.4c inter-block surface (the model wrapper recovers between blocks by construction), so these tensors are flagged `inter_block_plain=True` and graded with the recommendation `inter_block_plain_recovered` — this is a structural model-wrapper limitation, NOT a Stage 5.5b attacker finding. The headline grade is reported as both `max_risk_level_masked_only` (excludes inter-block-plain tensors) and `max_risk_level_overall` (includes them) so the structural caveat stays visible.
- `scripts/run_real_token_activation_attacks.py` — runner writing `outputs/real_token_activation_attacks.{json,csv,md}`. CLI: `--num-prompts`, `--prompt-max-length`, `--max-new-tokens`, `--attempt-tokenizer-load`, `--attempt-real-model-load`, `--model-id`, `--max-layers`, `--attacker-steps`, `--bundle …` (repeated), `--no-use-pad`, `--synthetic-*`. CSV is long-format (`section,scope,tensor_name,mitigation_bundle,attack,metric,value,notes`); Markdown contains: Experiment Scope, Model and Tokenizer Loading Status, Prompt Set Summary, Trace Collection Summary, Target Tensor Inventory (Prefill & Decode), Prefill Real-Token Activation Attacks, Decode-Step Real-Token Activation Attacks, Linear Inverter Results, Small MLP Inverter Results, Permutation Recovery Results, Linkability Results, Bundle Comparison, Per-Bundle Headline, Generation Token Match, Comparison with Stage 5.5 Random-Hidden Real-Activation Attacker, Recommendation, Limitations, Next Stage Plan. Defensive `_strip_traces` + `assert "tensor(" not in json_text` + long-numeric-array regex guard prevent tensor leakage.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands` gains `real_token_activation_attacker_status="implemented"`, `real_token_activation_attacker_scope="modern_decoder_model_level_prefill_decode"`, `real_token_activation_attacker_artifact="outputs/real_token_activation_attacks.json"`, `security_profile_detail_with_real_token_activation="real-token-real-activation-adaptive-proxy-evaluated, not formal"`. **Unchanged**: `security_profile`, `implemented=False`, `wall_time_source="projected_from_op_counts"`, `full_runtime_integrated=False`.
- `src/pllo/experiments/cross_architecture_summary.py` — `modern_decoder_only` row gains the three `real_token_activation_attacker_*` fields plus `security_profile_detail_with_real_token_activation`. Cross-arch markdown adds a "Stage 5.5b Real-Token-Prompted Real-Activation Attacker" callout inside the Modern Decoder Model-Level section.
- `tests/test_real_token_trace.py` (10 tests), `tests/test_real_token_activation_attacker.py` (11 tests) — synthetic-by-default; no network. Coverage: synthetic token fallback (`tokenizer_status == "not_requested"`), trace_summary covers prefill AND decode scopes, decode_step_log records position + cache_seq_len_before/after, generation_summary reports `mean_token_match_rate == 1.0` for synthetic, both bundles present in `target_tensor_results`, masked-only recommendation `acceptable_with_mitigation_under_real_token_proxy`, JSON contains no `tensor(` / no overlong numeric arrays, deterministic input_ids across runs (same seed ⇒ same fingerprint), Markdown includes required honesty phrases ("real-token-prompted adaptive proxy attacks, not formal security proofs", "synthetic token fallback", "Dense sandwiching reduces tested recovery but does not imply semantic security", "not formal security", "not a real TEE measurement").
- Headline Stage 5.5b numbers (synthetic-token + synthetic-model fallback, 2 layers, hidden=32, intermediate=64, q_heads=4, kv_heads=2, head_dim=8, vocab=64, 8 prompts × prompt_len=8, max_new_tokens=3, 200 attacker steps):

| scope | tensor | bundle | linear rel_l2 | MLP rel_l2 | perm top1 | random chance | linkability cos | risk |
|---|---|---|---|---|---|---|---|---|
| prefill | gate | full bundle | ≈ 1.26 | ≈ 1.13 | ≈ 0.02 | 1/64 = 0.016 | ≈ -0.02 | **low** |
| prefill | up | full bundle | ≈ 1.26 | ≈ 1.13 | ≈ 0.02 | 1/64 | ≈ -0.02 | **low** |
| prefill | swiglu_intermediate | full bundle | ≈ 1.26 | ≈ 1.13 | ≈ 0.02 | 1/64 | ≈ -0.02 | **low** |
| prefill | q / k / v | full bundle | ≈ 1.05–1.20 | ≈ 1.05 | n/a | n/a | ≈ 0.0 | **low** |
| prefill | post_island | full bundle | ≈ 1.10 | ≈ 1.05 | n/a | n/a | ≈ 0.0 | **low** |
| prefill | boundary_input / final | both | **≈ 0.00** | **≈ 0.00** | n/a | n/a | **≈ 1.0** | **high — inter_block_plain** |
| decode  | gate / up / swiglu_intermediate | full bundle | ≈ 1.05–1.30 | ≈ 1.10 | ≈ 0.02 | 1/64 | ≈ 0.0 | **low** |
| decode  | q / k / v | full bundle | ≈ 1.05 | ≈ 1.05 | n/a | n/a | ≈ 0.0 | **low** |

  Masked-only headline: all SwiGLU island + QKV tensors → `low` risk under both bundles. `default_on_recommendation_full_bundle_masked_only = "acceptable_with_mitigation_under_real_token_proxy"`. Greedy generation `mean_token_match_rate = 1.000` for both bundles (Stage 6.4c correctness inherited).

  Overall headline (including structurally-plain inter-block tensors): `high`. The high grade is structural (`inter_block_plain_recovered`), not an adversarial finding — the Stage 6.4c model-level wrapper recovers between blocks by construction, so `boundary_input` and `final` are plain at the inter-block boundary regardless of mitigation bundle. Recommendation reports both grades so the structural caveat stays visible to downstream consumers.

- Stage 5.5b does **not** claim formal / semantic security, does **not** flip `implemented` to `True`, does **not** change the default `nonlinear_mode` ("trusted") or default `mitigation_bundle` ("fresh_perm_only"), does **not** evaluate black-box query attacks, does **not** evaluate side-channel attacks, does **not** introduce beam / top-k / top-p sampling, does **not** implement LoRA training, and is **not** a real TEE measurement. Real tokenizer / real model loading is opt-in (`--attempt-tokenizer-load --attempt-real-model-load`); pytest stays synthetic and never downloads from HuggingFace. Synthetic-token fallback results are explicitly labelled `token_source="synthetic_token_ids"` so they are not conflated with real prompt distributions. The two Stage 5.3e bundles still produce numerically identical traces under the model-level wrapper (same per-call fresh-mask sampling). Inter-block hidden states being plain is a known Stage 6.4c structural limitation; closing it requires masking the inter-block residual (a Stage 5.6 design item) and is intentionally OUT of Stage 5.5b's scope.

Implemented in Stage 5.6:

- `src/pllo/experiments/blackbox_attacker.py` — `BlackboxAttackerConfig` + `run_blackbox_attacker(config)`. A query-only attacker that NEVER reads `per_layer_traces`, internal hidden states, masks, permutations, or KV cache contents. Per-API-call response = `{token_ids, per_step_logits_summary[top1_id, top5_ids, entropy, max_logit, top1_margin], output_length}`. Three sub-attacks: (a) prompt linkability (same-vs-different prompt cosine + linkability-AUC proxy + nearest-prompt retrieval top-1), (b) prompt class inference (4-way 1-NN over hashed signatures vs. random-chance baseline), (c) mitigation-mode distinguishability (1-NN over signatures with labels = `nonlinear_mode × mitigation_bundle × use_pad`). `attacker_view_inventory` is an explicit allow-list with no internal trace names; `internal_trace_access = "denied"` is also surfaced. Synthetic-by-default; tokenizer / real-model loading is opt-in.
- `src/pllo/experiments/timing_sidechannel_proxy.py` — `TimingSidechannelConfig` + `run_timing_sidechannel_proxy(config)`. Model-based latency proxy: `latency_ms ≈ boundary_calls × tee_call_overhead_ms + gpu_ops / gpu_flops_per_ms + trusted_ops / (gpu_flops_per_ms / tee_to_gpu_flops_ratio) + bytes / tee_bytes_per_ms + Gaussian noise`. Drives Stage 5.2c's per-forward op-count formulas via `_per_forward_boundary_calls / _per_forward_gpu_ops / _per_forward_trusted_compute_ops / _per_forward_trusted_transfer_bytes` and projects via `_project_wall_time_ms`. Four sub-attacks: prompt-length leakage (kNN + Pearson), decode-step leakage, method distinguishability, mitigation-bundle distinguishability. Plus a static `boundary_call_pattern` table per method. Risk = `low` if accuracy ≤ 1.5× random AND |corr| ≤ 0.3, `medium` ≤ 3× / 0.7, `high` otherwise. `cost_model_note` explicitly states this is NOT a real TEE wall-time measurement.
- `src/pllo/experiments/inter_block_masking_probe.py` — `InterBlockMaskingProbeConfig` + `run_inter_block_masking_probe(config)`. Three layers: (A) accounting baseline reads `outputs/real_token_activation_attacks.json` and confirms Stage 5.5b's structural finding (boundary_input / final flagged `inter_block_plain_recovered`); (B) single-transition math probe verifies that an orthogonal `N_inter` is absorbed by the next block's RMSNorm + γ-folded Q-projection (`w_q_tilde = N_inter.T @ (γ ⊙ w_q)`), with allclose for `rmsnorm_invariant`, `q_projection_path`, `residual_recovery`; (C) full `masked_boundary_experimental` mode at the model-wrapper level is **explicitly** `not_implemented_in_stage_5_6` (deferred to Stage 5.6 extension / Stage 7.0) — the probe returns that label rather than silently passing.
- `src/pllo/experiments/stronger_attackers.py` — `StrongerAttackersConfig` + `run_stronger_attackers(config)`. Orchestrator that drives the three sub-modules above and computes a two-axis risk synthesis: **envelope-integrity risk** (`mode_classification_accuracy` from black-box + `mitigation_accuracy` from timing — if both `low`, the mitigation envelope is sound) and **structural-leakage risk** (decode-step / prompt-length / method distinguishability + inter-block plain boundary — acknowledged limitations, NOT envelope failures). The split lets the recommendation distinguish "envelope holds, structural leakage acknowledged" (eligible for label upgrade) from "envelope fails" (do not promote).
- `scripts/run_stronger_attackers.py` — runner writing `outputs/stronger_attackers.{json,csv,md}`. CLI: `--num-prompts`, `--prompt-max-length`, `--max-new-tokens`, `--attacker-trials`, `--timing-noise-std`, `--attempt-tokenizer-load`, `--attempt-real-model-load`, `--model-id`, `--max-layers`, `--inter-block-mask-mode {plain_boundary,masked_boundary_experimental}`, `--mitigation-bundle …`, `--no-use-pad`, `--synthetic-*`. CSV long-format (`section,attack,scope,metric,value,notes`); Markdown contains: Experiment Scope, Threat Model, Black-Box Query Attacker, Timing / Boundary-Call Side-Channel Proxy, Inter-Block Residual Masking Gap, Single-Transition Masking Probe, Comparison with Stage 5.4 / 5.5 / 5.5b, Overall Risk Summary, Recommendation, Limitations, Next Stage Plan. Defensive `_strip_traces` + long-numeric-array regex guard + `assert "tensor(" not in json_text/md_text`.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands` gains 7 Stage 5.6 fields: `stronger_attackers_status="implemented"`, `stronger_attackers_artifact`, `blackbox_proxy_status="implemented"`, `timing_sidechannel_proxy_status="implemented"`, `inter_block_masking_gap_status="identified"`, `inter_block_masking_experimental_status="not_implemented_in_stage_5_6"`, `security_profile_detail_with_stronger_attackers="adaptive-blackbox-and-timing-proxy-evaluated, not formal"`. **Unchanged**: `security_profile`, `implemented=False`, `wall_time_source="projected_from_op_counts"`, `full_runtime_integrated=False`.
- `src/pllo/experiments/cross_architecture_summary.py` — `modern_decoder_only` row gains the same 7 Stage 5.6 fields. Cross-arch markdown adds a "Stage 5.6 Stronger Attackers" callout summarising envelope-integrity vs structural-leakage findings.
- `tests/test_blackbox_attacker.py` (6 tests), `tests/test_timing_sidechannel_proxy.py` (8 tests), `tests/test_inter_block_masking_probe.py` (6 tests), `tests/test_stronger_attackers.py` (8 tests) — synthetic-by-default; no network. Coverage: black-box attacker view is restricted (no internal trace names in inventory; `internal_trace_access == "denied"`), mode distinguishability ≤ 2× random chance under Stage 6.4c's exact-token-match guarantee, timing simulator runs across (length × decode_step × method × bundle) with kNN + Pearson, `cost_model_note` references "not a real TEE", inter-block gap flags `boundary_input` / `final`, single-transition probe `rmsnorm_invariant_allclose / q_projection_path_allclose / residual_recovery_allclose` all True, `masked_boundary_experimental_status == "not_implemented_in_stage_5_6"` when explicitly requested, runner-end-to-end emits the 11 required Markdown sections + 7 required limitations phrases.
- Headline Stage 5.6 numbers (synthetic-token + synthetic-model fallback, 2 layers, hidden=32, intermediate=64, q_heads=4, kv_heads=2, head_dim=8, vocab=64, 8 prompts × prompt_len=8, max_new_tokens=3, timing noise std=0.05):

| dimension | result | risk |
|---|---|---|
| black-box mode distinguishability | accuracy ≈ 0.00 vs random chance ≈ 0.25 (≤ 1.5× rc) | **low** |
| black-box prompt class inference | accuracy ≈ random chance | **low** |
| timing mitigation distinguishability | accuracy ≈ 0.47 vs random chance ≈ 0.50 (≤ 1.5× rc) | **low** |
| timing prompt-length leakage | corr ≈ 0.03, accuracy ≈ random chance | **low** |
| timing decode-step leakage | corr ≈ 0.79, accuracy ≈ 0.78 vs random ≈ 0.25 | **high (structural)** |
| timing method distinguishability | accuracy ≈ 0.65 vs random ≈ 0.33 | **medium (structural)** |
| inter-block accounting baseline | boundary_input / final flagged `inter_block_plain_recovered` | **high (structural)** |
| inter-block single-transition probe | rmsnorm_invariant + q_projection_path + residual_recovery all allclose | **probe_passed** |
| inter-block masked_boundary_experimental | model-wrapper-level full mode | **`not_implemented_in_stage_5_6`** |
| **envelope-integrity risk** | mode + bundle indistinguishable from API output AND from timing | **low** ✓ |
| **structural-leakage risk** | decode step + inter-block plain boundary are observable | **high (acknowledged)** |

  Promotion eligibility: **YES** — envelope-integrity risk is `low`, so `security_profile_detail_with_stronger_attackers = "adaptive-blackbox-and-timing-proxy-evaluated, not formal"` is recorded as additive metadata. `security_profile` itself stays `"proxy-evaluated, not formal"` until Stage 5.6 extension / Stage 7.0 closes the inter-block plain boundary and adds constant-time mitigations for the decode-step latency leakage.

- Stage 5.6 does **not** claim formal / semantic security, does **not** flip `implemented` to `True`, does **not** change the default `nonlinear_mode` ("trusted") or default `mitigation_bundle` ("fresh_perm_only"), does **not** introduce real TEE measurements (`wall_time_source` stays `projected_from_op_counts`), does **not** introduce hardware side-channel attacks (cache / power / EM), does **not** implement LoRA training, does **not** ship the full `masked_boundary_experimental` model-wrapper mode (only a single-transition math probe; full mode is `not_implemented_in_stage_5_6`), does **not** implement constant-time mitigations for the observed decode-step latency leakage, does **not** change the default `inter_block_mask_mode` ("plain_boundary"), and is **not** a formal security proof. Black-box attacker is white-box-of-the-API-only (it sees the same logits-summary surface a deployed inference endpoint would expose). Timing proxy is a Stage 5.2c op-count cost-model simulation, NOT a real-TEE wall-time measurement.

Implemented in Stage 5.4:

- `src/pllo/experiments/adaptive_island_attacker.py` — three adaptive proxy attackers against operator-compatible nonlinear-island masking, evaluated across six strategies (`fixed_permutation`, `fresh_permutation_per_session`, `permutation_pool`, `dense_sandwich`, `boundary_pad_only_boundary_view`, `boundary_pad_only_activation_view`). Strategy datasets share a structured channel distribution `X[:, j] = mean_j + scale_j * (noise + skew_j * (noise**2 - 1))` so per-channel signatures are recoverable (more realistic than isotropic Gaussian).
  - **Attack 1 — Learned linear inverter.** Ridge least-squares ``W = (V^T V + λI)^-1 V^T X`` from a labelled `(V_train, X_train)` pool; evaluated on held-out `(V_test, X_test)`. Reports `mse`, `relative_l2_error`, `cosine_similarity`.
  - **Attack 2 — Small MLP inverter.** Two-layer ReLU MLP (default `mlp_hidden_size=128`), Adam with `attacker_lr=1e-2`, `attacker_steps=200`, batch 64. Reports the same metrics plus `improvement_over_linear` and `mlp_minus_linear_relative_l2_error`.
  - **Attack 3 — Adaptive permutation recovery.** Two methods over the per-channel signature shared with Stage 5.2b — `signature_matching` (reuses `recover_permutation_by_signature`) and `soft_assignment` (Sinkhorn-style log-domain row/column normalisation, `temperature=0.05`, `iters=50`). The mitigation decision takes the max-of-attackers top-1 so the strongest signal governs.
- `src/pllo/experiments/workload_profiler.py` — additive metadata on `methods.ours_compatible_nonlinear_islands`: `adaptive_proxy_evaluated=True`, `security_profile_detail="adaptive-proxy-evaluated, not formal"`, `adaptive_proxy_artifact="outputs/adaptive_island_attacks.json"`. `security_profile` itself remains `"proxy-evaluated, not formal"` so Stage 5.2a/5.2b/5.2c/5.3a/5.3b/5.3c consumers continue to pass; `implemented`, `wall_time_source`, `wrapper_integration_status` are unchanged.
- `scripts/run_adaptive_island_attacks.py` writes `outputs/adaptive_island_attacks.{json,csv,md}`. CSV is long-format (`section,attack,strategy,metric,value,notes`); Markdown includes Experiment Scope, Threat Model, Structured Synthetic Activation Distribution, Learned Linear Inverter, Small MLP Inverter, Adaptive Permutation Recovery, Mitigation Decision Table, Comparison with Stage 5.2b Naive Proxy, Limitations, Next Stage Plan.
- `tests/test_adaptive_island_attacker.py` (17 tests) — section structure, six-strategy coverage, linear inverter `fixed < dense_sandwich`, permutation recovery `max(sig, soft) fixed > dense_sandwich`, `boundary_view` >> `activation_view`, fixed → `unsafe_default_on`, dense_sandwich → `acceptable_with_mitigation`, boundary activation view → `unsafe_default_on`, fresh permutation rejected from `acceptable_with_mitigation` under the adaptive attacker, recommended default-on candidate lists fresh + sandwich + pad, default-on caveat disclaims formal security and TEE, comparison-with-naive uplift fields, secret-tensor refusal (no `tensor(...)` substring, no numeric array of length ≥ `hidden_size`), end-to-end script run, structured-data shape + monotonic per-channel mean profile.
- Headline Stage 5.4 numbers (hidden=64, 16 sessions × 32 samples/session, 200 MLP steps):

| strategy | linear rel_l2 | MLP rel_l2 | best perm top1 | risk | default-on |
|---|---|---|---|---|---|
| `fixed_permutation` | **0.000** | 0.103 | **0.266** | **high** | `unsafe_default_on` |
| `fresh_permutation_per_session` | 1.146 | 1.203 | 0.250 | medium | `needs_more_evaluation` |
| `permutation_pool` | 0.747 | 0.640 | 0.250 | medium | `needs_more_evaluation` |
| `dense_sandwich` | 1.115 | 1.221 | **0.016** ≈ 1/64 | **low** | `acceptable_with_mitigation` |
| `boundary_pad_only_boundary_view` | 1.097 | 1.203 | n/a | low | `acceptable_with_mitigation` |
| `boundary_pad_only_activation_view` | **0.000** | 0.114 | **0.266** | **high** | `unsafe_default_on` |

- Stage 5.4 does **not** claim formal security. `security_profile` stays `"proxy-evaluated, not formal"`; `adaptive_proxy_evaluated=True` is a flag, not a guarantee. `compatible_islands` remains gated behind the Stage 5.3a `nonlinear_mode` feature flag — default `"trusted"` — and the Stage 5.4 mitigation table is the source of truth for which mask strategies may be considered safe under the tested adaptive proxy threat model.

Implemented in Stage 5.3c:

- `src/pllo/experiments/encoder_ffn_island_probe.py` — BERT FFN compatible-island probe. `EncoderFFNIslandProbeConfig(nonlinear_mode=..., use_pad=..., ...)` with `normalize_nonlinear_mode` validation. Discovers `intermediate.dense` + `output.dense` on the first encoder layer, detects activation type via `intermediate_act_fn` class name (with `config.hidden_act` fallback), and routes the GELU / ReLU / SiLU MLP island through Stage 5.2a's `run_gelu_mlp_island`. LayerNorm is **not** modified; MLM head, pooler, and classifier are **not** integrated. Returns full audit metadata: `permutation_dim == intermediate_size`, `online_extra_matmul_count = 0`, `pad_placement ∈ {"linear_boundary_only", "n/a"}`, `uses_fresh_permutation=True`, `security_profile="proxy-evaluated, not formal"`, and the Stage 5.2b security caveats.
- `src/pllo/experiments/encoder_decoder_ffn_island_probe.py` — T5 / BART FFN compatible-island probe with automatic FFN structure detection. Recognises three patterns: `t5_dense_relu_dense` (`wi`/`wo`, activation read from `config.feed_forward_proj` — covers tiny-random-t5 with ReLU), `t5_gated` (`wi_0`/`wi_1`/`wo` paired permutation via `run_swiglu_mlp_island` when SiLU; gated-GELU is reported as `status="unsupported"` with an explicit reason — **no silent pass**), and `bart_fc1_fc2`. The Stage 6.2 cross-attention probe invariants are **not** modified; the LM head and encoder-decoder generation are **not** integrated.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands.wrapper_integration_status.{bert,t5}` flip from `"not_yet"` → `"implemented_probe_level"`. New per-method fields: `measured_integration_scope = "cross_architecture_probe_level"`, `all_architecture_probe_level_implemented = True`, `full_runtime_integrated = False`. The top-level `wrapper_integration_status.ours_compatible_nonlinear_islands` mirrors. Crucially `implemented` remains `False` and `wall_time_source` remains `"projected_from_op_counts"` — `implemented=True` is reserved for full-runtime cross-architecture integration; Stage 5.3c lands probe-level BERT / T5 only.
- `src/pllo/experiments/cross_architecture_summary.py` — new top-level field `compatible_island_integration_status` carrying a per-architecture row (decoder_only / encoder_only / encoder_decoder) with `integration_level ∈ {"model_level", "probe_level", "not_yet"}`, `nonlinear_mode_available`, `use_pad_supported`, `online_extra_matmul_count`, `security_proxy_status`, and per-architecture `limitations`. Global summary gains `compatible_island_integration_status_available`, `compatible_island_full_runtime_integrated`, `compatible_island_all_architecture_probe_level_implemented`.
- `outputs/cross_architecture_summary.md` now has a dedicated **Compatible Island Integration Status** section showing the integration-level table for all three architectures, plus the explicit phrases `GPT-2 model-level integration is available`, `BERT/T5 are probe-level integrations, not full wrappers`, `measured_integration_scope = "cross_architecture_probe_level"`, `full_runtime_integrated = False`, `all_architecture_probe_level_implemented = True`, `LayerNorm remains trusted unless explicitly stated otherwise`, `no generation changes for BERT/T5`, `security follows Stage 5.2b caveats`, `not a real TEE measurement`, and `not full BERT/T5 wrapper integration`.
- `outputs/workload_profile.md` gains the same five integration-status lines under the Stage 5.3a Wrapper Integration Status subsection.
- `scripts/run_cross_architecture_compatible_island_smoke.py` aggregates GPT-2 model-level smoke results (from Stage 5.3b's JSON) plus fresh BERT and T5 / BART FFN probes under `use_pad ∈ {False, True}` and writes `outputs/cross_architecture_compatible_island_smoke.{json,md}`. On tiny-bert (hidden=128, intermediate=512, GELU) the BERT FFN island recovers `allclose=True` with `max_abs_error ≈ 4e-6`, `permutation_dim=512`. On tiny-random-t5 (d_model=32, d_ff=37, ReLU, ungated) the T5 FFN island recovers `allclose=True` with `max_abs_error ≈ 5e-7`, `permutation_dim=37`, `ffn_type="t5_dense_relu_dense"`.
- `tests/test_encoder_compatible_islands.py` (13 tests) — mode acceptance + invalid-mode rejection, default trusted, BERT FFN island correctness under both `use_pad ∈ {False, True}` (including `tilde_invariant_metrics`), `permutation_dim == intermediate_size != hidden_size`, `online_extra_matmul_count == 0`, `pad_placement ∈ {"linear_boundary_only", "n/a"}`, LayerNorm-remains-trusted, MLM/pooler/classifier non-integration, security caveats, activation-type detection.
- `tests/test_cross_attention_compatible_islands.py` (14 tests) — mode acceptance + invalid-mode rejection, default trusted, T5 / BART FFN island correctness, FFN-type detection, `permutation_dim == intermediate_size`, `online_extra_matmul_count == 0`, `pad_placement`, paired-permutation flag only for gated FFNs, LM head / generation / cross-attention probe non-modification, security caveats, explicit gated-GELU unsupported reason (skip with reason, never silent pass).
- `tests/test_workload_profiler_cross_architecture_islands.py` (10 tests) — workload JSON `bert/t5 = "implemented_probe_level"`, `measured_integration_scope = "cross_architecture_probe_level"`, `all_architecture_probe_level_implemented = True`, `full_runtime_integrated = False`, `implemented` remains `False`, top-level wrapper-integration mirror, workload markdown phrase checks, cross-architecture summary JSON / Markdown emission for three architectures with the new integration-status table.
- Stage 5.3c does **not** claim formal security; `compatible_islands` remains `proxy-evaluated, not formal`. The Stage 5.2b mitigations (fresh permutation per session, dense sandwich at Linear boundaries, pad at Linear boundaries only) remain required. Default mode for every wrapper / probe remains `trusted`. BERT MLM head / pooler / classifier and T5 / BART LM head / encoder-decoder generation remain untouched. Full BERT / T5 wrappers are deferred (Stage 6.4 Qwen migration will reuse the probe-level pattern).

Implemented in Stage 5.3b:

- `src/pllo/hf_wrappers/gpt2_model_wrapper.py` — `ObfuscatedGPT2ModelWrapper.__init__` now accepts `nonlinear_mode=...` (default `"trusted"`, validated by `normalize_nonlinear_mode`) and passes it down to every `ObfuscatedGPT2BlockWrapper` it constructs. All blocks share the same mode in Stage 5.3b; per-block mixing is intentionally not supported.
- The wrapper exposes two new audit accessors: `island_reports` (per-block raw `island_report` list) and `island_summary` (recomputed on every read). The summary aggregates `nonlinear_mode`, `num_blocks`, `blocks_with_compatible_islands`, `total_mlp_island_permutation_draws`, `online_extra_matmul_count`, `layernorm_remains_trusted`, `lm_head_not_modified`, `generation_path_not_modified`, `pad_placement` (collapsed when uniform across blocks), `security_profile`, `security_caveats`, and `wrapper_integration_scope="gpt2_model_level"`.
- Scope kept tight: under `nonlinear_mode="compatible_islands"` every block's MLP GELU is routed through the Stage 5.2a permutation island via the Stage 5.3a feature flag — LayerNorm remains a trusted shortcut, the LM head / vocab output mask are unchanged, the obfuscated KV cache / `prefill` / `decode_step` / `generate_greedy` control flow is unchanged, BERT and T5 wrappers are not modified, and `compatible_islands` is **not** the default mode.
- `scripts/run_gpt2_model_compatible_island_smoke.py` runs full-model forward + greedy generation under both `use_pad ∈ {False, True}` against a hand-written plain HF greedy loop (no `model.generate()`), writes `outputs/gpt2_model_compatible_island_smoke.{json,md}`, and verifies full-forward `allclose=True`, `top1_match_rate = 1.0`, `sequence_exact_match = 1.0`, `token_match_rate = 1.0`, `blocks_with_compatible_islands == num_blocks`, `total_mlp_island_permutation_draws >= num_blocks` per full forward, `online_extra_matmul_count == 0`, `pad_placement ∈ {"linear_boundary_only", "n/a"}`, and `layernorm_remains_trusted == True`. On `sshleifer/tiny-gpt2` (batch=2, seq=8, max_new_tokens=4, fp32) the smoke records `max_abs_error ≈ 4.5e-8` (use_pad=False) / `≈ 6.7e-8` (use_pad=True), `cosine_similarity ≈ 1.0`, `max_logits_error ≈ 3e-8`.
- `tests/test_gpt2_model_compatible_islands.py` — 16 tests covering: model-wrapper mode acceptance + invalid-mode rejection, default mode byte-for-byte equality with explicit `"trusted"`, full-forward correctness vs. plain HF logits (`use_pad ∈ {False, True}`, batch ∈ {1, 2}), greedy generation token-sequence equality vs. plain HF greedy (`use_pad ∈ {False, True}`, batch ∈ {1, 2}), island summary aggregation (active block count, permutation-draw count, zero extra matmul, `linear_boundary_only` pad placement, trusted LayerNorm, untouched LM head / generation path, `gpt2_model_level` scope), `n/a` pad placement when `use_pad=False`, trusted-mode summary inactivity, both modes co-existing, HF GPT-2 / GPT-2 LM head / LayerNorm modules un-replaced after both `forward()` and `generate_greedy()`, and the workload profile recording `gpt2_model_level=implemented` while keeping `implemented=False` / `bert=not_yet` / `t5=not_yet`.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands` now records `wrapper_integration_status.gpt2_model_level = "implemented"` plus `measured_integration_scope = "gpt2_model_level"` and `measured_wall_time_scope = "gpt2_model_level_smoke"`. The top-level `wrapper_integration_status.ours_compatible_nonlinear_islands` mirrors the per-method status with an explanatory note. Crucially `implemented` remains `False` and `wall_time_source` remains `"projected_from_op_counts"` — Stage 5.3b is a partial integration only.
- `outputs/workload_profile.md` gains the lines **`GPT-2 model-level compatible island integration available (Stage 5.3b); BERT/T5 integration pending Stage 5.3c`**, **`measured GPT-2 model-level smoke, not full cross-architecture measurement`**, and **`measured_integration_scope = "gpt2_model_level"`** in the Stage 5.3a Wrapper Integration Status subsection.
- Stage 5.3b does **not** claim formal security; `compatible_islands` remains `proxy-evaluated, not formal` and must stay behind a feature flag. The Stage 5.2b mitigations (fresh permutation per session, dense sandwich at Linear boundaries, pad at Linear boundaries only) remain required, and the measured smoke is explicitly **not** a real TEE measurement nor a full cross-architecture measurement.

Implemented in Stage 5.3a:

- `src/pllo/hf_wrappers/nonlinear_modes.py` — feature-flag enum `nonlinear_mode ∈ {"trusted", "compatible_islands"}` plus `DEFAULT_NONLINEAR_MODE = "trusted"`. Default behaviour is byte-for-byte identical to Stage 4.6 / 4.7 / 4.9; existing tests run unchanged.
- `src/pllo/hf_wrappers/gpt2_block_wrapper.py` — `ObfuscatedGPT2BlockWrapper` now accepts `nonlinear_mode=...` and dispatches its MLP path. Under `nonlinear_mode="compatible_islands"` the GELU MLP is routed through Stage 5.2a's permutation island: a per-call fresh permutation `P` of size `intermediate_size` is absorbed into adjacent Conv1D weights (`W_fc[:, perm]` and `W_proj[perm, :]`), `Z_tilde = X_tilde W_fc_tilde + b_fc_tilde + C_fc = Z P`, `A_tilde = GELU(Z_tilde) = GELU(Z) P` runs on GPU, and `Y_tilde = A_tilde (W_proj[perm, :] @ N_out) + b_proj N_out = Y N_out`. Pad compensation is applied only at the `c_fc` Linear boundary; the pad is never pushed through GELU.
- New per-block `island_report` audit struct exposes `nonlinear_mode`, `mlp_gelu_island_active`, `mlp_island_permutation_dim` (== `intermediate_size`, not `hidden_size`), `mlp_island_pad_placement` (`"linear_boundary_only"` when `use_pad=True`, else `"n/a"`), `mlp_island_uses_fresh_permutation=True`, `mlp_island_permutation_draws`, `online_extra_matmul_count = 0`, `layernorm_remains_trusted=True`, `lm_head_not_modified=True`, `generation_path_not_modified=True`, `security_profile="proxy-evaluated, not formal"`, and the Stage 5.2b security caveats.
- Scope kept tight: LayerNorm remains trusted, the LM head is untouched, the KV cache and `prefill` / `decode_step` paths are not modified, the `ObfuscatedGPT2ModelWrapper` is not modified, BERT / T5 wrappers are not modified, and `compatible_islands` is **not** the default — `trusted` remains the default mode.
- `scripts/run_gpt2_compatible_island_smoke.py` runs one block under both `use_pad=False` and `use_pad=True`, writes `outputs/gpt2_compatible_island_smoke.{json,md}`, and verifies `allclose=True`, `permutation_dim == intermediate_size`, `pad_placement == "linear_boundary_only"` when padded, and `online_extra_matmul_count == 0`. On `sshleifer/tiny-gpt2` (block 0, batch=2, seq=8, fp32): `max_abs_error ≈ 4.77e-7`, `cosine_similarity = 1.0`, `permutation_dim = 8` (= `intermediate_size = 4 * n_embd`).
- `tests/test_gpt2_compatible_islands.py` — 17 tests covering: mode-enum validity, invalid-mode rejection, default-mode byte-for-byte equality with explicit `"trusted"`, trusted-mode `island_report` inactivity, compatible-island correctness vs plain block (`use_pad=False` / `use_pad=True`, float32 / float64), permutation-dim equals `intermediate_size`, `online_extra_matmul_count == 0`, `pad_placement == "linear_boundary_only"`, fresh permutation per forward call, security-caveat reporting, both modes co-existing, HF modules untouched, and block-level `prefill` + `decode_step` recovering plain semantics under `compatible_islands`.
- `src/pllo/experiments/workload_profiler.py` — `methods.ours_compatible_nonlinear_islands` now carries `partial_implementation=True` plus `wrapper_integration_status = {"gpt2_single_block": "implemented", "gpt2_model_level": "not_yet", "bert": "not_yet", "t5": "not_yet"}`. The same dict is mirrored at the top-level `wrapper_integration_status` field. `ours_compatible_nonlinear_islands` is **not** marked `implemented=True` — full-model measured runtime is pending Stage 5.3b.
- `outputs/workload_profile.md` gains a **Stage 5.3a Wrapper Integration Status** subsection under the Compatible Nonlinear Islands Method section, with the integration matrix, the "default mode remains `trusted`" reminder, and the phrase **GPT-2 single-block integration available; full-model measured runtime pending Stage 5.3b**.
- Stage 5.3a does **not** claim formal security; `compatible_islands` remains `proxy-evaluated, not formal` and must stay behind a feature flag in production. Stage 5.2b's mitigations (fresh permutation per session, dense sandwich at Linear boundaries, pad at Linear boundaries only) remain required.

Implemented in Stage 5.2c:

- New workload method `ours_compatible_nonlinear_islands` added to the Stage 5.0.1 registry. It is a *projected* (not measured, not wrapper-integrated) method that models the boundary-call / trusted-compute / preprocessing profile of executing Stage 5.2a's operator-compatible nonlinear islands. Marked `implemented=False`, `wall_time_source="projected_from_op_counts"`, `online_extra_matmul_count=0`, `security_profile="proxy-evaluated, not formal"`. The dataclass extension is purely additive — every pre-existing method gets default `uses_compatible_nonlinear_islands=False` / `online_extra_matmul_count=0` / `security_profile="n/a"`, so Stage 5.0.1 behaviour is preserved.
- Boundary-call formula: **`L + 2`** per forward (1 input mask + L per-layer dense-mask transition between Norm / Activation / MLP islands + 1 LM head). Conservatively modeled: between `ours_current` (`4L + 1`) and `ours_ideal_gpu_nonlinear` (`1`). On `sshleifer/tiny-gpt2` (L=2, 4 forwards, batch=2): 36 → **16** boundary calls vs `ours_current` (55.6% reduction). Trusted-compute drops from 1,116,310 ops (`ours_current`) to **1,105,830 ops** (only LM head recovery + dense-mask residual carry; LN and GELU now run on GPU). GPU ops match `ours_ideal_gpu_nonlinear` (LN / GELU included).
- New interaction-breakdown categories: `preprocessing_affine_folding`, `preprocessing_permutation_absorption`, `compatible_norm_core_gpu`, `compatible_activation_island_gpu`, `dense_sandwich_transition`, `security_proxy_requirements`. The last category carries the Stage 5.2b mitigations (fresh permutation, dense sandwich, pad at Linear boundaries) and the "compatible mask families are weaker than dense" caveat.
- Method-level `paper_metrics.ours_compatible_nonlinear_islands` records `boundary_call_reduction_vs_ours_current`, `trusted_compute_reduction_vs_ours_current`, `preprocessing_cost_increase_vs_ours_current`, `online_extra_matmul_count`, `gpu_offload_ratio`, `projected_not_measured=True`, `security_proxy_available=True`, plus a `security_proxy_caveats` list.
- `outputs/workload_profile.md` now has a dedicated **Compatible Nonlinear Islands Method** section with Boundary Call Formulas / Trusted Compute Reduction / Preprocessing Cost Increase / Online Extra Matmul Count / Security Proxy Caveats subsections, plus the required phrases `"ours_compatible_nonlinear_islands is a projected method"`, `"not yet integrated into GPT-2 / BERT / T5 wrappers"`, `"Compatible mask families are weaker than unrestricted dense masks"`, `"Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations"`, `"online_extra_matmul_count = 0"`, and `"not a real TEE measurement"`.
- `outputs/cross_architecture_summary.md` now has a dedicated **Compatible Nonlinear Island Workload Projection** section emitting one row per architecture (decoder_only / encoder_only / encoder_decoder), each marked `projected_from_probe` with the current vs compatible boundary formulas, trusted-compute / boundary reductions, `online_extra_matmul_count = 0`, and `security_proxy_status = "proxy-evaluated, not formal"`. The Stage 5.2b security caveats are quoted in the same section.

Implemented in Stage 5.2b:

- `src/pllo/experiments/nonlinear_island_security.py` — three security proxies over the Stage 5.2a operator-compatible mask scheme: (1) **permutation recovery** via per-channel ``(mean, std, median, q25, q75, mean_abs)`` signature matched by greedy cosine nearest-neighbour, compared across `fixed_permutation` / `fresh_permutation_per_session` / `permutation_pool` / `dense_sandwich_reference`; (2) **island linkability** of the GPU-visible tensor across requests under `fixed_perm_no_pad` / `fixed_perm_with_linear_boundary_pad` (dual view — boundary + activation) / `fresh_perm_with_linear_boundary_pad` (dual view) / `dense_to_perm_to_dense_sandwich` (triple view); (3) static **mask family security accounting** for `dense_invertible` / `orthogonal` / `mean_preserving_orthogonal` / `permutation` / `paired_permutation`, each with `used_for` / `correctness_role` / `preserved_statistics` / `gpu_visible_leakage` / `mitigation` / `security_strength_relative_to_dense` / `notes` fields.
- `scripts/run_nonlinear_island_security.py` writes `outputs/nonlinear_island_security.{json,csv,md}` (long-format CSV `section,strategy,metric,value,notes`). The Markdown includes a Threat Model section, the three proxy tables, an Interpretation summary, and the spec-mandated Limitations bullet list (security proxies, weaker-than-dense, multiset leakage, fresh-permutation, dense sandwiching, no adaptive attacks, no real TEE, no semantic security claim).
- Output safety: only aggregate metrics, sha-256 fingerprints (where used), and short text are emitted. `tests/test_nonlinear_island_security.py::test_outputs_contain_no_full_mask_tensors` programmatically rejects any numeric array of length ≥ hidden_size in the JSON, plus any `tensor(` / `torch.Tensor` markers in the JSON / CSV / Markdown bodies. No secret mask tensor leaves the trusted side.
- Stage 5.2b validates Stage 5.2a's predicted relations: fixed permutation top-1 recovery (~0.25 at hidden=64) clearly exceeds fresh (~0.10) and the dense sandwich (~0.02, ≈ random chance 1/64); fixed-no-pad linkability cosine ≈ 1.0 vs fresh-with-pad ≈ 0.02 vs sandwich ≈ 0.002. These are *naive-observer upper bounds only* — Stage 5.4 adaptive attackers remain out of scope.

Implemented in Stage 5.2a:

- `src/pllo/ops/compatible_masks.py` — operator-compatible mask family generators: `generate_dense_invertible`, `generate_orthogonal`, `generate_mean_preserving_orthogonal` (orthonormal-basis construction with the all-ones direction as the first vector, so `N 1 = 1` and `N^T C N = C`), and `generate_permutation` (returns both index form `perm`/`inv_perm` and dense matrix form). Plus `orthogonal_error`, `mean_preservation_error`, `centered_orthogonality_error`, `center_matrix`, `matrix_fingerprint` helpers.
- `src/pllo/ops/nonlinear_islands.py` — nonlinear island ops: `layernorm_core` / `rmsnorm_core` (no-affine references), `fold_layernorm_affine_into_linear` / `fold_rmsnorm_affine_into_linear` (offline gamma/beta folding into the following Linear), and the island forwards: `run_rmsnorm_orthogonal_island`, `run_layernorm_mean_preserving_island`, `run_activation_permutation_island` (GELU / ReLU / SiLU), `run_swiglu_paired_permutation_island`, `run_gelu_mlp_island`, `run_swiglu_mlp_island`. Every island folds mask + permutation transitions into the masked weight tensors offline; `online_extra_matmul_count = 0` for every MLP island cell.
- `src/pllo/experiments/nonlinear_island_probe.py` + `scripts/run_nonlinear_island_experiments.py` — 28-cell sweep across norm-compatible, activation-permutation, SwiGLU paired-permutation, and full MLP islands. Writes `outputs/nonlinear_island_experiments.{json,csv,md}`. Markdown explicitly states **"Operator-Compatible Mask Families"**, **"Pad Placement Rule"**, **"Compatible mask families are weaker than unrestricted dense masks"**, and **"Permutation islands hide channel identity but do not hide coordinate-value multisets"**.
- **Pad placement rule**: pad is applied only at Linear boundaries and compensated via `C = T W N_out`; pad is never pushed through an activation. The island input is `(X - T_in) N_in`, the activation input is `Z P` (pad-free), and the island output leaves through another Linear at which downstream code may re-introduce a fresh pad.
- Stage 5.2a is a correctness probe only — it does **not** include the nonlinear-island security proxy (deferred to Stage 5.2b), does **not** integrate the islands into the existing GPT-2 / BERT / T5 wrappers, does **not** implement adaptive permutation-recovery attacks, and does **not** implement real TEE isolation. The mask families used inside nonlinear islands are weaker than unrestricted dense masks; that limitation is recorded in the report.

Implemented in Stage 5.1:

- `src/pllo/ops/norm.py` — unified trusted norm primitive. `TrustedNormConfig` + `trusted_norm_forward(x_tilde, n_in_inv, norm_weight, norm_bias, n_out, norm_type, eps, pad_in=None, pad_out=None)` recovers plaintext `X = x_tilde N_in_inv [+ T_in]`, runs LayerNorm or RMSNorm in the trusted side, then re-masks the output as `Y_tilde = Y N_out` (or `(Y - T_out) N_out`). RMSNorm is supported with `bias=None` (LLaMA / T5 / Qwen style). Returns `y_plain` / `y_tilde` / `y_recovered` plus the headline metric set.
- `src/pllo/experiments/norm_probe.py` — two probes share this module: (1) `run_trusted_norm_probe` drives `trusted_norm_forward` over one `(norm_type, batch_size, seq_len, hidden_size, use_pad)` cell and verifies both the recovered-output invariant and the `y_tilde` shape invariant against a separate reference; (2) `run_rmsnorm_orthogonal_probe` samples QR-orthogonal masks `N` and verifies `N^T N ≈ I`, `rms(X N) ≈ rms(X)`, `normalize(X N) ≈ normalize(X) N`, scalar-gamma commutation, and *non*-commutation of vector gamma (the latter is the headline restriction).
- `scripts/run_norm_experiments.py` sweeps `norm_type ∈ {layernorm, rmsnorm}`, `batch ∈ {1, 2}`, `seq ∈ {4, 8}`, `hidden ∈ {64, 128}`, `use_pad ∈ {true, false}` (32 trusted cells) plus two orthogonal-probe cells (`hidden ∈ {64, 128}`, 16 trials each). Writes `outputs/norm_experiments.{json,csv,md}` whose Markdown explicitly states **"General right masks do not commute with LayerNorm"** and **"Vector gamma breaks simple right-mask commutation"**.
- This stage **standardises** the trusted LayerNorm shortcut behind one primitive name but does **not** yet eliminate trusted compute, does **not** implement a GPU-side norm protocol for the general right-mask family, and does **not** claim formal security. The orthogonal-mask result is a feasibility note for a future restricted-mask protocol — it is not used by any existing wrapper yet.

Implemented in Stage 6.3:

- `src/pllo/experiments/cross_architecture_summary.py` — pure aggregator over Stage 5.0 / 6.0 / 6.1 / 6.2 JSON artifacts plus the Stage 5.0.1 workload profile. Produces one unified summary across the three architectures with `architecture_type`, `model_id`, `attention_kind`, `cache_type`, `num_cells`, `num_rows`, `all_loaded_allclose`, `max_output_error` / `max_score_error` / `max_prob_error` / `max_cache_error`, `use_pad_supported`, `padding_mask_supported`, `bias_present`, `has_relative_attention_bias`, per-architecture trusted shortcuts and limitations. Missing upstream JSONs are recorded as `status="missing"` unless `require_existing_outputs=True`. `scripts/run_cross_architecture_summary.py` writes `outputs/cross_architecture_summary.{json,csv,md}`; an opt-in `--rerun-upstream` flag re-executes the upstream sweeps before aggregating.
- `src/pllo/experiments/security_proxy.py` — four lightweight security proxy experiments: (1) pad-vs-no-pad pairwise linkability across `fixed_mask_no_pad` / `fresh_mask_no_pad` / `fixed_mask_fresh_pad` / `fresh_mask_fresh_pad`; (2) mask freshness / uniqueness audit using sha256 fingerprints over per-trial generated masks (mask contents are never emitted, only counts and condition-number aggregates); (3) static boundary leakage accounting partitioning every simulated tensor (`obfuscated_input`, `transformed_linear_weight`, `compensation_terms`, `obfuscated_q/k/v`, `obfuscated_kv_cache`, `obfuscated_encoder_memory_cache`, `obfuscated_logits`, ...) into `gpu_visible` vs `trusted_only` with a per-item leakage note; (4) cache leakage proxy that nearest-neighbour-matches plain K/V against `K_tilde = K N_K` / `V_tilde = V N_V` for both the KV cache and the encoder memory cache. `scripts/run_security_proxy_experiments.py` writes `outputs/security_proxy_experiments.{json,csv,md}`.
- All four proxies are explicitly upper bounds on naive-observer adversary success. The report's Limitations section states that they are **not** formal security proofs, do **not** implement adaptive or learned inversion attacks, do **not** evaluate real TEE isolation, do **not** cover side channels, and do **not** prove LoRA adapter extraction resistance.

## Not Included

This project currently does **not** include:

- RMSNorm
- ModelScope integration
- Real TEE integration
- GPT-2 module replacement
- Direct use of HuggingFace `past_key_values` inside the wrapper (it is consumed only as a plaintext reference)
- HuggingFace `generate()` integration (Stage 4.9 implements its own greedy loop on top of `prefill()` / `decode_step()`)
- Obfuscated LayerNorm (trusted shortcut used in Stages 2–4.9)
- Obfuscated GELU (trusted shortcut used in Stages 2–4.9)
- LM head one-time pad (vocab output mask only in Stages 4.7–4.9)
- Qwen2.5 / Llama / TinyLlama loading
- Beam search, top-k, top-p, sampling, temperature, or EOS early-stop

## Installation

```bash
pip install -e ".[dev]"
```

The `dev` extra includes HuggingFace dependencies. You can also install them explicitly:

```bash
pip install -e ".[dev,hf]"
```

## Run Tests

```bash
pytest
```

## Run Correctness Scripts

```bash
python scripts/run_static_correctness.py
python scripts/run_lora_correctness.py
python scripts/run_tiny_transformer_correctness.py
python scripts/run_kv_cache_correctness.py
python scripts/run_generation_correctness.py
python scripts/inspect_hf_model.py --model-id sshleifer/tiny-gpt2
python scripts/run_hf_gpt2_smoke.py --model-id sshleifer/tiny-gpt2
python scripts/run_gpt2_conv1d_mapping.py --model-id sshleifer/tiny-gpt2
python scripts/run_gpt2_block_correctness.py --model-id sshleifer/tiny-gpt2
python scripts/run_gpt2_model_correctness.py --model-id sshleifer/tiny-gpt2
python scripts/run_gpt2_model_correctness.py --model-id sshleifer/tiny-gpt2 --use-pad false
python scripts/run_gpt2_cache_correctness.py --model-id sshleifer/tiny-gpt2
python scripts/run_gpt2_cache_correctness.py --model-id sshleifer/tiny-gpt2 --use-pad false
python scripts/run_gpt2_generation_correctness.py --model-id sshleifer/tiny-gpt2
python scripts/run_gpt2_generation_correctness.py --model-id sshleifer/tiny-gpt2 --use-pad false
python scripts/run_experiment_summary.py --rerun
python scripts/run_attention_experiments.py
python scripts/run_workload_profile.py
python scripts/run_gpt2_compatible_island_smoke.py
python scripts/run_gpt2_model_compatible_island_smoke.py
python scripts/run_cross_architecture_compatible_island_smoke.py
python scripts/run_adaptive_island_attacks.py
python scripts/run_modern_decoder_probe.py
python scripts/run_architecture_coverage.py
python scripts/run_encoder_attention_experiments.py
python scripts/run_cross_attention_experiments.py
python scripts/run_cross_architecture_summary.py
python scripts/run_security_proxy_experiments.py
python scripts/run_norm_experiments.py
python scripts/run_nonlinear_island_experiments.py
python scripts/run_nonlinear_island_security.py
```

Useful variants:

```bash
python scripts/run_static_correctness.py --dtype float32 --no-use-pad --no-bias
python scripts/run_lora_correctness.py --dtype float32 --pad-scale 0.5
python scripts/run_tiny_transformer_correctness.py --dtype float32
python scripts/run_kv_cache_correctness.py --dtype float32
python scripts/run_generation_correctness.py --dtype float32
```

By default, results are written to:

- `outputs/static_correctness.json`
- `outputs/lora_correctness.json`
- `outputs/tiny_transformer_correctness.json`
- `outputs/kv_cache_correctness.json`
- `outputs/generation_correctness.json`
- `outputs/hf_model_inspection.json`
- `outputs/hf_gpt2_smoke.json`
- `outputs/gpt2_conv1d_mapping.json`
- `outputs/gpt2_block_correctness.json`
- `outputs/gpt2_model_correctness.json`
- `outputs/gpt2_cache_correctness.json`
- `outputs/gpt2_generation_correctness.json`
- `outputs/experiment_summary.json` / `.csv` / `.md` (Stage 4.10 aggregator)
- `outputs/attention_experiments.json` / `.csv` / `.md` (Stage 5.0 attention probe)
- `outputs/workload_profile.json` / `.csv` / `.md` (Stage 5.0 workload profiler)
- `outputs/architecture_coverage.json` / `.csv` / `.md` (Stage 6.0 architecture coverage)
- `outputs/encoder_attention_experiments.json` / `.csv` / `.md` (Stage 6.1 encoder-only attention probe)
- `outputs/cross_attention_experiments.json` / `.csv` / `.md` (Stage 6.2 encoder-decoder cross-attention probe)
- `outputs/cross_architecture_summary.json` / `.csv` / `.md` (Stage 6.3 cross-architecture coverage + correctness + workload aggregator)
- `outputs/security_proxy_experiments.json` / `.csv` / `.md` (Stage 6.3 security proxy experiments — pad linkability, mask freshness, boundary leakage accounting, cache leakage proxy)
- `outputs/norm_experiments.json` / `.csv` / `.md` (Stage 5.1 norm primitive — trusted LayerNorm / RMSNorm correctness + restricted RMSNorm orthogonal-mask feasibility probe)
- `outputs/nonlinear_island_experiments.json` / `.csv` / `.md` (Stage 5.2a nonlinear-island correctness — norm-compatible / activation-permutation / SwiGLU paired / full MLP)
- `outputs/nonlinear_island_security.json` / `.csv` / `.md` (Stage 5.2b nonlinear-island security proxies — permutation recovery, island linkability dual/triple views, mask family accounting)

Each JSON file reports:

- `max_abs_error`
- `mean_abs_error`
- `relative_l2_error`
- `cosine_similarity`
- `allclose`
- `top1_match_rate` for the tiny Transformer logits script
- `token_match_rate` and `sequence_exact_match` for greedy generation
- cache invariant metrics for Stage 3 K/V cache checks

## Next Stage Plan

Planned extensions after Stage 5.6:

- Stage 5.6 extension (optional) — Implement the full `masked_boundary_experimental` mode at the `ObfuscatedModernDecoderModelWrapper` level so the inter-block residual stays masked across layers under an orthogonal `N_inter`. Requires modifying the attention path to consume `x_tilde` directly (the q/k/v fold and the residual add must absorb the same `N_inter`) and the LM head to recover the final `N_inter`. Together with constant-time decode (latency-equalising delays per decode step) this closes the structural-leakage half of Stage 5.6's finding and would let `security_profile` move toward `"adaptive-blackbox-timing-and-structural-proxy-evaluated, not formal"`.
- Stage 7.0 (deferred) — LoRA private-training path under the same obfuscation envelope: mask scheduling under autograd, fresh-mask budget per step, gradient-side trace surface. Requires Stage 5.6 extension to first close the inter-block residual gap.
- Stage 5.3d (deferred — engineering-heavy, paper-marginal) — Full BERT and T5 obfuscated wrappers (not just probes). Stage 5.3e provides the per-call bundle wiring; Stage 5.3d wraps it into BertModel / T5Model `forward` + KV cache + LM head. Only after this can `ours_compatible_nonlinear_islands.implemented` flip to `True`, `wall_time_source` to `measured`, and `full_runtime_integrated` to `True`.
- Stage 8.0 (deferred — security) — Hardware-side-channel evaluation (cache / power / EM) on a real TEE deployment. Requires a real TEE budget and is explicitly out of the current research scope.
