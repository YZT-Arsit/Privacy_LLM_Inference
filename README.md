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

Planned extensions after Stage 5.0:

- Stage 5.1 — GPU-side LayerNorm primitive prototype (replaces the trusted shortcut)
- Stage 5.2 — GELU primitive feasibility (replaces the trusted shortcut)
- Stage 5.3 — security proxy experiments (information leakage, mask uniqueness, pad freshness)
- Stage 5.4 — Qwen / ModelScope migration (full Qwen2.5 wrapper, dual-source loading)
