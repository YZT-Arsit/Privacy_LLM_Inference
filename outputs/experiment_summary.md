# Privacy LLM Obfuscation — Experiment Summary (Stage 4.10)

This document aggregates the per-stage correctness JSON files into a single reproducibility report. Numbers below are read from `outputs/_summary_runs/*.json` (fresh `--rerun`).

## Stage Coverage

| Stage | Title | Pad variants | Source script |
|---|---|---|---|
| 1 | Static Linear (mask + pad) | yes | `scripts/run_static_correctness.py` |
| 1-lora | LoRA Linear (independent low-rank branch) | yes | `scripts/run_lora_correctness.py` |
| 2 | Tiny decoder-only Transformer (full sequence) | single | `scripts/run_tiny_transformer_correctness.py` |
| 3-cache | Tiny Transformer prefill / decode / KV cache | single | `scripts/run_kv_cache_correctness.py` |
| 3-gen | Tiny Transformer greedy generation | single | `scripts/run_generation_correctness.py` |
| 4.6 | GPT-2 single-block obfuscated wrapper | yes | `scripts/run_gpt2_block_correctness.py` |
| 4.7 | GPT-2 multi-block full forward logits | yes | `scripts/run_gpt2_model_correctness.py` |
| 4.8 | GPT-2 prefill / decode / KV cache | yes | `scripts/run_gpt2_cache_correctness.py` |
| 4.9 | GPT-2 greedy generation | yes | `scripts/run_gpt2_generation_correctness.py` |

## Trusted-side Engineering Shortcuts (still active)

| Stage | Trusted shortcuts |
|---|---|
| 1 | Mask & pad generation lives in SimulatedTEE. |
| 1-lora | Mask & pad generation lives in SimulatedTEE. |
| 2 | Trusted LayerNorm (Stage 2 shortcut).; Trusted GELU (MLP activation evaluated in plaintext). |
| 3-cache | Trusted LayerNorm carried forward from Stage 2.; Trusted GELU carried forward from Stage 2. |
| 3-gen | Trusted LayerNorm carried forward from Stage 2.; Trusted GELU carried forward from Stage 2.; No sampling / beam search / EOS early-stop. |
| 4.6 | Trusted LayerNorm (ln_1 / ln_2 on plaintext).; Trusted GELU.; HF GPT-2 model is not modified (Conv1D-as-linear extraction). |
| 4.7 | Trusted LayerNorm (ln_1 / ln_2 / ln_f on plaintext).; Trusted GELU.; LM head: diagonal vocab output mask only — no pad. |
| 4.8 | Trusted LayerNorm.; Trusted GELU.; LM head: vocab output mask only.; HF past_key_values used only as plaintext reference. |
| 4.9 | Trusted LayerNorm.; Trusted GELU.; LM head: vocab output mask only.; Greedy only — no sampling / beam / EOS early-stop. |

## Per-Stage Metrics

### Stage 1 — Static Linear (mask + pad)

Right-multiply mask + one-time pad correctness for a standalone linear layer.

| metric | use_pad=true | use_pad=false |
|---|---|---|
| max_abs_error | 1.066e-14 | 9.770e-15 |
| allclose | true | true |

### Stage 1-lora — LoRA Linear (independent low-rank branch)

Mask + pad correctness for a LoRA-adapted linear where base weight and adapter are obfuscated separately.

| metric | use_pad=true | use_pad=false |
|---|---|---|
| max_abs_error | 2.709e-14 | 1.954e-14 |
| allclose | true | true |

### Stage 2 — Tiny decoder-only Transformer (full sequence)

End-to-end obfuscated forward through a hand-written tiny Transformer; logits compared against the plain reference.

| metric | use_pad=true |
|---|---|
| max_abs_error | 1.110e-14 |
| allclose | true |
| top1 | 1.000e+00 |

### Stage 3-cache — Tiny Transformer prefill / decode / KV cache

Prefill + decode correctness with persistent obfuscated K/V cache; per-layer K_tilde / V_tilde invariants validated.

| metric | use_pad=true |
|---|---|
| prefill_max_err | 9.243e-15 |
| prefill_allclose | true |
| decode_max_err | 6.189e-15 |
| decode_allclose | true |
| cache_max_key_err | 6.696e-15 |
| cache_max_val_err | 8.604e-15 |
| cache_allclose | true |

### Stage 3-gen — Tiny Transformer greedy generation

Greedy generation correctness for the tiny Transformer. Compares plain vs obfuscated token sequences.

| metric | use_pad=true |
|---|---|
| token_match | 1.000e+00 |
| seq_exact | 1.000e+00 |

### Stage 4.6 — GPT-2 single-block obfuscated wrapper

Single HuggingFace GPT-2 block obfuscated via fused c_attn block-diagonal Q/K/V masks; hidden states compared against the plain HF block.

| metric | use_pad=true | use_pad=false |
|---|---|---|
| max_abs_error | 2.384e-07 | 4.768e-07 |
| allclose | true | true |

### Stage 4.7 — GPT-2 multi-block full forward logits

Full GPT-2 forward through chained ObfuscatedGPT2BlockWrapper instances; diagonal vocab output mask on the LM head.

| metric | use_pad=true | use_pad=false |
|---|---|---|
| max_abs_error | 2.086e-07 | 8.196e-08 |
| allclose | true | true |
| top1 | 1.000e+00 | 1.000e+00 |

### Stage 4.8 — GPT-2 prefill / decode / KV cache

Internal ObfuscatedGPT2KVCache, prefill + decode_step, per-head K/V mask reuse, K_tilde / V_tilde invariants.

| metric | use_pad=true | use_pad=false |
|---|---|---|
| prefill_max_err | 3.353e-07 | 1.006e-07 |
| prefill_allclose | true | true |
| decode_max_err | 2.747e-07 | 2.235e-08 |
| decode_allclose | true | true |
| cache_max_key_err | 9.313e-09 | 2.328e-09 |
| cache_max_val_err | 1.118e-08 | 7.451e-09 |
| cache_allclose | true | true |

### Stage 4.9 — GPT-2 greedy generation

generate_greedy() built directly on prefill + decode_step. HF generate() is not called; no sampling / beam / EOS.

| metric | use_pad=true | use_pad=false |
|---|---|---|
| token_match | 1.000e+00 | 1.000e+00 |
| seq_exact | 1.000e+00 | 1.000e+00 |
| logits_max_err | 2.235e-08 | 6.706e-08 |
| logits_allclose | true | true |
| cache_max_key_err | 4.191e-09 | 1.863e-09 |
| cache_max_val_err | 1.490e-08 | 4.657e-09 |
| cache_allclose | true | true |

## Reproducibility — How to Regenerate Everything

```bash
pip install -e ".[dev,hf]"
pytest
python scripts/run_experiment_summary.py --rerun
```

The `--rerun` flag drives each upstream correctness script for both `use_pad=true` and `use_pad=false` (where applicable) and writes them to `outputs/_summary_runs/`. Without `--rerun` the aggregator reads the snapshot files already present in `outputs/`.
