# ObfuscaTune-Qwen integration

Stage 2 of the ObfuscaTune baseline: adapt the paper's obfuscation scheme
(arXiv:2407.02960) to **Qwen2 / Qwen2.5** decoder-only models and wire it into
our experiment registry + unified comparison schema, so it can be compared
apples-to-apples against our amulet-style / trusted-shortcut scheme.

Everything lives under the `obfuscatune` namespace
(`src/pllo/baselines/obfuscatune/qwen_*.py`, `scripts/obfuscatune/run_qwen_*.py`,
`tests/obfuscatune/test_qwen_*.py`). It does **not** modify amulet-style /
trusted-shortcut code. Simulator only — **no real TEE**.

## 1. Why an ObfuscaTune-Qwen baseline

The paper demonstrates on GPT-2/nanoGPT. Our whole evaluation is on Qwen (KV
cache, GQA, RoPE, SwiGLU, folded-remote). To place ObfuscaTune in the same
tables we need it running on the *same architecture* and emitting the *same
fields* — otherwise the numbers are not comparable. This stage provides that.

## 2. Threat-model difference (do not conflate)

| | ObfuscaTune | Ours (amulet-style / A_rightmul) |
|---|---|---|
| Protects | proprietary **secret model weights** + private data | user **input** / LoRA / KV cache / logits |
| Base model | **secret** (obfuscated, `requires_private_base_model=True`) | **public** |
| Q/K/V outside TEE | **plaintext (exposed)** | masked |
| Attention scores | plaintext in TEE, but Q/K/V public | masked |
| KV cache | **plaintext intermediate exposed** | protected |
| Non-linear (RMSNorm/RoPE/SiLU/softmax) | **inside TEE** | on untrusted GPU (single entry/exit) |
| TEE | authenticated (assumed) | attested (TDX) |

The registry entry and every result row carry a `notes` field stating this.
Only **correctness** and **cost** columns are directly comparable.

## 3. Qwen architecture adaptation points

- **RMSNorm** (not LayerNorm): runs in the simulated TEE on plaintext.
  `qwen_rmsnorm_reference` matches `Qwen2RMSNorm` (fp32 variance, cast back).
  Note RMSNorm is *equivariant* under an orthogonal mask (norm-preserving), but
  ObfuscaTune keeps it in the TEE for generality (non-orthogonal/conditioned
  masks break equivariance).
- **RoPE**: applied to the TRUE plaintext Q/K in the TEE domain
  (`rope_plain_domain=True`); reuses HF `apply_rotary_pos_emb` for exact match.
- **GQA** (`num_key_value_heads != num_attention_heads`): `repeat_kv` after
  RoPE; the KV cache stores `n_kv` heads (pre-repeat), matching HF's cache
  layout. Tested with heads=4 / kv=2.
- **SwiGLU MLP**: `gate_proj`/`up_proj` are input-obfuscated (their masks cancel
  → true plaintext gate/up on the untrusted side); `silu(gate)*up` runs in the
  TEE; `down_proj` is output-obfuscated and de-obfuscated in the TEE.
- **KV cache**: `QwenObfCache` holds post-RoPE plaintext K/V per layer
  (`cache_obfuscation="plain_intermediate_exposed"`). Prefill/decode append and
  causal masking with an absolute-position offset match HF recompute.
- **LM head**: final RMSNorm + `lm_head` run in the TEE on de-obfuscated hidden.

## 4. Computation split (simulated TEE / outside TEE)

- **Simulated TEE (plaintext)**: token embedding, RMSNorm, RoPE, SiLU+SwiGLU
  product, softmax, residual adds, bias, de-/re-obfuscation, final norm, LM
  head / sampling, cache bookkeeping.
- **Outside TEE (obfuscated matmuls)**: `q/k/v/o_proj`, `gate/up/down_proj`.
- **Exposed plaintext intermediates** (paper semantics, `public_qkv=True`):
  Q, K, V, attention scores, MLP gate/up, and the KV cache.

## 5. Support granularity

Linear ✅ · MLP (SwiGLU) ✅ · Attention (RoPE+GQA) ✅ · Full decoder block ✅ ·
Full model ✅ · Prefill ✅ · Decode (KV cache) ✅ · Greedy generation ✅ ·
LoRA ❌ (not implemented for Qwen) · real TEE ❌ (simulated) · finetuning ❌.

## 6. File map

```
src/pllo/baselines/obfuscatune/
  qwen_config.py      # tiny random Qwen2 config + offline load + robust module discovery + QwenArch
  qwen_modules.py     # obfuscated attention (RMSNorm/RoPE/GQA) + SwiGLU MLP + decoder block
  qwen_cache.py       # QwenObfCache (plaintext K/V) + cache_metrics + hf_cache_shapes
  qwen_obfuscatune.py # full forward, run_prefill / run_decode_step / generate_greedy
  qwen_metrics.py     # unified schema (correctness/cache/security_proxy/cost/numerical) + method_id
scripts/obfuscatune/
  run_qwen_correctness_check.py       run_qwen_prefill_decode_check.py
  run_qwen_generation_smoke.py        run_qwen_condition_number_sweep.py
  run_qwen_latency_profile.py         run_qwen_vs_amulet_summary.py
tests/obfuscatune/
  test_qwen_rmsnorm_boundary.py       test_qwen_attention_obfuscation.py
  test_qwen_mlp_swiglu_obfuscation.py test_qwen_kv_cache_correctness.py
  test_qwen_generation_smoke.py       test_qwen_experiment_registry.py
```
Registry: `obfuscatune_qwen_{orthogonal,random,cond_8,cond_32,cond_128}` in
`src/pllo/experiments/experiment_registry.py` (`BASELINE_METHODS` /
`OBFUSCATUNE_QWEN_METHODS`, separate from `WORKLOAD_METHODS`).

## 7. Local minimal run (never downloads)

```bash
python scripts/obfuscatune/run_qwen_correctness_check.py --tiny-random-qwen
python scripts/obfuscatune/run_qwen_prefill_decode_check.py --tiny-random-qwen
python scripts/obfuscatune/run_qwen_generation_smoke.py --tiny-random-qwen --max-new-tokens 8
python scripts/obfuscatune/run_qwen_condition_number_sweep.py --tiny-random-qwen
python scripts/obfuscatune/run_qwen_latency_profile.py --tiny-random-qwen --num-runs 10
python scripts/obfuscatune/run_qwen_vs_amulet_summary.py
# any script: add --dry-run to print the planned config only.
```

Expected (tiny random Qwen, GQA 4/2): orthogonal prefill max-abs logit error
~1e-7 (fp32) / ~0 (fp64), argmax match 1.0, greedy token match 1.0; decode cache
len prefill+1, cache shape matches HF; condition-number error grows monotonically
with κ, random worst.

### Using a local Qwen checkpoint (offline)

```bash
python scripts/obfuscatune/run_qwen_correctness_check.py \
    --model-name-or-path /path/to/Qwen2.5-0.5B --seq-len 64 --dtype float32
```
The loader uses `local_files_only=True`. If the path is missing, the error tells
you to pass a local path or `--tiny-random-qwen`. To fetch a model, use
ModelScope/HF **manually** (not run here), e.g.
`modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ...`.

## 8. Server-side commands (suggested; do NOT run here)

```bash
# real Qwen2.5-7B, fp32 prefill/decode correctness + latency on GPU
python scripts/obfuscatune/run_qwen_correctness_check.py --model-name-or-path $QWEN7B --seq-len 512 --dtype float32 --device cuda
python scripts/obfuscatune/run_qwen_prefill_decode_check.py --model-name-or-path $QWEN7B --seq-len 512 --device cuda
python scripts/obfuscatune/run_qwen_latency_profile.py --model-name-or-path $QWEN7B --seq-len 512 --num-runs 50 --device cuda
python scripts/obfuscatune/run_qwen_vs_amulet_summary.py --amulet-glob 'results/ours_amulet/**/*.json'
```

## 9. Output JSON/CSV interpretation

Each result row (see `qwen_metrics.qwen_result_row`):
`method`, `model_family="qwen"`, `mode` (prefill/decode/generation), model dims,
and blocks:
- **correctness**: `max_abs_error`, `mean_abs_error`, `relative_l2_error`,
  `logits_argmax_match_rate`, `token_match_rate`, `exact_token_match`.
- **cache**: `enabled`, `prefill_cache_len`, `decode_cache_len`,
  `cache_shape_match`, `cache_obfuscation` (`plain_intermediate_exposed`).
- **security_proxy**: `protected_input` (partial), `protected_model_weights`
  (true), `protected_lora` (not_implemented), `protected_kv_cache` (false),
  `protected_logits` (true), `public_qkv`/`public_attention_scores` (true),
  `tee_auth_assumed` (true), `notes`.
- **cost**: `wall_time_ms`, `slowdown_vs_unprotected`,
  `boundary_calls_per_forward`, `trusted_transfer_bytes`, `tee_param_ratio`,
  `gpu_param_ratio` (CPU-simulator proxy; not a real TEE measurement).
- **numerical**: `random_matrix_type` (orthogonal/gaussian/conditioned),
  `condition_number_target/mean/max`, `inverse_method`
  (transpose/inverse/svd_constructed), `nan_or_inf_count`.

## 10. Comparing with amulet-style

`run_qwen_vs_amulet_summary.py` merges ObfuscaTune-Qwen rows (and optional
amulet rows via `--amulet-glob`) into `outputs/obfuscatune/qwen_vs_amulet_summary.{md,csv}`.
Keep the threat-model note: **correctness + cost are comparable; security columns
describe different guarantees**, and `protected_kv_cache` must stay `false` for
ObfuscaTune (plaintext K/V exposed) unless a real cache-obfuscation primitive is
added and tested.

## 11. Limitations & next steps

- **Simulated TEE** only; the split is bookkeeping + proxy counters.
- Correctness validated on **tiny random Qwen2** (GQA) offline; real checkpoints
  supported via `--model-name-or-path` but untested here.
- **No LoRA** for Qwen (`protected_lora="not_implemented"`); **no finetuning**.
- **KV cache is plaintext** (paper semantics) — never labeled protected.
- Security is **structural annotation + proxy**; no attack evaluation.
- Latency is a **CPU-simulator wall-time proxy**, not a TEE deployment number.
- Next: run on a real Qwen2.5 checkpoint on GPU; merge with amulet-style rows;
  optionally add an *experimental* Q/K/V-obfuscated attention variant (kept as
  non-default) if it can preserve correctness.
