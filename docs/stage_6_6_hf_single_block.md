# Stage 6.6 — HF/ModelScope LLaMA/Qwen Single-Decoder-Layer Adapter

Maps the Stage 6.5 synthetic-block masking design onto **real** HuggingFace
LLaMA / Qwen2 decoder-layer weights, and verifies the masked path
reproduces the plain reference computed from the *same extracted weights*.

**Not** a full model. No tokenizer, embedding, LM head, sampling, or
generation loop. transformers is an **optional** dependency; any model
loading is `local_files_only=True` (no network). No formal, cryptographic,
or semantic security is claimed.

## Flow

1. **Introspect** a `LlamaDecoderLayer` / `Qwen2DecoderLayer` →
   `HFSingleBlockConfig` (`infer_config_from_hf_layer`; prefers a passed
   `model_config`, falls back to `layer.self_attn.config`).
2. **Extract** weights to row-vector convention (`extract_hf_single_block_weights`):
   HF `Linear` stores `[out, in]`, so `W_internal = weight.detach().T`;
   optional biases preserved; tensors cloned/cast to float64; the original
   layer is never mutated.
3. **Plain reference** forward from the extracted weights
   (`hf_single_block_plain_prefill`) — *not* the HF layer's own `forward`,
   since HF attention/RoPE internals vary across versions. Adjacent-pair
   RoPE (Stage 6.4) is used for both plain and masked paths, so the
   masked-vs-plain invariant holds regardless of HF's RoPE convention.
4. **Fold + mask** (`fold_hf_single_block_weights`, bias-aware) and verify
   `y_tilde == y_plain @ n_res` plus every intermediate invariant.

## Weight + bias folding (with optional bias)

| weight | folded form | bias |
|---|---|---|
| `Wq̃` | `n_res⁻¹·diag(rms1)·Wq·blockdiag(Mq)` | `bq·blockdiag(Mq)` |
| `Wk̃` | `n_res⁻¹·diag(rms1)·Wk·blockdiag(Mk)` | `bk·blockdiag(Mk)` |
| `Wṽ` | `n_res⁻¹·diag(rms1)·Wv·blockdiag(Mv)` | `bv·blockdiag(Mv)` |
| `Wõ` | `blockdiag(Vinv_qhead)·Wo·n_res` | `bo·n_res` |
| `Wgatẽ` | `n_res⁻¹·diag(rms2)·Wgate[:,perm]` | `bgate[perm]` |
| `Wuũ` | `n_res⁻¹·diag(rms2)·Wup[:,perm]` | `bup[perm]` |
| `Wdowñ` | `Wdown[perm,:]·n_res` | `bdown·n_res` |

LLaMA has no biases (all `None`); Qwen2 has `q/k/v` biases (`o_proj` and MLP
bias-free) — both verified.

## Files

- `src/pllo/hf_wrappers/llama_qwen_single_block.py` — optional-dependency
  helpers, config introspection, extraction, plain forward, mask + fold,
  masked prefill/decode, random-layer constructor.
- `src/pllo/experiments/hf_single_block_probe.py` — local-checkpoint or
  random-layer probe; clean `skipped` status when transformers / a family /
  a local path is unavailable.
- `scripts/run_hf_single_block_probe.py` — CLI (`--model-family`,
  `--local-model-path`, `--layer-index`).
- `tests/test_hf_single_block_wrapper.py` — 11 tests (skip cleanly without
  transformers).

## Result

Random tiny LLaMA and Qwen2 layers verify at float64 machine precision
(final-output error ≤1.8e-15), prefill and multi-step decode, with the KV
cache append invariants holding under the RoPE-compatible masks.

## Limitations / next stage

Single decoder layer only; extracted-weight reference (not HF generation);
no tokenizer/embedding/LM-head/sampling; no NTK/YaRN RoPE scaling; attention
scores remain GPU-visible; KV-cache masks reused within a session.

**Stage 6.7** — embedding/input masking and the LM-head/logits/sampling
boundary for real-model inference.
