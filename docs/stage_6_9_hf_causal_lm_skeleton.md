# Stage 6.9 — HF Full-Model / Local Tiny-Checkpoint Masked CausalLM Skeleton

This stage integrates the three prior masked-pipeline pieces into a
*whole-model* path driven by weights **extracted** from a HuggingFace-style
LLaMA / Qwen2 `...ForCausalLM`:

- **Stage 6.6** ([llama_qwen_single_block.py](../src/pllo/hf_wrappers/llama_qwen_single_block.py)) — per-layer weight extraction, RoPE/GQA-compatible masks, affine+mask folding, bias-aware masked single-layer forward.
- **Stage 6.7** ([causal_lm_boundaries.py](../src/pllo/ops/causal_lm_boundaries.py)) — trusted embedding input boundary, vocab-logit mask, masked-logits output boundary, TEE recovery + sampling.
- **Stage 6.8** ([masked_causal_lm_skeleton.py](../src/pllo/ops/masked_causal_lm_skeleton.py)) — per-layer residual masks `N_0..N_L` with an honest, explicit handoff.

The goal is **not** production generation. It validates that a HF CausalLM
decomposes into `embedding → decoder layers → final norm → LM head` and that
the masked pipeline reproduces our extracted-weight plaintext reference to
machine precision, including a bounded greedy decode loop.

## Why the reference is our extracted-weight forward (not `model.forward`)

HF attention / RoPE / KV-cache conventions vary across `transformers`
versions, so we do **not** compare against `model.forward` or `model.generate`.
Both our plain and masked paths use the same adjacent-pair RoPE (Stage 6.4),
so the verified quantity is the masked-vs-plain invariant `y_tilde == y_plain @ N`
— independent of HF internals.

## Modes

1. **`random_tiny_hf_model`** (default) — instantiate a tiny LLaMA/Qwen2
   `...ForCausalLM` from config (no checkpoint, no download), extract layers,
   run masked prefill + bounded greedy decode vs the extracted-weight plain
   reference.
2. **`local_tiny_checkpoint`** — if `local_model_path` is given, load with
   `local_files_only=True`. Missing path → `skipped_local_model_unavailable`;
   `vocab_size > max_vocab_size` (or oversized hidden) → `skipped_local_model_too_large`.
   Never crashes, never downloads.

Tiny config: `hidden=32`, `intermediate=64`, `heads=4`, `kv_heads=2`,
`max_pos=64`, `rms_eps=1e-5`, `rope_theta=10000`, `tie_word_embeddings=False`,
`vocab=min(max_vocab_size, 512)`.

## Input pad default

`use_input_pad=False` by default for real-HF fidelity: an input pad `T_in`
shifts the modelled sequence through RMSNorm (Stage 6.8), so true fidelity
needs `T_in = 0`. `use_input_pad=True` is allowed only as a synthetic stress
option.

## Mask handoff honesty (carried from Stage 6.8)

Per-layer residual masks make a pre-norm block's output mask equal its input
mask, so the change of basis `T_ell = N_ell^{-1} @ N_{ell+1}` is applied as
**one `[H,H]` GEMM** on the masked hidden state at each layer boundary. The
non-skip terms are offline-fusable; the skip term genuinely needs the GEMM.
Metadata records `handoff_skip_term_needs_gemm = True`. No claim of zero
handoff cost.

## Verified invariants (CPU, float64, atol=rtol=1e-8)

Random tiny models, `max_layers=2`, `prefill_seq_len=4`, `decode_steps=2`,
`vocab≤256` — observed max-abs errors (machine precision):

| metric | LLaMA | Qwen2 |
|---|---|---|
| embedding mask | 0 | 0 |
| per-layer (output/score/mlp/cache) | 1.7e-16 | 3.5e-16 |
| per-layer handoff invariant | 4.2e-17 | 3.8e-17 |
| final hidden | 4.2e-17 | 3.8e-17 |
| masked logits | 5.0e-16 | 5.8e-16 |
| recovered logits | 2.8e-16 | 3.6e-16 |
| greedy token match | 1.0 | 1.0 |
| decode token-match rate | 1.0 | 1.0 |

## Files

- [src/pllo/hf_wrappers/hf_causal_lm_skeleton.py](../src/pllo/hf_wrappers/hf_causal_lm_skeleton.py)
- [src/pllo/experiments/hf_causal_lm_skeleton_probe.py](../src/pllo/experiments/hf_causal_lm_skeleton_probe.py)
- [scripts/run_hf_causal_lm_skeleton_probe.py](../scripts/run_hf_causal_lm_skeleton_probe.py) → `outputs/hf_causal_lm_skeleton_probe_{llama,qwen2}.{json,md}` (compact, no tensor dumps)
- [tests/test_hf_causal_lm_skeleton.py](../tests/test_hf_causal_lm_skeleton.py) — 15 tests (skip cleanly without transformers)

## Boundary / leakage surface

GPU sees: masked embeddings, masked hidden states, masked KV caches, masked
logits. GPU never sees: `input_ids`, plaintext embeddings, plaintext logits,
sampled token ids. **Attention scores remain GPU-visible**; vocab
permutation+scaling is weaker than dense vocab masking.

## Required statement

This stage validates a local HuggingFace-style full CausalLM skeleton using
extracted weights and trusted input/output boundaries. It does not validate
production generation or claim semantic security.

## Limitations

Extracted-weight reference (not HF generate); tiny/random model by default;
no tokenizer/chat template; greedy only; no large-checkpoint benchmark; no
production inference; no semantic/formal/cryptographic security claim;
attention scores GPU-visible; vocab masking weaker than dense.

## Next stage

**Stage 7.3** — paper / implementation cleanup, or **Stage 7.4** — real local
tiny-checkpoint smoke test if a small local model is available.
