# Stage 7.0 — Full-Pipeline Cost, Leakage, and Ablation Evaluation

This stage adds **no new masking**. It evaluates the existing masked
CausalLM pipeline (Stages 6.4–6.8) honestly: an analytical cost model, CPU
wall-clock proxies, GPU-visible leakage surfaces, numerical leakage proxies,
and a safe/unsafe paper-claims split. No GPU, no transformers, no downloads.
No formal/cryptographic/semantic security is claimed.

## Variants

| id | variant | masking | handoff GEMM | output boundary |
|----|---------|---------|--------------|-----------------|
| A | `plain_synthetic` | none (baseline) | – | plain |
| B | `masked_same_residual_mask` | one shared `N` | none | GPU masked logits |
| C | `masked_per_layer_residual_mask` | `N_0..N_L` | yes | GPU masked logits |
| D | `masked_per_layer_no_vocab_scaling` | `N_0..N_L` | yes | GPU masked logits, perm-only vocab |
| E | `masked_per_layer_with_vocab_scaling` | `N_0..N_L` | yes | GPU masked logits, perm+scale (**preferred**) |
| F | `output_hidden_to_tee` (analytical) | `N_0..N_L` | yes | GPU returns masked hidden; TEE does norm+LM-head |
| G | `gpu_masked_lm_head` | `N_0..N_L` | yes | GPU masked logits (current boundary) |

## Cost model

FLOPs counted as `2·M·N·K` per matmul. Key formulas (per the inline code):

- **handoff GEMM** (per-layer masks only): `(L−1)·2·B·T·H·H` prefill +
  `(L−1)·decode_steps·2·B·H·H` decode; **0** for a shared mask. The skip-path
  change-of-basis cannot be folded offline (Stage 6.8 caveat).
- **LM head** (last token): `2·B·H·V` per produced token — on the GPU for
  `gpu_masked`/`plain`, on the **TEE** for `output_hidden_to_tee`.
- **logits recovery** (TEE, perm+scale): `O(B·V)` per token; zero for the
  TEE-LM-head and plain variants.
- **boundary calls**: `2 + 2·decode_steps` (prefill in/out + 2 per step); **0**
  for the plain baseline — no intermediate TEE calls inside the decoder.
- **transfer** (deployment, last token): TEE→GPU masked embeddings `B·T·H`;
  GPU→TEE masked logits `B·V` (`gpu_masked`) or masked hidden `B·H`
  (`output_hidden_to_tee`). With `V > H`, returning hidden is cheaper to
  transfer but moves the LM head onto the TEE.
- **KV cache**: `L·B·n_kv·(T+decode_steps)·head_dim·2·bytes`.

## Leakage surfaces

The masked variants give the GPU: masked embeddings, masked hidden states,
masked KV cache, and (except `output_hidden_to_tee`) masked logits — **never**
`input_ids`, plaintext embeddings, plaintext hidden, plaintext KV cache,
plaintext logits, or sampled token ids. **Attention scores/probabilities
remain GPU-visible** in the current masked-attention design (honest caveat).

## Leakage proxies (NOT security proofs)

- **Vocab-mask token-index linkability:** without the TEE secret, the GPU's
  argmax index aligns with the true token only at ≈chance under permutation
  (and is further perturbed by positive diagonal scaling), while the TEE
  recovers the top-1 exactly (rate 1.0). Permutation hides the index mapping;
  scaling perturbs magnitudes/ranking on the GPU side.
- **RoPE pair-norm linkability** (reused from Stage 6.4.1): rotation masks
  preserve per-pair norm (fully linkable), complex-scaling reduces
  cross-session correlation and NN matching — weaker than dense masks.

## Paper claims

**Safe:** no intermediate TEE calls inside decoder blocks; GPU sees masked
embeddings not raw ids; GPU sees masked logits not plaintext logits (preferred
boundary); synthetic full-skeleton correctness is verified; operator-compatible
masks reduce direct exposure while preserving operator-specific invariants.

**Unsafe wording to avoid (these must not be claimed; listed here only as an
audit of forbidden phrasing):** semantic security; cryptographic security;
"all intermediate states fully hidden"; "attention patterns hidden"; "output
semantics hidden"; dense-mask-equivalent security for RoPE/nonlinear islands.
We do not claim any of the above.

## Files

- `src/pllo/experiments/full_pipeline_cost_leakage.py`
- `scripts/run_full_pipeline_cost_leakage.py` →
  `outputs/full_pipeline_cost_leakage.{json,md,csv}`
- `tests/test_full_pipeline_cost_leakage.py` — 15 tests (no transformers).

## Limitations / next stage

Analytical cost model (not a hardware benchmark); wall-clock is a CPU float64
proxy; synthetic weights only; variant F is analytical-only; attention scores
GPU-visible; vocab/RoPE masks weaker than dense; output text unprotected once
returned.

**Stage 6.9** — real HF full-model skeleton / local tiny-checkpoint
integration, or **Stage 7.1** — paper-ready theorem + complexity-table
generation.
