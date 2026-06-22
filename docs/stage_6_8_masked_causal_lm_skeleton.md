# Stage 6.8 — Multi-Layer Mask Handoff + Full Masked CausalLM Skeleton

Composes the Stage 6.7 trusted input boundary, `num_layers` Stage 6.5
synthetic LLaMA/Qwen-like decoder blocks, the Stage 6.7 masked-logits output
boundary, and a bounded greedy decode loop into one CPU correctness probe.
Synthetic weights only; no HF checkpoints, no transformers, no GPU, no GPT-2
wrappers. No formal/cryptographic/semantic security is claimed.

## Pipeline

```
TEE:  input_ids -> Embed -> X_plain ; X_tilde = (X_plain - T_in) @ N_0   (release X_tilde)
GPU:  for ell in 0..L-1:  H_{ell+1}_tilde = MaskedBlock_ell(H_ell_tilde) ; handoff to N_{ell+1}
      masked logits  L_tilde = L_plain @ M_vocab   (vocab perm+scale)
TEE:  recover L = L_tilde @ M_vocab^{-1} ; greedy next token ; mask its embedding ; repeat
```

Verified invariants (float64, machine precision):
- per layer input: `H_ell_tilde == H_ell_plain @ N_ell`
- final logits: `logits_tilde == logits_plain @ M_vocab`, `recovered == logits_plain`
- greedy generation: trusted-greedy-from-masked == greedy-from-plaintext.

## Per-layer mask handoff — honest note

A **pre-norm residual block's skip connection carries its input mask straight
to its output**, so a single residual block's output mask must equal its
input mask — a block cannot change the residual mask for free. We therefore:

- run each layer `ell` as a single-mask (`N_ell`) Stage-6.5 block, and
- realise the handoff `N_ell → N_{ell+1}` as **one orthogonal change-of-basis**
  `T_ell = N_ell⁻¹ @ N_{ell+1}` applied to the masked hidden state at the
  layer boundary.

`T_ell` is orthogonal (product of orthogonals). The non-skip terms are
offline-fusable into the preceding block's projection GEMMs; the **skip term
genuinely needs the transform**, so it is one `[H,H]` GEMM per boundary here
— *not zero*. Sharing one residual mask across all layers removes the
transition entirely; per-layer masks are used here to exercise the handoff.
We do **not** claim a zero-cost handoff.

## Input pad

The masked input is `(X_plain − T_in) @ N_0`. Because RMSNorm is **not**
shift-invariant, an additive embedding pad changes the modelled sequence, so
the plain reference is taken on the de-masked input `X_plain − T_in`. For
true model fidelity use `T_in = 0`; the pad is exercised here as a boundary
feature and the masking algebra remains exact either way.

## Files

- `src/pllo/ops/masked_causal_lm_skeleton.py`
- `src/pllo/experiments/masked_causal_lm_skeleton_probe.py`
- `scripts/run_masked_causal_lm_skeleton_probe.py` → `outputs/masked_causal_lm_skeleton_probe.{json,md}`
- `tests/test_masked_causal_lm_skeleton.py` — 20 tests (no transformers needed).

## Result

GQA (`nh=4, nkv=2`, `num_layers=3`) and the MHA / single-layer edge cases all
pass: per-layer handoff ≤~1.2e-14, final hidden ≤~1.2e-14, masked/recovered
logits ≤~8e-15, prefill greedy match `1.0`, decode token-match `1.0`.

## Limitations / next stage

Synthetic skeleton (not a real HF full model); no tokenizer/chat template;
greedy decode only; full-vocab LM-head cost not optimized; vocab
permutation+scaling weaker than dense; attention scores remain GPU-visible;
KV-cache masks reused within a session; output text semantics unprotected
once returned.

**Stage 6.9** — real HF full-model skeleton / local tiny-checkpoint
integration, or **Stage 7.0** — leakage/cost evaluation for the full pipeline.
