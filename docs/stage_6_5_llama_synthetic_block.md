# Stage 6.5 — LLaMA/Qwen-like Synthetic Decoder Block

A correctness-first, CPU-only, synthetic probe that composes the existing
building blocks into one decoder layer and verifies it stays exactly
correct under operator-compatible masks. **No HF/ModelScope models, no
GPT-2 wrapper, no embeddings/LM-head/sampling, no NTK/YaRN RoPE scaling.
No formal, cryptographic, or semantic security is claimed.**

## Block (row-vector convention)

```
r1  = RMSNorm(x)
a   = RoPE-compatible GQA attention(r1)
x1  = x + a
r2  = RMSNorm(x1)
m   = SwiGLU(r2)
y   = x1 + m
```

## Masking design

A single **orthogonal residual mask** `n_res` (`[hidden, hidden]`) is shared
by the block input and output. Because RMSNorm's core (the normalize step)
preserves per-row L2 norm, an orthogonal right-multiply commutes with it
exactly:

```
rmsnorm_core(x @ n_res) == rmsnorm_core(x) @ n_res
```

The RMSNorm affine gain and the residual mask are folded into the following
projections, so the GPU-visible path operates entirely on the masked
residual stream `x_tilde = x @ n_res` and masked weights:

| weight | folded form |
|---|---|
| `Wq_tilde`   | `n_res^{-1} @ diag(rms1_w) @ Wq @ blockdiag(Mq_heads)` |
| `Wk_tilde`   | `n_res^{-1} @ diag(rms1_w) @ Wk @ blockdiag(Mk_heads)` |
| `Wv_tilde`   | `n_res^{-1} @ diag(rms1_w) @ Wv @ blockdiag(Mv_heads)` |
| `Wo_tilde`   | `blockdiag(Vinv_per_qhead) @ Wo @ n_res` |
| `Wgate_tilde`| `n_res^{-1} @ diag(rms2_w) @ Wgate[:, perm]` |
| `Wup_tilde`  | `n_res^{-1} @ diag(rms2_w) @ Wup[:, perm]` |
| `Wdown_tilde`| `Wdown[perm, :] @ n_res` |

- Attention masks `Mq/Mk/Mv` are the Stage 6.4.1 RoPE-compatible masks
  (default `pairwise_complex_scaling`; rotation remains a baseline). The
  score invariant `RoPE(Q M^{-T}) @ RoPE(K M)^T = RoPE(Q) @ RoPE(K)^T` and
  the value-mask/output-projection fold are inherited from Stage 6.4.1.
- The MLP uses the **paired-permutation** SwiGLU compatibility path (shared
  permutation over the intermediate dimension). The selector-lifted SwiGLU
  is **not** the default — its zero-row leakage caveat remains open.

End-to-end invariant, proven exactly and verified numerically:

```
y_tilde == y_plain @ n_res
```

## Files

- `src/pllo/ops/llama_synthetic_block.py` — config/weights, plain block,
  mask generation, weight folding, masked prefill + decode.
- `src/pllo/experiments/llama_synthetic_block_probe.py` — GQA + MHA cases,
  prefill + multi-step decode, per-stage metrics.
- `scripts/run_llama_synthetic_block_probe.py` — writes
  `outputs/llama_synthetic_block_probe.{json,md}`.
- `tests/test_llama_synthetic_block.py` — 20 tests (float64, atol/rtol 1e-8).

## Result

All per-stage max abs errors are at float64 machine precision (≤1.3e-14)
for both GQA (`nh=4, nkv=2`) and MHA (`nh=4, nkv=4`), prefill and
multi-step decode.

## Limitations

- Synthetic block only; not a real HF/ModelScope LLaMA/Qwen wrapper.
- No embedding, LM head, sampling, or tokenizer.
- No RoPE scaling variants (NTK/YaRN).
- RoPE-compatible masks preserve the per-pair partition and are weaker than
  dense masks; no formal/cryptographic/semantic security.

## Next stage

Stage 6.6 — real HF/ModelScope LLaMA/Qwen single-block wrapper.
