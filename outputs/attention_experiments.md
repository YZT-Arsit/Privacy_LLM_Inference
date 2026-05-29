# Privacy LLM Obfuscation — Attention Experiments (Stage 5.0)

Six invariants validated per `(batch_size, seq_len, decode_steps, use_pad)` cell:

1. `Q_tilde K_tilde^T ≈ Q K^T`
2. `softmax(Q_tilde K_tilde^T / sqrt(d)) ≈ softmax(Q K^T / sqrt(d))`
3. `A V_tilde ≈ (A V) N_V`
4. `AttnOut_tilde ≈ AttnOut N_res`
5. `K_cache_tilde_new ≈ K_cache_new N_K` (prefill + decode append)
6. `V_cache_tilde_new ≈ V_cache_new N_V`

All numbers are read from a fresh sweep over the registry in
`src.pllo.experiments.experiment_registry.ATTENTION_SWEEP`.

## Sweep coverage
| batch_size | seq_len | decode_steps | use_pad | full_allclose | cache_append_allclose |
|---|---|---|---|---|---|
| 1 | 4 | 1 | true | true | true |
| 1 | 4 | 1 | false | true | true |
| 1 | 4 | 2 | true | true | true |
| 1 | 4 | 2 | false | true | true |
| 1 | 4 | 4 | true | true | true |
| 1 | 4 | 4 | false | true | true |
| 1 | 8 | 1 | true | true | true |
| 1 | 8 | 1 | false | true | true |
| 1 | 8 | 2 | true | true | true |
| 1 | 8 | 2 | false | true | true |
| 1 | 8 | 4 | true | true | true |
| 1 | 8 | 4 | false | true | true |
| 1 | 16 | 1 | true | true | true |
| 1 | 16 | 1 | false | true | true |
| 1 | 16 | 2 | true | true | true |
| 1 | 16 | 2 | false | true | true |
| 1 | 16 | 4 | true | true | true |
| 1 | 16 | 4 | false | true | true |
| 2 | 4 | 1 | true | true | true |
| 2 | 4 | 1 | false | true | true |
| 2 | 4 | 2 | true | true | true |
| 2 | 4 | 2 | false | true | true |
| 2 | 4 | 4 | true | true | true |
| 2 | 4 | 4 | false | true | true |
| 2 | 8 | 1 | true | true | true |
| 2 | 8 | 1 | false | true | true |
| 2 | 8 | 2 | true | true | true |
| 2 | 8 | 2 | false | true | true |
| 2 | 8 | 4 | true | true | true |
| 2 | 8 | 4 | false | true | true |
| 2 | 16 | 1 | true | true | true |
| 2 | 16 | 1 | false | true | true |
| 2 | 16 | 2 | true | true | true |
| 2 | 16 | 2 | false | true | true |
| 2 | 16 | 4 | true | true | true |
| 2 | 16 | 4 | false | true | true |

## Headline invariants — worst cell per dimension
| metric | max over sweep | all cells allclose? |
|---|---|---|
| Q K^T (score) | 2.765e-10 | true |
| softmax probs | 0 | true |
| A V (per head) | 1.490e-08 | true |
| AttnOut (full path) | 9.219e-09 | true |
| Prefill K cache | 6.054e-09 | true |
| Prefill V cache | 1.118e-08 | true |
| Decode K cache append | 6.519e-09 | true |
| Decode V cache append | 4.098e-08 | true |
| N_Q N_K^T = I | 0 | — |

## use_pad = true vs false (worst cell of each)
| use_pad | max full_out_err | max score_err | max prefill_K_err | max decode_K_err | all_allclose |
|---|---|---|---|---|---|
| true | 9.219e-09 | 2.765e-10 | 6.054e-09 | 6.519e-09 | true |
| false | 2.619e-10 | 1.455e-10 | 2.328e-09 | 2.328e-09 | true |

## Reproducibility

```bash
python scripts/run_attention_experiments.py
```

Sweep registry: `batch_size ∈ {1, 2}`, `seq_len ∈ {4, 8, 16}`, `decode_steps ∈ {1, 2, 4}`, `use_pad ∈ {true, false}` → 36 cells.
