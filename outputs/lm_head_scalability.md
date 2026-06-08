# Scalable LM-Head Masking

_Stage 7.7a: compare scalable LM-head masking strategies under CPU local emulation; dense ``N_vocab`` is not feasible for real LLM vocab sizes._

## Configuration

| Field | Value |
|---|---|
| real_vocab_sizes | [97, 1024, 4096] |
| estimated_vocab_sizes | [16384, 50000] |
| hidden_size | 64 |
| batch_size | 2 |
| seq_len | 4 |
| block_size | 64 |
| topk | 8 |
| dense_max_real_v | 4096 |

## Real CPU Runs

| Mode | V | exactness | max_abs_error | greedy_match | memory_bytes_mask | online_recovery_ops_estimate | feasibility |
|---|---|---|---|---|---|---|---|
| `dense_vocab_mask_baseline` | 97 | exact | 3.442e-15 | 1.0 | 75272 | 75272 | feasible_only_for_small_V |
| `dense_vocab_mask_baseline` | 1024 | exact | 1.421e-14 | 1.0 | 8388608 | 8388608 | feasible_only_for_small_V |
| `dense_vocab_mask_baseline` | 4096 | exact | 2.442e-14 | 1.0 | 134217728 | 134217728 | feasible_only_for_small_V |
| `vocab_permutation_mask` | 97 | exact | 0.0 | 1.0 | 776 | 776 | scalable |
| `vocab_permutation_mask` | 1024 | exact | 0.0 | 1.0 | 8192 | 8192 | scalable |
| `vocab_permutation_mask` | 4096 | exact | 0.0 | 1.0 | 32768 | 32768 | scalable |
| `block_diagonal_vocab_mask` | 97 | exact | 2.220e-15 | 1.0 | 65536 | 65536 | scalable_with_block_size_tunable |
| `block_diagonal_vocab_mask` | 1024 | exact | 3.109e-15 | 1.0 | 524288 | 524288 | scalable_with_block_size_tunable |
| `block_diagonal_vocab_mask` | 4096 | exact | 3.775e-15 | 1.0 | 2097152 | 2097152 | scalable_with_block_size_tunable |
| `topk_trusted_recovery_mode` | 97 | exact_for_greedy_top1__not_full_softmax_unless_full_recovery | 0.0 | 1.0 | 776 | 776 | scalable_top1_only |
| `topk_trusted_recovery_mode` | 1024 | exact_for_greedy_top1__not_full_softmax_unless_full_recovery | 0.0 | 1.0 | 8192 | 8192 | scalable_top1_only |
| `topk_trusted_recovery_mode` | 4096 | exact_for_greedy_top1__not_full_softmax_unless_full_recovery | 0.0 | 1.0 | 32768 | 32768 | scalable_top1_only |

## Symbolic Estimates (No Dense Allocation)

| Mode | V | memory_bytes_mask | online_recovery_ops_estimate | feasibility |
|---|---|---|---|---|
| `dense_vocab_mask_baseline` | 16384 | 2147483648 | 2147483648 | infeasible_for_real_llm_vocab |
| `dense_vocab_mask_baseline` | 50000 | 20000000000 | 20000000000 | infeasible_for_real_llm_vocab |
| `vocab_permutation_mask` | 16384 | 131072 | 131072 | scalable |
| `vocab_permutation_mask` | 50000 | 400000 | 400000 | scalable |
| `block_diagonal_vocab_mask` | 16384 | 8388608 | 8388608 | scalable_with_block_size_tunable |
| `block_diagonal_vocab_mask` | 50000 | 25624576 | 25624576 | scalable_with_block_size_tunable |

## Limitations

- CPU local emulation only; no real TEE / GPU.
- Dense N_vocab not feasible for real LLM vocab sizes (V >= 16k); estimated symbolically only.
- Permutation mask preserves the multiset of logits; this is observable side information, not formal cryptographic security.
- Block-diagonal mask reveals block membership of each vocab index unless the block partition is itself permuted.
- topk_trusted_recovery_mode is exact for top-1 greedy decoding; for sampling that depends on the full distribution, full recovery must be performed before truncation.
- This is NOT formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> Dense orthogonal N_vocab is not scalable to real LLM vocab sizes. Permutation and block-diagonal masks scale but disclose either the sorted logit multiset or the block partition; we present them as scalable algebraic alternatives with explicit leakage notes, not formal cryptographic security.

## Unsafe Wording to Avoid

- Dense vocab mask is scalable.
- Permutation mask cryptographically hides logits.
- topk_trusted_recovery_mode preserves full softmax.
- This is formal cryptographic security.

