# Generation Processor Coverage

_Stage 7.8c: verify that every standard logit processor (greedy / temperature / top-k / top-p / repetition penalty / stop / bad words / forced token) is exact under the masked-logits recovery boundary._

## Main Theorem

If z_recovered == z_plain at machine precision then any logit processor D that depends only on (z, generated history, processor params, trusted randomness rho) produces the same deterministic output / same sampling distribution as plaintext execution under the same rho.

## Configuration

| Field | Value |
|---|---|
| batch_size | 4 |
| vocab_size | 64 |
| n_trials | 8 |
| temperature | 0.7 |
| top_k | 8 |
| top_p | 0.9 |
| repetition_penalty | 1.2 |
| stop_token_id | 7 |
| bad_word_ids | [3, 5] |
| forced_token_id | 11 |
| seq_history_len | 4 |

## Recovery Bound

`logit_recovery_max_abs_error` = `3.109e-15`

## Processor Status

| Processor | Status |
|---|---|
| greedy | tested |
| temperature | tested |
| top_k | tested |
| top_p | tested |
| repetition_penalty | tested |
| stop_token | tested |
| bad_words | tested |
| forced_token | tested |
| beam_search | audit_only |
| grammar_constrained | audit_only |

## Privacy Flags

| Field | Value |
|---|---|
| processors_run_inside_trusted_side | True |
| accelerator_sees_processed_logits | False |
| accelerator_sees_sampling_candidates | False |

## Limitations

- CPU local emulation only.
- Beam search and grammar-constrained decoding are audit-only here; the main theorem says they apply, but they are not exercised end-to-end in this module.
- Output length / stop timing may still leak via observable generation length unless padded or hidden by batching policy -- THIS IS NOT IMPLEMENTED HERE.
- Trusted-side processor implementation MUST keep bad-word / forced-token / stop-token IDs trusted-side; exposing them in the accelerator transcript would leak the corresponding policies.
- Not formal cryptographic / semantic / differential-privacy security.
- No full Qwen / LLaMA deployment unless a real wrapper exists.

## Paper-Safe Wording

> Logit processors execute in the trusted side after logits recovery; since the recovery is exact at float64, every standard processor (greedy / temperature / top-k / top-p / repetition penalty / stop / bad words / forced token) produces an identical output under recovered and plain logits. Beam search and grammar-constrained decoding follow the same theorem and are listed as audit-only in this module.

## Unsafe Wording to Avoid

- Output length hidden.
- Stop timing side channel evaluated.
- Bad word list cryptographically hidden.
- Beam search fully implemented inside TEE.
- Real Qwen / LLaMA processors deployed.

