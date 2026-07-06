# AttackResult schema (v2)

`src/pllo/attacks/schema.py`. One `AttackResult` per (attack, target method,
setting). Plain dataclass, JSON-serialisable.

## Top-level fields

`attack_id`, `attack_name`, `attack_family` (`structural` / `optimization` /
`cryptanalysis` / `alignment` / `statistical` / `adapter` / `placeholder`),
`paper_source`, `implementation_level` (`full` / `best_effort` / `partial` /
`blocked`), `target_method`, `model_family`, `model_name_or_path`, `task_type`,
`threat_model` (`closed_model_no_weight_access` / `public_model` /
`known_plaintext` / `weight_leakage_worst_case` / `split_inference`), `status`
(`measured` / `failed` / `blocked` / `partial`), `error`, `notes`.

## Nested blocks

- `attacker_knowledge`: `has_algorithm` (Kerckhoffs, always true),
  `has_secret_keys` (always false), `has_model_weights`, `has_embedding_table`,
  `has_known_plaintext_pairs`, `has_intermediate_activations`, `has_logits`,
  `has_gradients`, `has_labels_or_posteriors`.
- `input_info`: `num_samples`, `batch_size`, `seq_len`, `hidden_size`,
  `vocab_size`, `dtype`, `device`.
- `attack_config`: `num_steps`, `num_restarts`, `lr`, `loss_type`, `topk`, `seed`.
- `metrics` (all default null): `attack_success_rate`,
  `one_minus_attack_success_rate`, `token_recovery_top{1,5,10,100}`,
  `sequence_exact_match`, `mean_token_accuracy`, `rouge_l_f1`, `edit_distance`,
  `cosine_similarity`, `mse`, `relative_l2_error`, `matrix_recovery_error`,
  `alignment_accuracy`, `prompt_recovery_success`,
  `permutation_recovery_accuracy`, `multiset_leakage_score`,
  `norm_profile_match_accuracy`, `distance_profile_match_accuracy`,
  `frequency_rank_correlation`.
- `runtime`: `wall_time_ms`, `num_steps_run`, `peak_memory_gb`, `requires_gpu`.

## Honesty invariants (`AttackResult.validate()`)

1. `status="blocked"` ⇒ a reason (`error`/`notes`) and no measured metric.
2. `status="measured"` ⇒ ≥1 non-null metric.
3. `attacker_knowledge.has_model_weights` ⇒ a white-box threat model
   (`weight_leakage_worst_case` / `split_inference` / `public_model`), never
   `closed_model_no_weight_access`.
4. `attacker_knowledge.has_known_plaintext_pairs` ⇒ `known_plaintext` (or worst case).
5. `implementation_level` ∈ {full, best_effort, partial, blocked}.

`validated()` raises on violation. `blocked_result(...)` and `failed_result(...)`
build compliant records; IO/table builders keep failed/blocked rows.

## IO
`result_io`: `write_jsonl`/`read_jsonl` (malformed lines skipped),
`results_to_csv`, `results_to_markdown`. `table_builder`:
`write_measured_table` / `write_qualitative_table`.
