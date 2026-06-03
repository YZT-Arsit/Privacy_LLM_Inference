# Paper Toy Task Workload (CPU only)

_This is a CPU local-emulation prototype on deterministic synthetic task-like inputs. No external dataset, no tokenizer, no network, not a real Qwen / TinyLlama / LLaMA fine-tune._

| task_name | num_samples | num_train_steps | train_loss_plain | train_loss_masked | loss_diff | accuracy_plain | accuracy_masked | accuracy_diff | logits_max_abs_error | token_match_rate | allclose | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| token_parity_classification | 128 | 20 | 0.6901150550410321 | 0.6901150550410321 | 0.0 | 0.5390625 | 0.5390625 | 0.0 | 9.992007221626409e-16 | 1.0 | True | 23.32412500004466 |
| first_last_token_relation | 128 | 20 | 0.688203308355396 | 0.6882033083553959 | 1.1102230246251565e-16 | 0.53125 | 0.53125 | 0.0 | 7.042977312465837e-16 | 1.0 | True | 12.599417000046742 |
| next_token_toy_lm | 128 | 20 | 4.8663778912838485 | 4.866377891283849 | 8.881784197001252e-16 | 0.015625 | 0.015625 | 0.0 | 3.3931191190106347e-15 | 1.0 | True | 13.031958000055965 |

## Limitations

- Task-like inputs are deterministic synthetic sequences; no external dataset, no tokenizer.
- Targets are deterministic functions of input ids and do NOT reflect any real-world label distribution.
- The toy model is a tiny LoRA-augmented decoder stack; not a full Qwen / TinyLlama / LLaMA fine-tune.
- Loss, optimizer, and backward remain trusted-side (Stage 7.0 / 7.1 contract).
- Local CPU runtime only; not real TEE wall-time and not GPU throughput.
- No formal / cryptographic / semantic security is claimed.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- Adapter is NEVER merged into the public base weight W (Stage 7.0 contract).
- Reports publish summary metrics only; raw tensors / masks / adapters / gradients / input ids are never emitted.
