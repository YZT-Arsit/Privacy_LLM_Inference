# Stage 7.3 — Multi-Layer LoRA End-to-End Training

## 1. Experiment Scope

- num_layers=2, hidden_size=32, intermediate_size=64, vocab_size=128.
- batch_size=4, seq_len=8, alpha=1.0, true_rank=4, padded_rank=8.
- num_steps=3, optimizer=sgd, lr=0.01, use_pad=True, fresh_u_per_step=True, dummy_strategy='paired_cancellation_dummy', dtype=float64.
- Synthetic private data + frozen public base weights; no network access; no PEFT integration.

## 2. Tiny Multi-Layer LoRA Model

- num_layers: 2
- hidden_size: 32
- intermediate_size: 64
- modules_per_layer: 7
- total_lora_modules: 14
- lora_targets: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
- Each layer: q/k/v/o attention-like linears + SwiGLU MLP (gate, up, down). Base weights frozen; LoRA adapters trainable.

## 3. Multi-Layer Forward Correctness

| step | loss_plain | loss_masked | loss_diff | logits_err | forward_err | dummy_norm |
|------|------------|-------------|-----------|------------|-------------|------------|
| 0 | 1.099101e+01 | 1.099101e+01 | 0.000e+00 | 1.297e-13 | 9.814e-14 | 3.448e-15 |
| 1 | 1.084324e+01 | 1.084324e+01 | 1.066e-14 | 1.332e-13 | 7.727e-14 | 3.565e-15 |
| 2 | 1.070102e+01 | 1.070102e+01 | 8.882e-15 | 9.459e-14 | 6.473e-14 | 3.576e-15 |

- max_loss_diff: 1.066e-14
- max_forward_err: 9.814e-14
- max_dummy_contribution_norm: 3.576e-15

## 4. Multi-Layer Masked Backward Correctness

| step | grad_A_real_err | grad_B_real_err | adapter_A_update_err | adapter_B_update_err |
|------|------------------|------------------|----------------------|----------------------|
| 0 | 1.093e-15 | 3.497e-15 | 5.551e-17 | 3.469e-17 |
| 1 | 1.408e-15 | 3.358e-15 | 5.551e-17 | 4.510e-17 |
| 2 | 1.845e-15 | 2.887e-15 | 5.551e-17 | 6.939e-17 |

- max grad_A real err: 1.845e-15
- max grad_B real err: 3.497e-15
- max update_A err: 5.551e-17
- max update_B err: 6.939e-17
- allclose: **True**

## 5. Rank Padding Across Layers

- dummy_strategy_requested: `paired_cancellation_dummy`
- true_rank: 4
- padded_rank: 8
- lora_targets: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
- num_lora_modules: 14
- true_rank_hidden_from_shape: **True**
- padded_rank_visible: **True**

## 6. Optimizer Handling

- location: **trusted**
- optimizer: sgd
- lr: 0.01
- any_optimizer_state_contains_dummy: **False**
- any_dummy_update_applied: **False**
- note: Optimizer state and trainable adapters are sized to true_rank for every LoRA module. The dummy slice is re-sampled per step and never enters the optimizer.

## 7. Per-Layer Metrics

| layer | module | true_r | pad_r | forward_err | grad_A_err | grad_B_err | update_A_err | update_B_err | visible_rank | hidden |
|-------|--------|--------|-------|-------------|------------|------------|--------------|--------------|--------------|--------|
| 0 | q_proj | 4 | 8 | 8.99e-15 | 5.16e-16 | 2.91e-15 | 5.55e-17 | 5.55e-17 | 8 | True |
| 0 | k_proj | 4 | 8 | 6.66e-15 | 6.77e-16 | 2.86e-15 | 5.55e-17 | 4.86e-17 | 8 | True |
| 0 | v_proj | 4 | 8 | 1.05e-14 | 4.69e-16 | 2.44e-15 | 5.55e-17 | 3.04e-17 | 8 | True |
| 0 | o_proj | 4 | 8 | 9.71e-15 | 1.84e-15 | 2.58e-15 | 5.55e-17 | 3.84e-17 | 8 | True |
| 0 | gate_proj | 4 | 8 | 1.01e-14 | 6.14e-16 | 2.41e-15 | 5.55e-17 | 3.38e-17 | 8 | True |
| 0 | up_proj | 4 | 8 | 1.11e-14 | 1.36e-15 | 3.39e-15 | 5.55e-17 | 6.94e-17 | 8 | True |
| 0 | down_proj | 4 | 8 | 1.66e-14 | 1.41e-15 | 3.50e-15 | 5.55e-17 | 4.68e-17 | 8 | True |
| 1 | q_proj | 4 | 8 | 2.58e-14 | 4.58e-16 | 2.71e-15 | 5.55e-17 | 4.60e-17 | 8 | True |
| 1 | k_proj | 4 | 8 | 2.22e-14 | 6.71e-16 | 3.22e-15 | 5.55e-17 | 5.77e-17 | 8 | True |
| 1 | v_proj | 4 | 8 | 2.49e-14 | 1.14e-15 | 2.10e-15 | 2.78e-17 | 3.35e-17 | 8 | True |
| 1 | o_proj | 4 | 8 | 5.42e-14 | 7.43e-16 | 2.80e-15 | 5.55e-17 | 4.51e-17 | 8 | True |
| 1 | gate_proj | 4 | 8 | 5.95e-14 | 6.26e-16 | 2.00e-15 | 5.55e-17 | 3.47e-17 | 8 | True |
| 1 | up_proj | 4 | 8 | 4.49e-14 | 5.18e-16 | 1.78e-15 | 2.78e-17 | 2.34e-17 | 8 | True |
| 1 | down_proj | 4 | 8 | 9.81e-14 | 3.43e-16 | 1.79e-15 | 2.78e-17 | 2.69e-17 | 8 | True |

## 8. Limitations

- This is a synthetic multi-layer LoRA training prototype over a tiny tile, not a full Qwen / TinyLlama / LLaMA fine-tuning.
- Loss + optimizer remain trusted-side (Stage 7.1 contract).
- Optimizer state is sized to true_rank, never padded_rank, for every LoRA module.
- Padded rank r_pad remains visible from tensor shape; only true_rank is hidden from shape-level leakage.
- No PEFT / DeepSpeed / vLLM / FlashAttention integration.
- No real TEE training; security_profile remains 'proxy-evaluated, not formal'.
- No distributed training.
- Attention is a simple scaled-dot-product proxy, not a correctness benchmark for full GQA / RoPE / KV-cache.
- Adapter is NEVER merged into the public base weight W.
- Reports publish summary metrics + fingerprints; private data, raw adapters, raw gradients, optimizer state, and dense masks are never emitted.

## 9. Next Stage Plan

- Stage 7.4 — stronger dummy distributions / spectral-rank hardening.
- Stage 7.x — real Qwen / TinyLlama / LLaMA LoRA fine-tuning behind a real TEE.
