# Protected LoRA Training Protection

This document describes the LoRA **training**-stage protection experiments added
in `src/pllo/experiments/lora_training_protection.py` and
`src/pllo/protocol/lora_training_audit.py`. The paper claims protection for user
inputs, LoRA adapters, **and** LoRA training data; inference-only protection is
not sufficient, so this suite covers the full training stage: forward, backward,
optimizer update, and a training-stage security audit + attack baselines.

> **Scope / claims discipline (read first).**
> - **Fully implemented + tested:** masked LoRA *training* on (A) a synthetic
>   linear task and (B) a tiny transformer (LoRA on the attention V projection +
>   the MLP projection), for ranks 4/8/16. Protected training is mathematically
>   equivalent to plaintext training (errors ~1e-14, `allclose=True`).
> - **Feasibility probe only:** Qwen2.5-7B LoRA is a one/few-step
>   forward/backward/update probe (`run_qwen_probe`), **not** full fine-tuning. We
>   do **not** claim full 7B LoRA training is complete.
> - **Gated:** GPT-2 LoRA training runs on the GPU server with a local
>   checkpoint; locally it reports `available_not_run` / `skipped`.
> - Training-data protection is claimed **only** because the training-stage audit
>   covers training samples, labels, gradients, optimizer state, and adapter
>   updates (§2) — not merely inference activations.

## 1. Protection model (exact, training-stage)

For each LoRA-adapted linear layer

```
Y = X W + (alpha / r) X B A          (W frozen; B:[in,r], A:[r,out])
```

the **frozen base** matmul `X W` is offloaded to the untrusted GPU in masked
form; everything else stays in the trusted boundary:

| Tensor | Where | Notes |
|---|---|---|
| training data `X`, labels, token ids | **trusted** | never sent to GPU |
| LoRA `A`, `B`, `B@A` | **trusted** | LoRA term computed in the boundary |
| gradients `dA`, `dB`, optimizer state, updates | **trusted** | never sent to GPU |
| mask secrets `N`, `M`, per-sample scale `c` | **trusted** | |
| masked activation `X_tilde = c·(X N)` | → GPU | signed-perm `N` + per-sample scale |
| folded base weight `W_tilde = N⁻¹ W M` | → GPU (once) | masked artifact |
| masked base output `c·(X W) M` | ← GPU | trusted side recovers `X W` exactly |

`N` is an orthogonal signed permutation over in-features; `M` is a permutation +
positive diagonal over out-features; `c` is a per-sample positive scale that is
divided out at recovery (and randomises the GPU-visible activation norm). Because
all three are exactly invertible, the recovered base equals plaintext `X W` up to
floating point, so **the entire training trajectory matches plaintext LoRA
training to fp tolerance**.

### Correctness (measured, ranks 4/8/16)

`outputs/lora_training_protection_summary.{json,csv,md}`. Per training step we
compare `train_loss`, `eval_loss`, LoRA `A`/`B`, `delta_W=BA`, gradients `dA`/`dB`,
optimizer state, final logits, `top1_match_rate`, and the final task metric.
Reported aggregates: `max_abs_error`, `mean_abs_error`, `relative_l2_error`,
`cosine_similarity`, `allclose`, `loss_curve_distance`, `final_eval_delta`.

Result: every comparison is `allclose=True`, `top1_match_rate=1.0`, parameter /
gradient / optimizer errors ~1e-14–1e-16, `loss_curve_distance` ~1e-15, and the
final task metric is identical between plaintext and protected. Both tasks train
(loss drops by >10×), confirming the gradients are correct (not trivially zero).

## 2. Training-stage security audit

`audit_lora_training_trace` inspects the *exact* GPU trace recorded during
protected training and verifies the GPU received none of:

`gpu_visible_train_examples`, `gpu_visible_labels`, `gpu_visible_input_ids`,
`gpu_visible_lora_a`, `gpu_visible_lora_b`, `gpu_visible_delta_w`,
`gpu_visible_lora_grad_a`, `gpu_visible_lora_grad_b`,
`gpu_visible_optimizer_state`, `gpu_visible_adapter_update`,
`gpu_visible_plain_hidden`, `gpu_visible_recovered_logits` — plus
`leaked_secret_fields` (mask secrets) and a structural forbidden-field-name scan.
All are False / empty for a correct protected run (`audit_passed=True`). Tests
inject each kind of leak and confirm the audit catches it.

## 3. Attack evaluation (synthetic linear)

| Attack | Metric | Protected | Baseline |
|---|---|---|---|
| **Adapter recovery** | `adapter_recovery_relative_error`, `delta_w_recovery_relative_error` | ≈ **1.0** (no A/B information on the wire) | 0.0 if A/B were exposed |
| **Gradient inversion** | `gradient_inversion_reconstruction_error` | > 1.0 (only masked activations) | ≈ **0.0** (plaintext gradient recovers `X`) |
| **Membership** | `membership_attack_auc` / `_accuracy` | ≈ **0.5** (per-sample scale randomises norms) | ≈ 1.0 (true norms distinguish members) |

The adapter-recovery result is information-theoretic: the LoRA term is computed
inside the boundary, so the GPU trace contains *zero* information about `A`/`B`.

## 4. Integration with the TEE↔GPU protocol

This reuses the Stage 8.5 trust split (`docs/tee_gpu_protocol.md`): the trusted
boundary owns training data, token ids, adapter params, gradients, and optimizer
state; the untrusted GPU receives only masked operator payloads
(`LoRAMaskedMatmulRequest`) and returns masked outputs; the GPU path reports
`tee_used_on_gpu=False`; `audit_passed=True`. The boundary can run inside the
attested TDX guest (runtime-hash binding as in §6/§7 of the protocol doc).

## 5. Limitations

- **Mask strength.** `N`/`M` are orthogonal/positive (signed permutation + scale),
  weaker than dense masking; norms/geometry are otherwise preserved (the
  per-sample scale `c` specifically randomises activation norm for the membership
  attack). Consistent with the Stage 8.3 threat model.
- **Public base weights.** If the *plaintext* frozen base weight `W` is known to
  the untrusted side, the folded `W_tilde = N⁻¹ W M` plus knowledge of `W` could
  in principle constrain the masks. This is the documented folding caveat; the
  adapter-recovery result does not depend on it (A/B never reach the GPU).
- **Scale.** Full correctness is demonstrated at synthetic + tiny-transformer
  scale. GPT-2 and Qwen2.5-7B are gated; Qwen is a one/few-step feasibility probe.

## 6. Reproduce

```bash
python scripts/run_lora_training_protection_experiments.py --ranks 4,8,16
python -m pytest tests/test_lora_training_protection.py \
                 tests/test_lora_training_security_audit.py -q
```

## 7. Future work

- Bridge GPT-2 / Qwen2.5-7B masked base matmuls into this training boundary and
  run a real few-step 7B LoRA probe on the H800 (memory/latency,
  `tee_used_on_gpu=false`), then full fine-tuning if feasible.
- Dense (vs signed-permutation) masking for stronger formal guarantees.
- Cross-machine protected training over the HTTP protocol (`pllo.protocol.remote`).
